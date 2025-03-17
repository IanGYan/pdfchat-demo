"""
PostgreSQL 向量存储模块，负责文档向量的存储和检索
"""

from typing import List, Optional
import logging
import psycopg2

from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core import VectorStoreIndex
from sqlalchemy import make_url

from app.utils.config import Config

logger = logging.getLogger(__name__)

class PGVectorManager:
    """PostgreSQL向量存储管理器"""
    
    def __init__(self, table_name: str = "pdf_documents"):
        """
        初始化PostgreSQL向量管理器
        
        Args:
            table_name: 存储表名
        """
        # 原始表名（不带前缀）
        self.base_table_name = table_name
        # 实际表名（LlamaIndex会自动添加"data_"前缀）
        self.actual_table_name = f"data_{table_name}"
        self.connection_string = Config.PGVECTOR_URL
        if not self.connection_string:
            raise ValueError("未设置PostgreSQL连接URL，请检查环境变量PGVECTOR_URL")
        
        # 嵌入维度
        self.embed_dim = Config.OPENAI_EMBEDDING_MODEL_DIM
        logger.info(f"使用嵌入向量维度: {self.embed_dim}")
        
        self.vector_store = None
        self.storage_context = None
        self.index = None
        # 保存一个单独的数据库连接用于辅助操作
        self.conn = None
        # 混合搜索设置
        self.hybrid_search = True
        # 文本搜索配置，使用PostgreSQL默认支持的'simple'
        self.text_search_config = "simple"
        
    def initialize(self) -> None:
        """初始化向量存储"""
        try:
            # 确保嵌入维度是有效的整数
            if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
                logger.warning(f"无效的嵌入维度: {self.embed_dim}，使用默认值1536")
                self.embed_dim = 1536
                
            logger.info("尝试初始化PostgreSQL向量存储...")
            
            # 创建一个单独的数据库连接用于辅助操作
            try:
                self.conn = psycopg2.connect(self.connection_string)
                # 设置自动提交为True
                self.conn.autocommit = True
                logger.info("成功创建辅助数据库连接，已启用自动提交")
                
                # 首先检查pgvector扩展是否已安装
                try:
                    with self.conn.cursor() as cursor:
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        logger.info("确保pgvector扩展已安装")
                except Exception as e:
                    logger.warning(f"创建pgvector扩展失败: {str(e)}，可能需要手动安装")
            except Exception as conn_err:
                logger.warning(f"创建辅助数据库连接失败: {str(conn_err)}，某些功能可能不可用")
                self.conn = None

            # 检查表是否存在，如果不存在则创建表
            if self.conn:
                try:
                    with self.conn.cursor() as cursor:
                        # 检查实际表名（带data_前缀）
                        cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{self.actual_table_name}')")
                        table_exists = cursor.fetchone()[0]
                        
                        if not table_exists:
                            logger.info(f"表 {self.actual_table_name} 不存在，尝试创建...")
                            # 创建表的SQL，使用simple文本搜索配置
                            create_table_sql = f"""
                            CREATE TABLE IF NOT EXISTS {self.actual_table_name} (
                                id BIGSERIAL NOT NULL,
                                text VARCHAR NOT NULL,
                                metadata_ JSON,
                                node_id VARCHAR,
                                embedding VECTOR({self.embed_dim}),
                                text_search_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('{self.text_search_config}', text)) STORED,
                                PRIMARY KEY (id)
                            );
                            """
                            cursor.execute(create_table_sql)
                            
                            # 创建索引
                            create_index_sql = f"""
                            CREATE INDEX IF NOT EXISTS {self.actual_table_name}_text_search_idx 
                            ON {self.actual_table_name} USING GIN (text_search_tsv);
                            
                            CREATE INDEX IF NOT EXISTS {self.actual_table_name}_embedding_idx
                            ON {self.actual_table_name} USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                            """
                            cursor.execute(create_index_sql)
                            
                            logger.info(f"表 {self.actual_table_name} 创建成功，并添加了必要的索引")
                        else:
                            # 检查text_search_tsv列是否存在
                            cursor.execute(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = '{self.actual_table_name}' AND column_name = 'text_search_tsv'
                            )
                            """)
                            tsvector_exists = cursor.fetchone()[0]
                            
                            if not tsvector_exists:
                                logger.info(f"表 {self.actual_table_name} 缺少text_search_tsv列，添加该列...")
                                alter_table_sql = f"""
                                ALTER TABLE {self.actual_table_name} 
                                ADD COLUMN text_search_tsv TSVECTOR 
                                GENERATED ALWAYS AS (to_tsvector('{self.text_search_config}', text)) STORED;
                                
                                CREATE INDEX IF NOT EXISTS {self.actual_table_name}_text_search_idx 
                                ON {self.actual_table_name} USING GIN (text_search_tsv);
                                """
                                cursor.execute(alter_table_sql)
                                logger.info(f"成功为表 {self.actual_table_name} 添加text_search_tsv列和索引")
                            
                            logger.info(f"表 {self.actual_table_name} 已存在并验证了结构")
                except Exception as e:
                    logger.error(f"检查和创建表失败: {str(e)}")
                    raise
            else:
                logger.warning("无法创建辅助数据库连接，某些功能可能不可用")
                raise

            # 解析连接字符串获取参数
            url = make_url(self.connection_string)
            
            # 创建PGVectorStore，使用hybrid_search=True
            try:
                logger.info(f"使用from_params方法创建PGVectorStore (hybrid_search={self.hybrid_search})...")
                
                self.vector_store = PGVectorStore.from_params(
                    host=url.host,
                    port=url.port or 5432,
                    database=url.database,
                    user=url.username,
                    password=url.password,
                    table_name=self.base_table_name,  # 使用原始表名，PGVectorStore会自动添加前缀
                    embed_dim=self.embed_dim,
                    hybrid_search=self.hybrid_search,
                    text_search_config=self.text_search_config,  # 使用simple文本搜索配置
                )
                logger.info("成功创建PGVectorStore")
            except Exception as e:
                logger.warning(f"使用from_params方法创建PGVectorStore失败: {str(e)}")
                raise
            
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            logger.info(f"成功初始化PostgreSQL向量存储，基础表名: {self.base_table_name}，实际表名: {self.actual_table_name}，向量维度: {self.embed_dim}，混合搜索: {self.hybrid_search}，文本搜索配置: {self.text_search_config}")
        except Exception as e:
            logger.error(f"初始化PostgreSQL向量存储失败: {str(e)}")
            raise
    
    def create_index_from_nodes(self, nodes: List[BaseNode], service_context=None) -> VectorStoreIndex:
        """
        从节点列表创建索引
        
        Args:
            nodes: 文档节点列表
            service_context: 服务上下文
            
        Returns:
            创建的索引对象
        """
        if not self.storage_context:
            self.initialize()
        
        try:
            # 创建索引 (自动提交模式，不需要显式commit)
            if service_context:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context,
                    service_context=service_context
                )
            else:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context
                )
                
            logger.info(f"成功创建索引，包含 {len(nodes)} 个文档节点")
            return self.index
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """获取当前索引对象"""
        return self.index
    
    def clear_data(self) -> None:
        """清空向量存储中的数据"""
        if self.vector_store:
            try:
                self.vector_store.delete(delete_all=True)
                logger.info(f"已清空表 {self.actual_table_name} 中的所有数据")
            except Exception as e:
                logger.error(f"清空数据失败: {str(e)}")
                # 尝试使用直接SQL语句清空数据
                if self.conn and not self.conn.closed:
                    try:
                        with self.conn.cursor() as cursor:
                            cursor.execute(f"TRUNCATE TABLE {self.actual_table_name}")
                            logger.info(f"通过SQL语句清空表 {self.actual_table_name} 成功")
                    except Exception as sql_err:
                        logger.error(f"通过SQL语句清空数据失败: {str(sql_err)}")
    
    def get_document_count(self) -> int:
        """获取存储的文档数量"""
        if not self.vector_store:
            return 0
            
        try:
            # 使用独立的数据库连接查询
            if self.conn and not self.conn.closed:
                try:
                    with self.conn.cursor() as cursor:
                        # 先检查表是否存在
                        cursor.execute(
                            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                            (self.actual_table_name,)
                        )
                        table_exists = cursor.fetchone()[0]
                        
                        if not table_exists:
                            logger.info(f"表 {self.actual_table_name} 不存在，可能是首次运行")
                            return 0
                            
                        # 表存在，查询数量
                        cursor.execute(f"SELECT COUNT(*) FROM {self.actual_table_name}")
                        count = cursor.fetchone()[0]
                        return count
                except Exception as sql_err:
                    logger.error(f"通过SQL查询文档数量失败: {str(sql_err)}")
                    return 0
            
            # 如果没有独立连接或查询失败，尝试使用向量存储的API
            try:
                # 注意：这部分取决于PGVectorStore的实现，可能需要调整
                # 由于PGVectorStore没有提供直接的计数方法，这里返回0
                logger.warning("无法直接从PGVectorStore获取文档数量")
                return 0
            except Exception as api_err:
                logger.error(f"通过API获取文档数量失败: {str(api_err)}")
                return 0
                
        except Exception as e:
            logger.error(f"获取文档数量失败: {str(e)}")
            return 0
            
    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
                logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {str(e)}") 