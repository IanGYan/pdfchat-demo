"""
知识库管理模块，负责构建和管理PDF知识库
"""

import logging
import os
from typing import List, Dict, Any, Optional

from llama_index.core.schema import Document, BaseNode
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.document_processing.pdf_loader import PDFProcessor
from app.database.pgvector_store import PGVectorManager
from app.core.retriever import RAGQueryEngine
from app.utils.config import Config

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """PDF知识库管理类"""
    
    def __init__(self, 
                 table_name: str = "pdf_documents",
                 rebuild: bool = False):
        """
        初始化知识库
        
        Args:
            table_name: 数据库表名
            rebuild: 是否重建知识库
        """
        # 验证配置
        Config.validate_config()
        
        self.table_name = table_name
        self.pdf_processor = PDFProcessor()
        self.pgvector_manager = PGVectorManager(table_name=table_name)
        
        # 初始化OpenAI LLM
        self.llm = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL_NAME,
            temperature=0.1
        )
        
        # 初始化OpenAI嵌入模型
        self.embedding_model = OpenAIEmbedding(
            api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_EMBEDDING_MODEL_NAME,
            embed_batch_size=10
        )
        
        logger.info(f"使用OpenAI嵌入模型: {Config.OPENAI_EMBEDDING_MODEL_NAME}, 维度: {Config.OPENAI_EMBEDDING_MODEL_DIM}")
        
        # 创建服务上下文，指定使用的LLM和嵌入模型
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model
        )
        
        # 初始化索引和查询引擎
        self.index = None
        self.query_engine = None
        
        # 如果需要重建，清空现有数据
        if rebuild:
            self.clear_knowledge_base()
            
        # 初始化向量存储
        self.pgvector_manager.initialize()
        
        # 尝试加载现有索引
        self.index = self.pgvector_manager.get_index()
    
    def clear_knowledge_base(self) -> None:
        """清空知识库"""
        logger.info("清空知识库...")
        self.pgvector_manager.clear_data()
        self.index = None
        self.query_engine = None
    
    def add_pdf_document(self, file_path: str) -> None:
        """
        添加单个PDF文档到知识库
        
        Args:
            file_path: PDF文件路径
        """
        try:
            logger.info(f"正在添加PDF文档: {file_path}")
            
            # 加载文档
            documents = self.pdf_processor.load_documents(file_path)
            
            # 处理文档
            self._process_and_index_documents(documents)
            
            logger.info(f"成功添加PDF文档: {file_path}")
        except Exception as e:
            logger.error(f"添加PDF文档失败: {str(e)}")
            raise
    
    def add_pdf_documents_from_dir(self, dir_path: str) -> None:
        """
        从目录添加多个PDF文档到知识库
        
        Args:
            dir_path: 包含PDF文件的目录路径
        """
        try:
            logger.info(f"正在从目录添加PDF文档: {dir_path}")
            
            # 加载目录中的所有文档
            documents = self.pdf_processor.load_documents_from_dir(dir_path)
            
            if not documents:
                logger.warning(f"目录 {dir_path} 中没有找到PDF文件")
                return
            
            # 处理文档
            self._process_and_index_documents(documents)
            
            logger.info(f"成功从目录 {dir_path} 添加PDF文档")
        except Exception as e:
            logger.error(f"从目录添加PDF文档失败: {str(e)}")
            raise
    
    def _process_and_index_documents(self, documents: List[Document]) -> None:
        """
        处理文档并建立索引
        
        Args:
            documents: 文档列表
        """
        # 处理文档，进行分块
        nodes = self.pdf_processor.process_documents(documents)
        
        if not nodes:
            logger.warning("没有生成任何文档节点，跳过索引创建")
            return
            
        # 清理节点元数据，确保没有'None'字符串或其他可能导致类型转换错误的值
        cleaned_nodes = []
        for node in nodes:
            if node.metadata:
                # 创建新的元数据字典
                cleaned_metadata = {}
                for key, value in node.metadata.items():
                    # 处理None值和'None'字符串
                    if value is None or (isinstance(value, str) and value.lower() == 'none'):
                        cleaned_metadata[key] = ''  # 使用空字符串替代
                    elif key == 'page' or key == 'page_number':
                        # 特殊处理页码字段，确保是整数
                        try:
                            page_num = int(value) if value and value != 'None' else 0
                            cleaned_metadata[key] = page_num
                        except (ValueError, TypeError):
                            logger.warning(f"无法将页码'{value}'转换为整数，使用默认值0")
                            cleaned_metadata[key] = 0
                    else:
                        cleaned_metadata[key] = value
                        
                # 更新节点的元数据
                node.metadata = cleaned_metadata
                
            # 确保节点的其他属性也没有None字符串
            if hasattr(node, 'node_id') and (node.node_id is None or node.node_id == 'None'):
                node.node_id = f"node_{len(cleaned_nodes)}"
                
            # 添加到清理后的节点列表
            cleaned_nodes.append(node)
            
        logger.info(f"清理后的节点数量: {len(cleaned_nodes)}")
        
        # 创建或更新索引，显式传递服务上下文
        self.index = self.pgvector_manager.create_index_from_nodes(
            nodes=cleaned_nodes,
            service_context=self.service_context
        )
        
        # 初始化查询引擎
        self._initialize_query_engine()
    
    def _initialize_query_engine(self) -> None:
        """初始化查询引擎"""
        if not self.index:
            logger.warning("索引不存在，无法初始化查询引擎")
            return
            
        self.query_engine = RAGQueryEngine(
            vector_index=self.index,
            llm=self.llm,
            embed_model=self.embedding_model,
            service_context=self.service_context
        )
        
        logger.info("查询引擎初始化完成")
    
    def query(self, query_str: str) -> Dict[str, Any]:
        """
        查询知识库
        
        Args:
            query_str: 查询字符串
            
        Returns:
            包含回答和引用的字典
        """
        if not self.query_engine:
            if not self.index:
                raise ValueError("知识库索引不存在，请先添加文档")
            self._initialize_query_engine()
        
        # 执行查询
        return self.query_engine.query(query_str)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取知识库状态
        
        Returns:
            包含知识库状态信息的字典
        """
        # 获取文档数量
        doc_count = self.pgvector_manager.get_document_count()
        
        return {
            "initialized": self.index is not None,
            "document_count": doc_count,
            "table_name": self.table_name,
            "embedding_model": Config.OPENAI_EMBEDDING_MODEL_NAME,
            "embedding_dim": Config.OPENAI_EMBEDDING_MODEL_DIM,
            "llm_model": Config.OPENAI_MODEL_NAME
        } 