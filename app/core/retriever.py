"""
检索引擎模块，简化实现，使用PGVectorStore内置的混合检索功能
"""

import logging
from typing import Dict, Any, Optional

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import MetadataFilters

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """RAG查询引擎，负责处理用户查询并生成回答"""
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM,
        embed_model: BaseEmbedding,
        service_context: Optional[ServiceContext] = None,
        similarity_top_k: int = 4,
        similarity_cutoff: float = 0.7,
        vector_store_query_mode: str = "hybrid",
        pgvector_options: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化RAG查询引擎
        
        Args:
            vector_index: 向量索引
            llm: 大语言模型
            embed_model: 嵌入模型
            service_context: 服务上下文
            similarity_top_k: 检索返回结果数量
            similarity_cutoff: 相似度阈值，低于此值的结果将被过滤
            vector_store_query_mode: 向量存储查询模式，可选值: "default", "hybrid", "sparse"
            pgvector_options: PostgreSQL向量存储特定的选项，如ivfflat_probes, hnsw_ef_search等
        """
        self.vector_index = vector_index
        self.llm = llm
        self.embed_model = embed_model
        self.service_context = service_context
        
        # 默认的PostgreSQL向量存储选项
        default_pgvector_options = {
            "ivfflat_probes": 10,  # 增加搜索探针数量以提高准确性
            "hnsw_ef_search": 100  # 增加搜索候选项以提高准确性
        }
        
        # 合并用户提供的选项
        self.pgvector_options = {**default_pgvector_options, **(pgvector_options or {})}
        
        logger.info(f"使用向量存储查询模式: {vector_store_query_mode}")
        logger.info(f"PGVector选项: {self.pgvector_options}")
        
        # 直接使用vector_index.as_query_engine创建查询引擎
        try:
            # 创建查询引擎，利用PGVectorStore内置的混合检索功能
            self.query_engine = vector_index.as_query_engine(
                service_context=service_context,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=vector_store_query_mode,  # 使用"hybrid"模式
                vector_store_kwargs=self.pgvector_options,  # 传递PGVector特定选项
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
                ]
            )
            logger.info(f"成功创建查询引擎，使用{vector_store_query_mode}模式")
        except Exception as e:
            logger.error(f"创建查询引擎失败: {str(e)}")
            raise
    
    def apply_filter(self, metadata_filters: MetadataFilters) -> None:
        """
        应用元数据过滤条件
        
        Args:
            metadata_filters: 元数据过滤条件
        """
        # 根据query_engine的具体实现更新过滤条件
        if hasattr(self.query_engine, "retriever") and hasattr(self.query_engine.retriever, "filters"):
            self.query_engine.retriever.filters = metadata_filters
            logger.info(f"应用元数据过滤条件: {metadata_filters}")
        else:
            logger.warning("无法应用元数据过滤条件，查询引擎不支持此操作")
    
    def query(self, query_str: str, filters: Optional[MetadataFilters] = None) -> Dict[str, Any]:
        """
        处理用户查询并生成回答
        
        Args:
            query_str: 用户查询字符串
            filters: 查询特定的元数据过滤条件
            
        Returns:
            包含回答和引用的字典
        """
        try:
            logger.info(f"处理用户查询: {query_str}")
            
            # 临时应用过滤条件（如果有）
            original_filters = None
            if filters and hasattr(self.query_engine, "retriever") and hasattr(self.query_engine.retriever, "filters"):
                original_filters = self.query_engine.retriever.filters
                self.query_engine.retriever.filters = filters
            
            # 执行查询
            response = self.query_engine.query(query_str)
            
            # 恢复原始过滤条件
            if filters and original_filters is not None and hasattr(self.query_engine, "retriever"):
                self.query_engine.retriever.filters = original_filters
            
            # 提取引用信息
            source_nodes = getattr(response, "source_nodes", [])
            
            # 格式化引用信息
            formatted_citations = []
            for i, node in enumerate(source_nodes):
                citation = {
                    "content": node.node.get_content(),
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                formatted_citations.append(citation)
            
            return {
                "response": str(response),
                "citations": formatted_citations,
            }
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
            raise 