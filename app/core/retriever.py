"""
检索引擎模块，简化实现，使用PGVectorStore内置的混合检索功能
"""

import logging
from typing import Dict, Any, Optional, List

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter, MetadataFilter

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
        similarity_cutoff: float = 0.5,  # 降低相似度阈值，提高检索成功率
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
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        self.vector_store_query_mode = vector_store_query_mode
        
        # 默认的PostgreSQL向量存储选项
        default_pgvector_options = {
            "ivfflat_probes": 20,  # 增加搜索探针数量以提高召回率
            "hnsw_ef_search": 200,  # 增加搜索候选项以提高召回率
            "alpha": 0.75,  # hybrid搜索中向量搜索的权重
        }
        
        # 合并用户提供的选项
        self.pgvector_options = {**default_pgvector_options, **(pgvector_options or {})}
        
        logger.info(f"使用向量存储查询模式: {vector_store_query_mode}")
        logger.info(f"PGVector选项: {self.pgvector_options}")
        logger.info(f"相似度阈值: {self.similarity_cutoff}, 返回结果数量: {self.similarity_top_k}")
        
        # 检查索引是否存在
        if vector_index is None:
            raise ValueError("向量索引为空，请确保已正确创建索引")
            
        # 查看索引中的节点数量
        try:
            if hasattr(vector_index, "_vector_store") and hasattr(vector_index._vector_store, "_collection"):
                logger.info(f"向量存储类型: {type(vector_index._vector_store).__name__}")
        except Exception as e:
            logger.warning(f"无法获取向量存储信息: {str(e)}")
        
        # 直接使用vector_index.as_query_engine创建查询引擎
        try:
            # 创建查询引擎，利用PGVectorStore内置的混合检索功能
            logger.info(f"创建查询引擎，使用{vector_store_query_mode}模式...")
            
            # 禁用SimilarityPostprocessor以获取更多结果
            self.query_engine = vector_index.as_query_engine(
                service_context=service_context,
                similarity_top_k=similarity_top_k * 2,  # 检索更多结果，稍后手动过滤
                vector_store_query_mode=vector_store_query_mode,
                vector_store_kwargs=self.pgvector_options,
            )
            logger.info(f"成功创建查询引擎，使用{vector_store_query_mode}模式")
        except Exception as e:
            logger.error(f"创建查询引擎失败: {str(e)}")
            # 尝试回退到默认检索模式
            try:
                logger.info("尝试使用默认检索模式创建查询引擎...")
                self.query_engine = vector_index.as_query_engine(
                    service_context=service_context,
                    similarity_top_k=similarity_top_k,
                )
                logger.info("成功使用默认检索模式创建查询引擎")
            except Exception as fallback_e:
                logger.error(f"使用默认模式创建查询引擎也失败: {str(fallback_e)}")
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
    
    def _format_debug_info(self, source_nodes: List[NodeWithScore]) -> str:
        """格式化调试信息，显示检索到的内容和分数"""
        if not source_nodes:
            return "未检索到任何内容"
            
        debug_info = "\n检索到的内容:\n"
        for i, node in enumerate(source_nodes):
            debug_info += f"[{i+1}] 分数: {node.score:.4f}, 内容: {node.node.get_content()[:100]}...\n"
        return debug_info
    
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
                logger.info(f"应用临时过滤条件: {filters}")
            
            # 尝试直接查询
            try:
                response = self.query_engine.query(query_str)
                logger.info("查询执行成功")
            except Exception as query_err:
                logger.error(f"直接查询失败: {str(query_err)}，尝试备用方法")
                
                # 尝试备用方法：直接使用检索器
                if hasattr(self.query_engine, "retriever"):
                    try:
                        logger.info("使用备用方法：直接调用检索器")
                        retriever = self.query_engine.retriever
                        nodes = retriever.retrieve(query_str)
                        
                        if not nodes:
                            logger.warning("检索器未返回任何结果")
                            return {
                                "response": "对不起，我没有找到相关的信息。",
                                "citations": [],
                            }
                            
                        # 使用LLM生成回答
                        from llama_index.core.response_synthesizers import get_response_synthesizer
                        response_synthesizer = get_response_synthesizer(
                            service_context=self.service_context,
                        )
                        response = response_synthesizer.synthesize(
                            query=query_str,
                            nodes=nodes,
                        )
                        logger.info("使用备用方法成功生成回答")
                    except Exception as retriever_err:
                        logger.error(f"使用检索器的备用方法也失败: {str(retriever_err)}")
                        return {
                            "response": "对不起，处理您的查询时出现了问题。",
                            "citations": [],
                        }
                else:
                    return {
                        "response": "对不起，处理您的查询时出现了问题。",
                        "citations": [],
                    }
            
            # 恢复原始过滤条件
            if filters and original_filters is not None and hasattr(self.query_engine, "retriever"):
                self.query_engine.retriever.filters = original_filters
                logger.info("恢复原始过滤条件")
            
            # 提取引用信息
            source_nodes = getattr(response, "source_nodes", [])
            
            # 应用相似度过滤
            filtered_nodes = []
            for node in source_nodes:
                if node.score is None or node.score >= self.similarity_cutoff:
                    filtered_nodes.append(node)
            
            if filtered_nodes:
                logger.info(f"检索到 {len(filtered_nodes)}/{len(source_nodes)} 个相关节点")
                logger.debug(self._format_debug_info(filtered_nodes))
            else:
                logger.warning(f"所有检索结果 ({len(source_nodes)}) 的相似度低于阈值 {self.similarity_cutoff}")
                if source_nodes:
                    # 如果有结果但都被过滤，使用原始结果的前两个
                    filtered_nodes = source_nodes[:2]
                    logger.info(f"使用前 {len(filtered_nodes)} 个原始结果，忽略相似度阈值")
                else:
                    logger.warning("未检索到任何内容")
            
            # 格式化引用信息
            formatted_citations = []
            for i, node in enumerate(filtered_nodes):
                citation = {
                    "content": node.node.get_content(),
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                formatted_citations.append(citation)
            
            # 检查是否有回答
            answer = str(response) if response else "无法生成回答，请尝试重新表述您的问题。"
            if answer.strip() == "":
                answer = "对不起，我无法根据现有知识回答这个问题。"
                logger.warning("生成的回答为空")
            
            return {
                "response": answer,
                "citations": formatted_citations,
            }
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
            return {
                "response": "处理查询时发生错误，请稍后再试。",
                "citations": [],
            } 