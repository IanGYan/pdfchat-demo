"""
PDF文档加载和处理模块，负责读取PDF文件并提取文本内容
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import SentenceSplitter

from app.utils.config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None):
        """
        初始化PDF处理器
        
        Args:
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.pdf_reader = PDFReader()
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        加载PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            加载的文档列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"不支持的文件类型: {file_path}，仅支持PDF文件")
        
        try:
            logger.info(f"开始加载PDF文件: {file_path}")
            documents = self.pdf_reader.load_data(file_path)
            
            # 添加文件元数据
            file_name = os.path.basename(file_path)
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["file_name"] = file_name
                doc.metadata["file_path"] = file_path
            
            logger.info(f"成功加载PDF文件: {file_path}，共 {len(documents)} 页")
            return documents
        except Exception as e:
            logger.error(f"加载PDF文件失败: {str(e)}")
            raise
    
    def load_documents_from_dir(self, dir_path: str) -> List[Document]:
        """
        从目录加载所有PDF文档
        
        Args:
            dir_path: 包含PDF文件的目录路径
            
        Returns:
            加载的文档列表
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")
            
        all_documents = []
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"目录 {dir_path} 中没有找到PDF文件")
            return []
        
        for pdf_file in pdf_files:
            file_path = os.path.join(dir_path, pdf_file)
            try:
                docs = self.load_documents(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        
        logger.info(f"成功从目录 {dir_path} 加载 {len(all_documents)} 页PDF文档")
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        处理文档，进行分块
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的节点列表
        """
        if not documents:
            return []
            
        try:
            logger.info(f"开始处理 {len(documents)} 个文档")
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"文档处理完成，生成了 {len(nodes)} 个节点")
            return nodes
        except Exception as e:
            logger.error(f"处理文档时出错: {str(e)}")
            raise 