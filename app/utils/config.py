"""
配置模块，负责加载环境变量和提供全局配置
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional, Any

# 加载环境变量
load_dotenv()

class Config:
    """配置类，提供对环境变量的访问"""
    
    # 有效的OpenAI模型列表
    VALID_OPENAI_MODELS = [
        "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-4-0125-preview", 
        "gpt-4-turbo-preview", "gpt-4-vision-preview", "gpt-4-0613", 
        "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo", 
        "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", 
        "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301",
        "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct"
    ]
    
    # 默认模型
    DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
    
    # 数据库配置
    PGVECTOR_URL = os.getenv("PGVECTOR_URL")  # PostgreSQL数据库连接URL
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 获取模型名称并验证
    _model_name = os.getenv("OPENAI_MODEL_NAME", DEFAULT_OPENAI_MODEL)
    # 如果模型名称无效，使用默认模型
    OPENAI_MODEL_NAME = _model_name if _model_name in VALID_OPENAI_MODELS else DEFAULT_OPENAI_MODEL
    
    # OpenAI Embedding配置
    OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    # 安全地获取嵌入维度，确保它是有效的整数
    _embed_dim_str = os.getenv("OPENAI_EMBEDDING_MODEL_DIM")
    if _embed_dim_str is None or _embed_dim_str.lower() == "none" or not _embed_dim_str.strip():
        # 如果环境变量未设置、为"None"或为空，使用默认值
        OPENAI_EMBEDDING_MODEL_DIM = 1536
    else:
        try:
            OPENAI_EMBEDDING_MODEL_DIM = int(_embed_dim_str)
            if OPENAI_EMBEDDING_MODEL_DIM <= 0:
                print(f"警告: 嵌入维度值 {OPENAI_EMBEDDING_MODEL_DIM} 无效，使用默认值1536")
                OPENAI_EMBEDDING_MODEL_DIM = 1536
        except ValueError:
            print(f"警告: 无法将嵌入维度值 '{_embed_dim_str}' 转换为整数，使用默认值1536")
            OPENAI_EMBEDDING_MODEL_DIM = 1536
    
    # 应用配置
    APP_PORT = int(os.getenv("APP_PORT", "7860"))
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    
    # 文档处理配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    @classmethod
    def validate_config(cls):
        """验证配置是否有效"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("未设置OpenAI API Key，请检查.env文件")
        
        if not cls.PGVECTOR_URL:
            raise ValueError("未设置PostgreSQL连接URL，请检查.env文件")
        
        # 验证模型名称
        if cls.OPENAI_MODEL_NAME not in cls.VALID_OPENAI_MODELS:
            print(f"警告: 模型名称 '{cls.OPENAI_MODEL_NAME}' 无效，将使用默认模型 '{cls.DEFAULT_OPENAI_MODEL}'")
            # 使用默认模型
            cls.OPENAI_MODEL_NAME = cls.DEFAULT_OPENAI_MODEL

    @classmethod
    def get_active_llm_config(cls) -> Dict[str, Any]:
        """获取当前激活的LLM配置"""
        
        # 优先使用OpenAI
        if cls.OPENAI_API_KEY:
            return {
                "api_key": cls.OPENAI_API_KEY,
                "model_name": cls.OPENAI_MODEL_NAME,
                "provider": "openai"
            }
        else:
            raise ValueError("未找到有效的LLM配置，请检查.env文件") 