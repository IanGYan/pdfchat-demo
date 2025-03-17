"""
应用入口文件，负责启动应用
"""

import os
import sys
import logging
import argparse

# 添加项目根目录到Python路径
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取app目录的路径
app_dir = os.path.dirname(current_file)
# 获取项目根目录的路径
project_root = os.path.dirname(app_dir)
# 将项目根目录添加到sys.path
sys.path.insert(0, project_root)

from app.utils.logger import setup_logging
from app.utils.config import Config
from app.web.gradio_interface import GradioInterface

# 设置日志
logger = setup_logging()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PDF知识库RAG应用")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=Config.APP_PORT,
        help="Web服务端口"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=Config.APP_HOST,
        help="Web服务主机"
    )
    
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="重建知识库"
    )
    
    return parser.parse_args()

def main():
    """应用主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 验证配置
    try:
        Config.validate_config()
    except ValueError as e:
        logger.error(f"配置验证失败: {str(e)}")
        return
    
    logger.info("正在启动PDF知识库RAG应用...")
    logger.info(f"服务地址: {args.host}:{args.port}")
    logger.info(f"使用LLM模型: {Config.OPENAI_MODEL_NAME}")
    logger.info(f"使用嵌入模型: {Config.OPENAI_EMBEDDING_MODEL_NAME} (维度: {Config.OPENAI_EMBEDDING_MODEL_DIM})")
    
    try:
        # 创建Gradio界面
        app = GradioInterface()
        
        # 启动应用
        app.launch(server_port=args.port, server_host=args.host)
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 