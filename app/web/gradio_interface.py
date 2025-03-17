"""
Gradio Web界面模块，提供用户交互界面
"""

import os
import logging
import tempfile
import time
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple

from app.core.knowledge_base import KnowledgeBase
from app.utils.config import Config

logger = logging.getLogger(__name__)

class GradioInterface:
    """Gradio Web界面类"""
    
    def __init__(self):
        """初始化Gradio界面"""
        self.knowledge_base = None
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"临时文件目录: {self.temp_dir}")
        print(f"临时文件目录: {self.temp_dir}")
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self, rebuild: bool = False) -> Dict[str, Any]:
        """
        初始化知识库
        
        Args:
            rebuild: 是否重建知识库
            
        Returns:
            知识库状态信息
        """
        try:
            # 显示正在初始化的消息
            logger.info("正在初始化知识库...")
            print("正在初始化知识库...")
            
            self.knowledge_base = KnowledgeBase(rebuild=rebuild)
            
            # 验证知识库是否正确初始化
            if self.knowledge_base is None:
                raise ValueError("知识库初始化后为空")
                
            status = self.knowledge_base.get_status()
            logger.info(f"知识库初始化成功，共有 {status['document_count']} 个文档")
            print(f"知识库初始化成功，共有 {status['document_count']} 个文档")
            
            return {"status": "成功", "message": f"知识库初始化完成，共有 {status['document_count']} 个文档"}
        except Exception as e:
            logger.error(f"知识库初始化失败: {str(e)}")
            print(f"知识库初始化失败: {str(e)}")
            return {"status": "失败", "message": f"知识库初始化失败: {str(e)}"}
    
    def upload_pdf(self, files: List[Any]) -> Dict[str, Any]:
        """
        上传PDF文件
        
        Args:
            files: 上传的文件列表
            
        Returns:
            处理结果信息
        """
        if not files:
            return {"status": "失败", "message": "未选择任何文件"}
            
        try:
            # 确保知识库已初始化
            if not self.knowledge_base:
                logger.info("知识库未初始化，正在尝试初始化...")
                print("知识库未初始化，正在尝试初始化...")
                init_result = self.initialize_knowledge_base()
                if init_result["status"] == "失败":
                    return init_result
                
            # 再次检查知识库是否成功初始化
            if not self.knowledge_base:
                logger.error("知识库初始化失败，无法上传文件")
                print("知识库初始化失败，无法上传文件")
                return {"status": "失败", "message": "知识库初始化失败，无法上传文件"}
                
            file_paths = []
            for file_obj in files:
                # 打印文件对象类型，帮助调试
                logger.info(f"文件对象类型: {type(file_obj)}")
                print(f"文件对象类型: {type(file_obj)}")
                
                try:
                    # 尝试获取文件名（兼容不同版本的Gradio）
                    if hasattr(file_obj, 'name'):
                        filename = file_obj.name
                        logger.info(f"从属性获取文件名: {filename}")
                    elif isinstance(file_obj, tuple) and len(file_obj) > 1:
                        filename = os.path.basename(file_obj[1])
                        logger.info(f"从元组获取文件名: {filename}")
                    elif isinstance(file_obj, dict) and 'name' in file_obj:
                        filename = file_obj['name']
                        logger.info(f"从字典获取文件名: {filename}")
                    else:
                        # 如果无法获取文件名，生成一个随机文件名
                        timestamp = int(time.time())
                        filename = f"uploaded_file_{timestamp}.pdf"
                        logger.info(f"使用生成的文件名: {filename}")
                    
                    # 处理中文文件名：确保能够正确编码
                    try:
                        # 尝试使用纯ASCII文件名（替换或删除非ASCII字符）
                        safe_filename = ''.join(c if ord(c) < 128 else '_' for c in filename)
                        if safe_filename != filename:
                            logger.info(f"将中文文件名 '{filename}' 转换为安全文件名 '{safe_filename}'")
                            filename = safe_filename
                            
                        # 如果文件名变成了空字符串或只有下划线，使用时间戳作为名称
                        if not filename.strip('_'):
                            timestamp = int(time.time())
                            filename = f"chinese_filename_{timestamp}.pdf"
                            logger.info(f"使用时间戳替代空文件名: {filename}")
                    except Exception as e:
                        logger.warning(f"处理文件名时出错: {str(e)}，使用替代名称")
                        timestamp = int(time.time())
                        filename = f"filename_{timestamp}.pdf"
                    
                    # 确保文件名以.pdf结尾
                    if not filename.lower().endswith('.pdf'):
                        filename += '.pdf'
                    
                    # 文件保存路径
                    file_path = os.path.join(self.temp_dir, filename)
                    logger.info(f"准备保存文件到: {file_path}")
                    
                    # 尝试不同的方法读取文件内容
                    file_content = None
                    
                    # 检查文件对象并打印详细信息
                    logger.info(f"文件对象详情: {dir(file_obj)}")
                    
                    try:
                        if hasattr(file_obj, 'read'):
                            # 传统方式 - 直接读取
                            file_content = file_obj.read()
                            logger.info("使用 .read() 方法读取文件内容")
                        elif isinstance(file_obj, tuple) and len(file_obj) > 0:
                            # Gradio 4.x 方式 - 元组的第一个元素是文件路径
                            src_path = file_obj[0]
                            logger.info(f"从元组获取文件路径: {src_path}")
                            with open(src_path, 'rb') as f:
                                file_content = f.read()
                        elif isinstance(file_obj, dict) and 'path' in file_obj:
                            # 另一种可能的方式 - 字典中包含路径
                            src_path = file_obj['path']
                            logger.info(f"从字典获取文件路径: {src_path}")
                            with open(src_path, 'rb') as f:
                                file_content = f.read()
                        elif isinstance(file_obj, str):
                            # 字符串可能是文件路径
                            logger.info(f"字符串作为文件路径: {file_obj}")
                            with open(file_obj, 'rb') as f:
                                file_content = f.read()
                        elif hasattr(file_obj, '__str__'):
                            # Gradio 4.x NamedString 类型 - 尝试获取字符串值
                            try:
                                # 针对 NamedString 类型的特殊处理
                                if 'NamedString' in str(type(file_obj)):
                                    logger.info("检测到 NamedString 类型")
                                    # 通常 NamedString 在 Gradio 4.x 用于从本地上传的文件
                                    # 在这种情况下，file_obj 实际上是一个模拟文件的对象
                                    # value 属性通常包含临时文件路径
                                    if hasattr(file_obj, 'value') and file_obj.value:
                                        logger.info(f"从 NamedString.value 获取内容: {file_obj.value}")
                                        if os.path.exists(file_obj.value):
                                            with open(file_obj.value, 'rb') as f:
                                                file_content = f.read()
                                        else:
                                            # 如果不是文件路径，可能是内容本身
                                            file_content = file_obj.value.encode('utf-8')
                                    elif str(file_obj):
                                        # 尝试将对象转换为字符串，可能是文件内容
                                        logger.info("将NamedString对象转换为字符串")
                                        file_content = str(file_obj).encode('utf-8')
                            except Exception as e:
                                logger.error(f"处理NamedString时出错: {str(e)}")
                                raise
                        
                        if file_content is None:
                            logger.warning(f"无法读取文件内容，尝试其他方法")
                            # 尝试直接使用文件对象的字符串表示
                            file_content = str(file_obj).encode('utf-8')
                    except Exception as e:
                        logger.error(f"读取文件内容时出错: {str(e)}")
                        raise
                    
                    # 保存文件内容
                    with open(file_path, "wb") as f:
                        f.write(file_content)
                    
                    logger.info(f"成功保存文件: {file_path}")
                    file_paths.append(file_path)
                    
                    # 打印文件路径
                    logger.info(f"上传文件已保存至: {file_path}")
                    print(f"上传文件已保存至: {file_path}")
                    
                    # 确保知识库仍然可用
                    if not self.knowledge_base:
                        logger.error("知识库在处理过程中变为None，尝试重新初始化")
                        print("知识库在处理过程中变为None，尝试重新初始化")
                        self.initialize_knowledge_base()
                        
                        if not self.knowledge_base:
                            logger.error("重新初始化知识库失败，跳过添加文档步骤")
                            print("重新初始化知识库失败，跳过添加文档步骤")
                            continue
                    
                    # 添加文档到知识库
                    logger.info(f"正在向知识库添加文档: {file_path}")
                    print(f"正在向知识库添加文档: {file_path}")
                    self.knowledge_base.add_pdf_document(file_path)
                    logger.info("文档添加成功")
                    print("文档添加成功")
                
                except Exception as e:
                    logger.error(f"处理文件时出错: {str(e)}")
                    print(f"处理文件时出错: {str(e)}")
                    continue
            
            if not file_paths:
                return {"status": "失败", "message": "所有文件处理都失败了"}
            
            # 确保知识库仍然可用
            if not self.knowledge_base:
                logger.error("知识库在处理完文件后变为None，无法获取状态")
                return {"status": "部分成功", "message": f"成功上传 {len(file_paths)} 个文件，但无法获取知识库状态"}
            
            try:
                status = self.knowledge_base.get_status()
                return {
                    "status": "成功", 
                    "message": f"成功上传 {len(file_paths)} 个文件，知识库现有 {status['document_count']} 个文档"
                }
            except Exception as e:
                logger.error(f"获取知识库状态失败: {str(e)}")
                return {
                    "status": "部分成功", 
                    "message": f"成功上传 {len(file_paths)} 个文件，但获取知识库状态失败: {str(e)}"
                }
        except Exception as e:
            logger.error(f"上传PDF文件失败: {str(e)}")
            return {"status": "失败", "message": f"上传PDF文件失败: {str(e)}"}
    
    def process_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        处理目录中的PDF文件
        
        Args:
            dir_path: 目录路径
            
        Returns:
            处理结果信息
        """
        if not dir_path:
            return {"status": "失败", "message": "未指定目录路径"}
            
        try:
            if not self.knowledge_base:
                self.initialize_knowledge_base()
                
            # 添加目录中的文档到知识库
            self.knowledge_base.add_pdf_documents_from_dir(dir_path)
            
            status = self.knowledge_base.get_status()
            return {
                "status": "成功", 
                "message": f"成功处理目录 {dir_path}，知识库现有 {status['document_count']} 个文档"
            }
        except Exception as e:
            logger.error(f"处理目录失败: {str(e)}")
            return {"status": "失败", "message": f"处理目录失败: {str(e)}"}
    
    def query_knowledge_base(self, query: str) -> Tuple[str, str]:
        """
        查询知识库
        
        Args:
            query: 查询字符串
            
        Returns:
            回答和引用
        """
        if not query or query.strip() == "":
            return "请输入有效的查询", ""
            
        try:
            if not self.knowledge_base:
                return "知识库尚未初始化，请先上传PDF文档", ""
                
            # 查询知识库
            result = self.knowledge_base.query(query)
            
            # 获取回答
            response = result.get("response", "")
            
            # 格式化引用
            citations = result.get("citations", [])
            formatted_citations = ""
            
            if citations:
                formatted_citations = "## 引用来源\n\n"
                for i, citation in enumerate(citations):
                    metadata = citation.get("metadata", {})
                    file_name = metadata.get("file_name", "未知文件")
                    
                    formatted_citations += f"### 引用 {i+1}（来自 {file_name}）\n\n"
                    formatted_citations += f"```\n{citation.get('content', '')}\n```\n\n"
            
            return response, formatted_citations
        except Exception as e:
            logger.error(f"查询知识库失败: {str(e)}")
            return f"查询失败: {str(e)}", ""
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        清空知识库
        
        Returns:
            处理结果信息
        """
        try:
            if not self.knowledge_base:
                return {"status": "成功", "message": "知识库尚未初始化"}
                
            self.knowledge_base.clear_knowledge_base()
            return {"status": "成功", "message": "知识库已清空"}
        except Exception as e:
            logger.error(f"清空知识库失败: {str(e)}")
            return {"status": "失败", "message": f"清空知识库失败: {str(e)}"}
    
    def create_gradio_interface(self) -> gr.Blocks:
        """
        创建Gradio界面
        
        Returns:
            Gradio界面对象
        """
        with gr.Blocks(title="PDF知识库RAG应用") as interface:
            gr.Markdown("# PDF知识库RAG应用")
            gr.Markdown("基于OpenAI LLM的PDF文档检索增强生成（RAG）系统，提供智能问答服务")
            gr.Markdown(f"使用模型: **{Config.OPENAI_MODEL_NAME}** | 嵌入模型: **{Config.OPENAI_EMBEDDING_MODEL_NAME}**")
            
            with gr.Tab("文档管理"):
                with gr.Row():
                    with gr.Column():
                        pdf_files = gr.File(
                            label="上传PDF文件",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        upload_button = gr.Button("上传文件")
                        
                        dir_path = gr.Textbox(label="PDF文件目录路径")
                        process_dir_button = gr.Button("处理目录")
                        
                        clear_button = gr.Button("清空知识库", variant="stop")
                    
                    with gr.Column():
                        result_info = gr.JSON(label="处理结果")
            
            with gr.Tab("问答"):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="输入问题",
                        placeholder="在此输入您的问题...",
                        lines=3
                    )
                
                query_button = gr.Button("提交问题")
                
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Markdown(label="回答")
                    
                    with gr.Column():
                        citations_output = gr.Markdown(label="引用来源")
            
            # 设置事件处理
            upload_button.click(
                fn=self.upload_pdf,
                inputs=[pdf_files],
                outputs=[result_info]
            )
            
            process_dir_button.click(
                fn=self.process_directory,
                inputs=[dir_path],
                outputs=[result_info]
            )
            
            clear_button.click(
                fn=self.clear_knowledge_base,
                inputs=[],
                outputs=[result_info]
            )
            
            query_button.click(
                fn=self.query_knowledge_base,
                inputs=[query_input],
                outputs=[answer_output, citations_output]
            )
        
        return interface
    
    def launch(self, server_port: int = None, server_host: str = None):
        """
        启动Gradio服务
        
        Args:
            server_port: 服务端口
            server_host: 服务主机
        """
        interface = self.create_gradio_interface()
        
        port = server_port or Config.APP_PORT
        host = server_host or Config.APP_HOST
        
        logger.info(f"启动Gradio服务，地址: {host}:{port}")
        
        # 使用更简单的方式启动Gradio，避免与FastAPI和Pydantic的冲突
        try:
            # 方式1：最简单的启动方式
            interface.launch(
                server_name=host,
                server_port=port,
                share=True
            )
        except Exception as e:
            logger.error(f"Gradio服务启动失败: {str(e)}")
            # 方式2：备用启动方式
            logger.info("尝试使用备用设置启动...")
            gr.close_all()
            interface.queue(concurrency_count=1).launch(
                server_port=port,
                share=True
            ) 