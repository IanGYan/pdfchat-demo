# PDF 文档知识库 RAG 应用

一个主要用于代码学习的基于 LlamaIndex 的 PDF 文档检索增强生成(RAG)应用，支持构建 PDF 知识库并进行智能问答。

## 功能特点

- PDF 文档处理：自动读取、分块并向量化 PDF 文档
- 混合检索：结合 BM25 和向量检索提高查询精度
- 智能问答：基于大语言模型生成高质量回答
- 引用追踪：提供原文引用，支持溯源验证
- 简洁界面：基于 Gradio 构建的用户友好 Web 界面
- 灵活部署：支持本地运行和 Docker 部署

## 技术栈

- **后端**：Python, LlamaIndex, PostgreSQL + PGVector
- **前端**：Gradio
- **LLM 集成**：支持各种 LLM API 服务

## 快速开始

### 环境要求

- Python 3.9+
- PostgreSQL 数据库 (需要启用 pgvector 扩展)

  ```sql
  CREATE EXTENSION vector;
  ```

### 安装步骤

1. 克隆项目

   ```bash
   git clone <repository-url>
   cd llamaindex-pdfchat
   ```

2. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境变量

   ```bash
   cp .env.sample .env
   # 编辑.env文件，填入您的API密钥和数据库连接信息
   ```

4. 启动应用

   ```bash
   python app/main.py
   ```

5. 在浏览器中访问应用

   ```text
   http://localhost:7860
   ```

## 使用说明

1. 在 Web 界面上传 PDF 文件或指定包含 PDF 的目录
2. 等待系统处理文档并构建知识库
3. 在查询框中输入您的问题
4. 系统将返回基于文档内容的回答，并提供原文引用

## 项目结构

```text
llamaindex-pdfchat/
├── app/
│   ├── core/          # 核心功能模块
│   ├── database/      # 数据库连接与操作
│   ├── document_processing/ # 文档处理相关功能
│   ├── web/           # Web界面
│   ├── utils/         # 工具函数
│   └── main.py        # 应用入口
├── .env.sample        # 环境变量示例
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明
```

## Docker 部署

```bash
# 构建Docker镜像
docker build -t pdf-rag-app .

# 运行容器
docker run -p 7860:7860 --env-file .env pdf-rag-app
```
