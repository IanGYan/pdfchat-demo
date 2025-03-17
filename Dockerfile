FROM python:3.9-slim

WORKDIR /app

# 安装PostgreSQL依赖
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app ./app
COPY .env.sample ./.env.sample

# 创建目录
RUN mkdir -p logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV APP_PORT=7860
ENV APP_HOST=0.0.0.0

# 暴露端口
EXPOSE 7860

# 启动应用
CMD ["python", "app/main.py"] 