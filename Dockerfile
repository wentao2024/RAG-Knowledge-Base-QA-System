FROM python:3.11-slim

LABEL maintainer="RAG Project"
LABEL description="中文RAG服务: RRF融合 · Query改写 · 多轮对话 · 评估闭环"

# 系统依赖（PyMuPDF 需要）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制 requirements 利用 Docker 缓存
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 预下载 sentence-transformers 模型（构建时缓存）
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('模型下载完成，维度:', model.get_sentence_embedding_dimension())"

# 预下载 jieba 词典
RUN python -c "import jieba; jieba.lcut('初始化jieba词典'); print('jieba 初始化完成')"

# 复制项目代码
COPY app/ ./app/
COPY frontend/ ./frontend/

# 创建数据目录
RUN mkdir -p /app/data/chroma /app/data/uploads

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "60"]
