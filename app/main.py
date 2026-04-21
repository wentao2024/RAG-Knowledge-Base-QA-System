"""
RAG 应用主入口
"""
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.config import settings
from app.api import chat, upload, eval as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热"""
    logger.info("=" * 50)
    logger.info("RAG 服务启动中...")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Base URL:  {settings.openai_base_url}")

    # 创建必要目录
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    # 预加载 Embedding 模型（避免首次请求慢）
    try:
        from app.core.embedder import Embedder
        embedder = Embedder()
        logger.info(f"Embedding 模型预热完成，维度: {embedder.get_dim()}")
    except Exception as e:
        logger.warning(f"Embedding 模型预热失败: {e}")

    logger.info("RAG 服务启动完成！")
    logger.info("=" * 50)
    yield
    logger.info("RAG 服务关闭")


app = FastAPI(
    title="中文 RAG 服务",
    description="支持 RRF 融合排序 · Query 改写 · 多轮对话 · 评估闭环",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 全局异常处理：确保所有错误都返回 JSON ─────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 校验错误 → JSON"""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors()), "body": str(exc.body)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底：所有未捕获异常 → JSON 500，避免返回 HTML"""
    tb = traceback.format_exc()
    logger.error(f"未捕获异常 [{request.method} {request.url}]:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"服务器内部错误: {type(exc).__name__}: {str(exc)}",
            "type": type(exc).__name__,
        },
    )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(eval_router.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "provider": "openai",
        "model": settings.llm_model,
        "base_url": settings.openai_base_url,
    }


# 静态前端
frontend_path = "/app/frontend"
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))