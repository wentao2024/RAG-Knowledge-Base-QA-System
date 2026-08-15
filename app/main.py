"""
RAG application entry point.
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
from app.api import chat, upload, eval as eval_router, users as users_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up on startup."""
    logger.info("=" * 50)
    logger.info("Starting RAG service...")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Base URL:  {settings.openai_base_url}")

    # Create required directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    # Pre-load all heavy components (avoids cold-start on first request)
    try:
        from app.core.embedder import Embedder
        embedder = Embedder()
        logger.info(f"Embedding model warmed up, dimension: {embedder.get_dim()}")
    except Exception as e:
        logger.warning(f"Embedding model warm-up failed: {e}")

    try:
        import asyncio as _asyncio
        from app.api.chat import get_components
        logger.info("Loading RAG components (reranker may take 1-3 min on first run)...")
        await _asyncio.to_thread(get_components)
        logger.info("All RAG components warmed up (retriever, reranker, generator, session manager)")
    except Exception as e:
        logger.warning(f"RAG component warm-up failed: {e}")

    logger.info("RAG service started successfully!")
    logger.info("=" * 50)
    yield
    logger.info("RAG service shut down")


app = FastAPI(
    title="RAG Service",
    description="RRF fusion ranking · Query rewriting · Multi-turn dialogue · Evaluation loop",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Global exception handlers: ensure all errors return JSON ──────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 validation error → JSON"""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors()), "body": str(exc.body)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: unhandled exceptions → JSON 500, prevents HTML responses."""
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception [{request.method} {request.url}]:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {type(exc).__name__}: {str(exc)}",
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

# API routes
app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(eval_router.router, prefix="/api")
app.include_router(users_router.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "provider": "openai",
        "model": settings.llm_model,
        "base_url": settings.openai_base_url,
    }


# Static frontend: prefer Docker path, fall back to local relative path
_docker_frontend = "/app/frontend"
_local_frontend = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
frontend_path = _docker_frontend if os.path.exists(_docker_frontend) else _local_frontend

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))