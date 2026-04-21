"""
完整 RAG 流程
Query改写 → 混合检索(RRF) → 生成 → 评估
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
    SessionHistory,
)
from app.core.retriever import HybridRetriever
from app.core.query_rewriter import QueryRewriter
from app.core.generator import RAGGenerator
from app.core.evaluator import RAGEvaluator
from app.core.session_manager import session_manager

router = APIRouter(prefix="/chat", tags=["Chat"])

# 初始化（首次请求时加载）
_retriever: HybridRetriever = None
_rewriter: QueryRewriter = None
_generator: RAGGenerator = None
_evaluator: RAGEvaluator = None


def get_components():
    global _retriever, _rewriter, _generator, _evaluator
    if _retriever is None:
        _retriever = HybridRetriever()
        _rewriter = QueryRewriter()
        _generator = RAGGenerator()
        _evaluator = RAGEvaluator()
    return _retriever, _rewriter, _generator, _evaluator


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    RAG 问答接口

    完整流程：
    1. 加载会话历史（多轮对话）
    2. Query 改写（补全指代，扩展关键词）
    3. RRF 混合检索 向量 + BM25
    4. 生成答案
    5. 更新会话历史
    """
    try:
        retriever, rewriter, generator, evaluator = get_components()
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        raise HTTPException(status_code=503, detail=f"服务初始化失败: {str(e)}")

    session_id = req.session_id
    history = session_manager.get_history(session_id)

    # ── 1. Query 改写 ─────────────────────────────────────────────────────────
    rewritten_query = req.query
    if req.enable_rewrite:
        try:
            rewrite_result = await rewriter.rewrite(req.query, history)
            rewritten_query = rewrite_result["rewritten"]
        except Exception as e:
            logger.warning(f"Query改写失败 使用原始Query: {e}")
            rewritten_query = req.query  # 降级到原始 query

    # ── 2. RRF 混合检索 ───────────────────────────────────────────────────────
    top_k = req.top_k or 5
    try:
        retrieved = retriever.retrieve(rewritten_query, top_k=top_k)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        retrieved = []

    # 构造 SourceChunk
    sources = []
    for doc in retrieved:
        meta = doc.get("metadata", {})
        sources.append(
            SourceChunk(
                content=doc["content"][:300],
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(float(doc.get("score", 0.0)), 4),
                chunk_id=str(meta.get("doc_id", "")),
            )
        )

    # ── 3. 生成答案 ───────────────────────────────────────────────────────────
    try:
        if retrieved:
            answer = await generator.generate(
                query=rewritten_query,
                retrieved_docs=retrieved,
                history=history,
            )
        else:
            answer = await generator.generate_no_context(rewritten_query, history)
    except Exception as e:
        logger.error(f"LLM 生成失败: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"LLM 调用失败: {type(e).__name__}: {str(e)}。请检查 API Key 是否配置正确。",
        )

    # ── 4. 更新会话历史 ───────────────────────────────────────────────────────
    session_manager.add_turn(session_id, req.query, answer)

    # ── 5. 构造响应 ───────────────────────────────────────────────────────────
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        original_query=req.query,
        rewritten_query=rewritten_query if rewritten_query != req.query else None,
        sources=sources,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """流式 RAG 问答接口"""
    try:
        retriever, rewriter, generator, evaluator = get_components()
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": f"服务初始化失败: {str(e)}"})

    session_id = req.session_id
    history = session_manager.get_history(session_id)

    # 改写
    rewritten_query = req.query
    if req.enable_rewrite:
        try:
            rw = await rewriter.rewrite(req.query, history)
            rewritten_query = rw["rewritten"]
        except Exception as e:
            logger.warning(f"改写失败: {e}")

    # 检索
    top_k = req.top_k or 5
    try:
        retrieved = retriever.retrieve(rewritten_query, top_k=top_k)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        retrieved = []

    async def event_generator():
        try:
            # 先发送元信息
            meta = {
                "event": "meta",
                "original_query": req.query,
                "rewritten_query": rewritten_query,
                "sources": [
                    {
                        "source": d.get("metadata", {}).get("source", ""),
                        "page": int(d.get("metadata", {}).get("page", 0)),
                        "score": round(float(d.get("score", 0)), 4),
                        "content": d.get("content", "")[:200],
                    }
                    for d in retrieved
                ],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # 流式生成
            full_answer = ""
            async for token in generator.stream_generate(
                query=rewritten_query,
                retrieved_docs=retrieved,
                history=history,
            ):
                full_answer += token
                yield f"data: {json.dumps({'event': 'token', 'text': token}, ensure_ascii=False)}\n\n"

            # 更新会话
            session_manager.add_turn(session_id, req.query, full_answer)
            yield f"data: {json.dumps({'event': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            err = json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history/{session_id}", response_model=SessionHistory)
async def get_history(session_id: str):
    """获取会话历史"""
    messages = session_manager.get_history(session_id)
    return SessionHistory(
        session_id=session_id,
        messages=messages,
        total_turns=session_manager.total_turns(session_id),
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清除会话历史"""
    session_manager.clear_session(session_id)
    return {"message": f"会话 {session_id} 已清除"}
