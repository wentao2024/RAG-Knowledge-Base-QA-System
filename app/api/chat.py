"""
Full RAG pipeline:
Query rewriting → parallel hybrid retrieval (RRF) → cross-encoder rerank → memory injection → generation → evaluation
"""
import asyncio
import json
import time
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
from app.core.generator import RAGGenerator
from app.core.evaluator import RAGEvaluator
from app.core.redis_session_manager import RedisSessionManager
from app.core.memory_store import MemoryStore
from app.core.memory_manager import MemoryManager
from app.core.query_analyzer import QueryAnalyzer

router = APIRouter(prefix="/chat", tags=["Chat"])

_retriever: HybridRetriever = None
_generator: RAGGenerator = None
_evaluator: RAGEvaluator = None
_session_manager: RedisSessionManager = None
_memory_store: MemoryStore = None
_memory_manager: MemoryManager = None
_analyzer: QueryAnalyzer = None


def get_components():
    global _retriever, _generator, _evaluator
    global _session_manager, _memory_store, _memory_manager, _analyzer

    if _retriever is None:
        from app.config import settings
        _retriever = HybridRetriever()
        _generator = RAGGenerator()
        _evaluator = RAGEvaluator()
        _session_manager = RedisSessionManager(settings.redis_url, ttl=settings.session_ttl)
        _memory_store = MemoryStore()
        _memory_manager = MemoryManager(_memory_store, _session_manager)
        _analyzer = QueryAnalyzer()

    return _retriever, _generator, _evaluator, _session_manager, _memory_store, _memory_manager, _analyzer


def _get_session_mgr():
    """Lightweight getter — only initialises the session manager, not the heavy ML components."""
    global _session_manager
    if _session_manager is None:
        from app.config import settings
        _session_manager = RedisSessionManager(settings.redis_url, ttl=settings.session_ttl)
    return _session_manager


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint — kept for compatibility; prefer /stream."""
    try:
        retriever, generator, evaluator, session_mgr, memory_store, memory_mgr, analyzer = get_components()
    except Exception as e:
        logger.error(f"Component initialisation failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service initialisation failed: {str(e)}")
    _start = time.perf_counter()

    session_id = req.session_id
    user_id = req.effective_user_id

    # ── 1. Load history + memory, AND analyze query — all in parallel ─────────
    (history, memories, user_summary), analysis = await asyncio.gather(
        asyncio.gather(
            asyncio.to_thread(session_mgr.get_history, session_id),
            asyncio.to_thread(memory_store.retrieve, user_id, req.query),
            asyncio.to_thread(session_mgr.get_user_summary, user_id),
        ),
        analyzer.analyze(req.query, None, req.enable_rewrite),
    )

    intent = analysis["intent"]
    rewritten_query = analysis["rewritten"] or req.query

    # ── 2. Chat path — no retrieval ────────────────────────────────────────────
    if intent == "chat":
        try:
            answer = await generator.generate_no_context(req.query, history)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}: {str(e)}")
        session_mgr.add_turn(session_id, req.query, answer)
        session_mgr.register_session_for_user(user_id, session_id)
        latency_ms = round((time.perf_counter() - _start) * 1000, 1)
        return ChatResponse(answer=answer, session_id=session_id,
                            original_query=req.query, rewritten_query=None,
                            sources=[], latency_ms=latency_ms)

    # ── 3. RAG path — retrieve → filter → generate ────────────────────────────
    top_k = req.top_k or 5
    t_preprocess = time.perf_counter()
    try:
        retrieved, retrieve_timing = await retriever.retrieve(rewritten_query, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        retrieved, retrieve_timing = [], {}
    retrieved = [d for d in retrieved if d.get("score", 0.0) >= 0.15]

    sources = [
        SourceChunk(
            content=doc["content"][:300],
            source=doc.get("metadata", {}).get("source", "unknown"),
            page=int(doc.get("metadata", {}).get("page", 0)),
            score=round(float(doc.get("score", 0.0)), 4),
            chunk_id=str(doc.get("metadata", {}).get("doc_id", "")),
        )
        for doc in retrieved
    ]

    try:
        gen_fn = generator.generate if retrieved else generator.generate_no_context
        answer = await gen_fn(
            query=rewritten_query, retrieved_docs=retrieved,
            history=history, memories=memories, user_summary=user_summary,
        ) if retrieved else await generator.generate_no_context(rewritten_query, history)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=502,
                            detail=f"LLM call failed: {type(e).__name__}: {str(e)}")

    session_mgr.add_turn(session_id, req.query, answer)
    session_mgr.register_session_for_user(user_id, session_id)
    user_total_turns = session_mgr.get_user_total_turns(user_id)
    from app.config import settings
    if settings.enable_user_memory:
        asyncio.create_task(
            memory_mgr.process_turn_async(user_id, session_id, req.query, answer, user_total_turns))

    latency_ms = round((time.perf_counter() - _start) * 1000, 1)
    logger.info(f"Request complete, latency={latency_ms}ms")
    return ChatResponse(answer=answer, session_id=session_id,
                        original_query=req.query,
                        rewritten_query=rewritten_query if rewritten_query != req.query else None,
                        sources=sources, latency_ms=latency_ms)


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """Streaming RAG Q&A endpoint."""
    try:
        retriever, generator, evaluator, session_mgr, memory_store, memory_mgr, analyzer = get_components()
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": f"Service initialisation failed: {str(e)}"})
    _start = time.perf_counter()

    session_id = req.session_id
    user_id = req.effective_user_id

    # ── 1. Load history + memory, AND analyze query — all in parallel ─────────
    (history, memories, user_summary), analysis = await asyncio.gather(
        asyncio.gather(
            asyncio.to_thread(session_mgr.get_history, session_id),
            asyncio.to_thread(memory_store.retrieve, user_id, req.query),
            asyncio.to_thread(session_mgr.get_user_summary, user_id),
        ),
        analyzer.analyze(req.query, None, req.enable_rewrite),
    )

    intent = analysis["intent"]
    rewritten_query = analysis["rewritten"] or req.query

    # ── 2. Chat path ───────────────────────────────────────────────────────────
    t_preprocess_done = time.perf_counter()
    if intent == "chat":
        async def chat_gen():
            yield f"data: {json.dumps({'event':'meta','original_query':req.query,'rewritten_query':None,'sources':[]}, ensure_ascii=False)}\n\n"
            full = ""
            first_token = True
            t_ttft = None
            try:
                async for tok in generator.stream_generate(
                    query=req.query, retrieved_docs=[], history=history,
                    memories=memories, user_summary=user_summary):
                    if first_token:
                        t_ttft = time.perf_counter()
                        first_token = False
                    full += tok
                    yield f"data: {json.dumps({'event':'token','text':tok}, ensure_ascii=False)}\n\n"
                session_mgr.add_turn(session_id, req.query, full)
                session_mgr.register_session_for_user(user_id, session_id)
                t_done = time.perf_counter()
                timing = {
                    "preprocess_ms": round((t_preprocess_done - _start) * 1000, 1),
                    "retrieve_ms":   0,
                    "ttft_ms":       round((t_ttft - _start) * 1000, 1) if t_ttft else None,
                    "total_ms":      round((t_done - _start) * 1000, 1),
                }
                yield f"data: {json.dumps({'event':'done','timing':timing}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'event':'error','detail':str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(chat_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # ── 3. RAG path — retrieve → filter → stream generation ───────────────────
    top_k = req.top_k or 5
    t_preprocess_done = time.perf_counter()
    try:
        retrieved, retrieve_timing = await retriever.retrieve(rewritten_query, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        retrieved, retrieve_timing = [], {}
    retrieved = [d for d in retrieved if d.get("score", 0.0) >= 0.15]
    t_retrieve_done = time.perf_counter()

    async def rag_gen():
        meta = {
            "event": "meta",
            "original_query": req.query,
            "rewritten_query": rewritten_query if rewritten_query != req.query else None,
            "memories_injected": len(memories),
            "sources": [{"source": d.get("metadata",{}).get("source",""),
                         "page": int(d.get("metadata",{}).get("page",0)),
                         "score": round(float(d.get("score",0)),4),
                         "content": d.get("content","")[:200]} for d in retrieved],
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        full = ""
        first_token = True
        t_ttft = None
        try:
            async for tok in generator.stream_generate(
                query=rewritten_query, retrieved_docs=retrieved, history=history,
                memories=memories, user_summary=user_summary):
                if first_token:
                    t_ttft = time.perf_counter()
                    first_token = False
                full += tok
                yield f"data: {json.dumps({'event':'token','text':tok}, ensure_ascii=False)}\n\n"
            session_mgr.add_turn(session_id, req.query, full)
            session_mgr.register_session_for_user(user_id, session_id)
            user_total_turns = session_mgr.get_user_total_turns(user_id)
            from app.config import settings
            if settings.enable_user_memory:
                asyncio.create_task(
                    memory_mgr.process_turn_async(user_id, session_id, req.query, full, user_total_turns))
            t_done = time.perf_counter()
            timing = {
                "preprocess_ms": round((t_preprocess_done - _start) * 1000, 1),
                "retrieve_ms":   round((t_retrieve_done - t_preprocess_done) * 1000, 1),
                "ttft_ms":       round((t_ttft - _start) * 1000, 1) if t_ttft else None,
                "total_ms":      round((t_done - _start) * 1000, 1),
                **retrieve_timing,
            }
            logger.info(
                f"[Stream] preprocess={timing['preprocess_ms']}ms"
                f"  retrieve={timing['retrieve_ms']}ms"
                f"  ttft={timing['ttft_ms']}ms"
                f"  total={timing['total_ms']}ms"
            )
            yield f"data: {json.dumps({'event':'done','timing':timing}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'event':'error','detail':str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(rag_gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.get("/history/{session_id}", response_model=SessionHistory)
async def get_history(session_id: str):
    """Retrieve session history."""
    session_mgr = _get_session_mgr()
    messages = session_mgr.get_history(session_id)
    return SessionHistory(
        session_id=session_id,
        messages=messages,
        total_turns=session_mgr.total_turns(session_id),
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear session history."""
    session_mgr = _get_session_mgr()
    session_mgr.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
