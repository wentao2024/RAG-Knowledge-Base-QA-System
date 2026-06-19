"""
RAG evaluation API — automated quality scoring of answers.
"""
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models.schemas import EvalRequest, EvalResponse
from app.core.evaluator import RAGEvaluator

router = APIRouter(prefix="/eval", tags=["Evaluation"])

_evaluator: RAGEvaluator = None


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


@router.post("", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    """
    Evaluate RAG output quality.

    Metrics:
    - faithfulness: 0-1
    - answer_relevancy: 0-1
    - context_precision: 0-1
    - context_recall: 0-1 (requires ground_truth)
    - overall_score: 0-1
    """
    if not req.contexts:
        raise HTTPException(status_code=400, detail="contexts must not be empty")

    try:
        evaluator = get_evaluator()
        result = await evaluator.evaluate(
            question=req.query,
            answer=req.answer,
            contexts=req.contexts,
            ground_truth=req.ground_truth,
        )
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/health")
async def eval_health():
    return {"status": "ok", "module": "evaluator"}
