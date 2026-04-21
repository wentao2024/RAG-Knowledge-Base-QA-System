"""
RAG 评估 API 对答案质量进行自动化评估
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
    评估 RAG 输出质量
    
    指标：
    - faithfulness: 忠实度 0-1
    - answer_relevancy: 答案相关性 0-1
    - context_precision: 上下文精确率 0-1
    - context_recall: 上下文召回率  ground_truth 
    - overall_score: 综合分 0-1
    """
    if not req.contexts:
        raise HTTPException(status_code=400, detail="contexts 不能为空")

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
        logger.error(f"评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.get("/health")
async def eval_health():
    return {"status": "ok", "module": "evaluator"}
