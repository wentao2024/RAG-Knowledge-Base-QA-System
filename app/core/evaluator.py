"""
RAG evaluation module: faithfulness, answer relevancy, and context precision.
Three LLM evaluations run in parallel (Semaphore controls concurrency to avoid rate limits).
"""
import asyncio
import re
from typing import List, Optional
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import EvalResponse


def _extract_score(text: str) -> float:
    """Extract a 0-1 numeric score from LLM output."""
    text = text.strip()
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0)
    return 0.5


FAITHFULNESS_PROMPT = """Evaluate how faithfully the following answer reflects the reference documents.

Reference documents:
{contexts}

Generated answer:
{answer}

Scoring criteria:
- 1.0: Answer is entirely based on the documents with no fabrication
- 0.7: Answer is mainly based on the documents with minor reasonable inferences
- 0.5: Answer is partially based on the documents with some fabrication
- 0.3: Most of the answer cannot be found in the documents
- 0.0: Answer is entirely fabricated

Return only a decimal between 0 and 1. No other content."""

RELEVANCY_PROMPT = """Evaluate how relevant the following answer is to the user's question.

User question: {question}

Generated answer: {answer}

Scoring criteria:
- 1.0: Answer fully addresses the question with sufficient information
- 0.7: Answer mostly addresses the question with minor deviations
- 0.5: Answer partially addresses the question
- 0.3: Answer is related to the question but does not truly answer it
- 0.0: Answer is completely irrelevant

Return only a decimal between 0 and 1. No other content."""

CONTEXT_PRECISION_PROMPT = """Evaluate how useful the retrieved documents are for answering the user's question.

User question: {question}

Retrieved documents:
{contexts}

Scoring criteria (are the documents useful for answering the question?):
- 1.0: All documents are highly relevant
- 0.7: Most documents are relevant
- 0.5: About half of the documents are relevant
- 0.3: Few documents are relevant
- 0.0: No documents are relevant

Return only a decimal between 0 and 1. No other content."""


def _extract_score(text: str) -> float:
    """Extract a numeric score from LLM output."""
    text = text.strip()
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0)
    return 0.5


class RAGEvaluator:
    def __init__(self):
        self.llm = LLMClient()

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvalResponse:
        """
        Comprehensive evaluation of RAG output quality.

        Three LLM evaluations are launched in parallel; Semaphore(2) limits concurrent in-flight requests.
        """
        ctx_text = "\n\n".join(
            [f"[Doc {i+1}] {c[:500]}" for i, c in enumerate(contexts)]
        )

        sem = asyncio.Semaphore(2)

        async def limited(coro):
            async with sem:
                return await coro

        faithfulness, relevancy, precision = await asyncio.gather(
            limited(self._eval_faithfulness(ctx_text, answer)),
            limited(self._eval_relevancy(question, answer)),
            limited(self._eval_context_precision(question, ctx_text)),
        )

        recall = None
        if ground_truth:
            recall = await self._eval_context_recall(ground_truth, ctx_text)

        scores = [faithfulness, relevancy, precision]
        if recall is not None:
            scores.append(recall)
        overall = round(sum(scores) / len(scores), 4)

        result = EvalResponse(
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(relevancy, 4),
            context_recall=round(recall, 4) if recall is not None else None,
            context_precision=round(precision, 4),
            overall_score=overall,
        )
        logger.info(f"Evaluation complete: {result.dict()}")
        return result

    async def _eval_faithfulness(self, contexts: str, answer: str) -> float:
        try:
            resp = await self.llm.complete(
                system="You are a strict evaluation expert. Return only a numeric score.",
                user=FAITHFULNESS_PROMPT.format(contexts=contexts, answer=answer),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5

    async def _eval_relevancy(self, question: str, answer: str) -> float:
        try:
            resp = await self.llm.complete(
                system="You are a strict evaluation expert. Return only a numeric score.",
                user=RELEVANCY_PROMPT.format(question=question, answer=answer),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"Relevancy evaluation failed: {e}")
            return 0.5

    async def _eval_context_precision(self, question: str, contexts: str) -> float:
        try:
            resp = await self.llm.complete(
                system="You are a strict evaluation expert. Return only a numeric score.",
                user=CONTEXT_PRECISION_PROMPT.format(
                    question=question, contexts=contexts
                ),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"Context precision evaluation failed: {e}")
            return 0.5

    async def _eval_context_recall(
        self, ground_truth: str, contexts: str
    ) -> float:
        """Simple keyword-coverage-based recall estimate."""
        try:
            import jieba
            gt_tokens = set(jieba.lcut(ground_truth))
            gt_tokens = {t for t in gt_tokens if len(t) > 1}
            ctx_text = contexts.lower()
            if not gt_tokens:
                return 0.5
            covered = sum(1 for t in gt_tokens if t in ctx_text)
            return covered / len(gt_tokens)
        except Exception as e:
            logger.warning(f"Context recall evaluation failed: {e}")
            return 0.5
