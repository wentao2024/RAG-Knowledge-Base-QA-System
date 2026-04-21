"""
RAG 评估模块：忠实度、答案相关性、上下文精确率
使用 LLM 作为评判 + 关键词覆盖辅助
"""
import re
from typing import List, Optional, Dict
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import EvalResponse


FAITHFULNESS_PROMPT = """请评估以下答案对参考文档的忠实程度。

参考文档：
{contexts}

生成的答案：
{answer}

评估标准：
- 1.0：答案完全基于文档，无任何捏造
- 0.7：答案主要基于文档，有少量合理推断
- 0.5：答案部分基于文档，有一定捏造
- 0.3：答案大部分内容文档中找不到
- 0.0：答案完全是捏造的

只返回一个0到1之间的小数，不要其他内容。"""

RELEVANCY_PROMPT = """请评估以下答案对用户问题的相关程度。

用户问题：{question}

生成的答案：{answer}

评估标准：
- 1.0：答案完全回答了问题，信息充分
- 0.7：答案基本回答了问题，有少量偏差
- 0.5：答案部分回答了问题
- 0.3：答案与问题相关但没有真正回答
- 0.0：答案完全不相关

只返回一个0到1之间的小数，不要其他内容。"""

CONTEXT_PRECISION_PROMPT = """请评估检索到的文档对回答用户问题的有用程度。

用户问题：{question}

检索到的文档：
{contexts}

评估标准（每个文档是否对回答问题有用）：
- 1.0：所有文档都高度相关
- 0.7：大部分文档相关
- 0.5：约一半文档相关
- 0.3：少部分文档相关
- 0.0：所有文档都不相关

只返回一个0到1之间的小数，不要其他内容。"""


def _extract_score(text: str) -> float:
    """从 LLM 输出中提取数值分数"""
    text = text.strip()
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0)
    return 0.5  # 默认中间分


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
        综合评估 RAG 输出质量
        
        Returns:
            EvalResponse with scores:
            - faithfulness: 忠实度（答案是否忠实于文档）
            - answer_relevancy: 相关性（答案是否回答了问题）
            - context_precision: 上下文精确率（检索文档质量）
            - context_recall: 上下文召回率（需要 ground_truth）
            - overall_score: 综合分
        """
        ctx_text = "\n\n".join(
            [f"[文档{i+1}] {c[:500]}" for i, c in enumerate(contexts)]
        )

        # 并行评估（顺序调用，避免并发限流）
        faithfulness = await self._eval_faithfulness(ctx_text, answer)
        relevancy = await self._eval_relevancy(question, answer)
        precision = await self._eval_context_precision(question, ctx_text)

        # 召回率（需要 ground_truth）
        recall = None
        if ground_truth:
            recall = await self._eval_context_recall(ground_truth, ctx_text)

        # 综合分
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
        logger.info(f"评估完成: {result.dict()}")
        return result

    async def _eval_faithfulness(self, contexts: str, answer: str) -> float:
        try:
            resp = await self.llm.complete(
                system="你是一个严格的评估专家，只返回数字分数。",
                user=FAITHFULNESS_PROMPT.format(contexts=contexts, answer=answer),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"忠实度评估失败: {e}")
            return 0.5

    async def _eval_relevancy(self, question: str, answer: str) -> float:
        try:
            resp = await self.llm.complete(
                system="你是一个严格的评估专家，只返回数字分数。",
                user=RELEVANCY_PROMPT.format(question=question, answer=answer),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"相关性评估失败: {e}")
            return 0.5

    async def _eval_context_precision(self, question: str, contexts: str) -> float:
        try:
            resp = await self.llm.complete(
                system="你是一个严格的评估专家，只返回数字分数。",
                user=CONTEXT_PRECISION_PROMPT.format(
                    question=question, contexts=contexts
                ),
                max_tokens=10,
                temperature=0,
            )
            return _extract_score(resp)
        except Exception as e:
            logger.warning(f"上下文精确率评估失败: {e}")
            return 0.5

    async def _eval_context_recall(
        self, ground_truth: str, contexts: str
    ) -> float:
        """基于关键词覆盖率的简单召回率估算"""
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
            logger.warning(f"召回率评估失败: {e}")
            return 0.5
