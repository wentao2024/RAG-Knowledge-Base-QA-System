"""
Query 改写模块：
1. 结合对话历史，将多轮问题补全为独立问题
2. 扩展关键词，提升检索召回率
3. 生成多个子查询（可选），用于多路召回
"""
from typing import List, Optional
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import Message


REWRITE_SYSTEM_PROMPT = """你是一个专业的搜索查询优化专家。你的任务是：
1. 分析用户的对话历史和当前问题
2. 将当前问题改写为一个独立、完整、清晰的检索查询
3. 扩展相关关键词，提升检索覆盖范围
4. 保持问题的原始意图不变

改写规则：
- 如果问题已经完整清晰，保持基本结构，补充专业术语
- 如果问题依赖上下文（如"这个"、"它"等指代），补全具体内容
- 保持中文输出
- 只返回改写后的查询，不要解释

示例：
历史: [用户:什么是transformer] [助手:Transformer是...]
当前: 它有哪些变体？
改写: Transformer模型的主要变体有哪些？包括BERT、GPT等架构"""

DECOMPOSE_SYSTEM_PROMPT = """你是查询分解专家。将复杂问题分解为2-3个简单的子查询，便于分别检索。

格式要求 JSON数组:
["子查询1", "子查询2", "子查询3"]

只返回JSON, 不要其他内容。"""


class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient()

    async def rewrite(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        enable_decompose: bool = False,
    ) -> dict:
        """
        改写查询
        
        Returns:
            {
                "original": str,
                "rewritten": str,
                "sub_queries": List[str]  # 若 enable_decompose=True
            }
        """
        result = {"original": query, "rewritten": query, "sub_queries": [query]}

        # ── 1. 结合历史改写 ─────────────────────────────────────────────────
        history_text = ""
        if history:
            recent = history[-6:]  # 最近 3 轮
            for msg in recent:
                role = "用户" if msg.role == "user" else "助手"
                history_text += f"{role}: {msg.content[:200]}\n"

        user_prompt = f"对话历史:\n{history_text}\n当前问题: {query}" if history_text else f"问题: {query}"

        try:
            rewritten = await self.llm.complete(
                system=REWRITE_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=256,
            )
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) < 500:
                result["rewritten"] = rewritten
                logger.info(f"Query改写: '{query}' → '{rewritten}'")
        except Exception as e:
            logger.warning(f"Query改写失败，使用原始query: {e}")
            result["rewritten"] = query

        # ── 2. 子查询分解（可选）──────────────────────────────────────────────
        if enable_decompose and len(query) > 20:
            try:
                import json
                decomposed_str = await self.llm.complete(
                    system=DECOMPOSE_SYSTEM_PROMPT,
                    user=f"问题: {result['rewritten']}",
                    max_tokens=256,
                )
                sub_queries = json.loads(decomposed_str.strip())
                if isinstance(sub_queries, list) and sub_queries:
                    result["sub_queries"] = [result["rewritten"]] + sub_queries[:2]
                    logger.info(f"子查询分解: {result['sub_queries']}")
            except Exception as e:
                logger.warning(f"子查询分解失败: {e}")
                result["sub_queries"] = [result["rewritten"]]
        else:
            result["sub_queries"] = [result["rewritten"]]

        return result
