"""
Query rewriting module:
1. Uses conversation history to expand multi-turn questions into self-contained queries.
2. Expands keywords to improve retrieval recall.
3. Optionally generates multiple sub-queries for multi-path retrieval.
"""
from typing import List, Optional
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import Message


REWRITE_SYSTEM_PROMPT = """You are a professional search query optimisation expert. Your tasks are:
1. Analyse the conversation history and the current question.
2. Rewrite the current question into a self-contained, complete, and clear retrieval query.
3. Expand relevant keywords to broaden retrieval coverage.
4. Preserve the original intent of the question.

Rewriting rules:
- If the question is already complete and clear, keep the core structure and add relevant terminology.
- If the question relies on context (e.g. "this", "it"), resolve the reference explicitly.
- Output in English.
- Return only the rewritten query, no explanation.

Example:
History: [User: What is a transformer?] [Assistant: A Transformer is ...]
Current: What are its variants?
Rewritten: What are the main variants of the Transformer model? Including architectures such as BERT and GPT."""

DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition expert. Break a complex question into 2-3 simple sub-queries for separate retrieval.

Required format — JSON array:
["sub-query 1", "sub-query 2", "sub-query 3"]

Return only JSON, no other content."""


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
        Rewrite the query.

        Returns:
            {
                "original": str,
                "rewritten": str,
                "sub_queries": List[str]  # only when enable_decompose=True
            }
        """
        result = {"original": query, "rewritten": query, "sub_queries": [query]}

        # ── 1. Rewrite using conversation history ──────────────────────────
        history_text = ""
        if history:
            recent = history[-6:]  # last 3 turns
            for msg in recent:
                role = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role}: {msg.content[:200]}\n"

        user_prompt = f"Conversation history:\n{history_text}\nCurrent question: {query}" if history_text else f"Question: {query}"

        try:
            rewritten = await self.llm.complete(
                system=REWRITE_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=256,
            )
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) < 500:
                result["rewritten"] = rewritten
                logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original query: {e}")
            result["rewritten"] = query

        # ── 2. Sub-query decomposition (optional) ─────────────────────────
        if enable_decompose and len(query) > 20:
            try:
                import json
                decomposed_str = await self.llm.complete(
                    system=DECOMPOSE_SYSTEM_PROMPT,
                    user=f"Question: {result['rewritten']}",
                    max_tokens=256,
                )
                sub_queries = json.loads(decomposed_str.strip())
                if isinstance(sub_queries, list) and sub_queries:
                    result["sub_queries"] = [result["rewritten"]] + sub_queries[:2]
                    logger.info(f"Sub-query decomposition: {result['sub_queries']}")
            except Exception as e:
                logger.warning(f"Sub-query decomposition failed: {e}")
                result["sub_queries"] = [result["rewritten"]]
        else:
            result["sub_queries"] = [result["rewritten"]]

        return result
