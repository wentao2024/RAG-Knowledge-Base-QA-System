"""
RAG Generator: assembles retrieval context and calls the LLM to produce an answer.
"""
from typing import List, Dict, Any, AsyncIterator, Optional
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import Message


_RAG_SYSTEM_PROMPT = """You are a professional knowledge Q&A assistant. Answer the user's question based on the information provided below.

{user_context}{doc_context}
---
Answer requirements:
1. **Stay faithful to the documents**: Use only information from the reference documents; do not add content not found in them.
2. **Be well-structured**: Organise your answer with appropriate formatting (lists, paragraphs).
3. **Cite sources**: When referencing specific information, indicate which document/page it comes from.
4. **Acknowledge limitations**: If the documents do not contain relevant information, clearly inform the user.
5. **Write in fluent English**: Express ideas naturally; avoid mechanical paraphrasing.

Please answer the user's question based on the information above."""


def _build_doc_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown source")
        page = meta.get("page", "?")
        content = doc.get("content", "")
        parts.append(f"[Doc {i}] Source: {source}, Page {page}\n{content}")
    return "\n\n".join(parts)


def _build_user_context(memories: List[str], user_summary: Optional[str]) -> str:
    parts = []
    if user_summary:
        parts.append(f"[User Background]\n{user_summary}")
    if memories:
        facts = "\n".join(f"- {m}" for m in memories)
        parts.append(f"[Memories about this user]\n{facts}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


class RAGGenerator:
    def __init__(self):
        self.llm = LLMClient()

    async def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        history: List[Message] = None,
        memories: List[str] = None,
        user_summary: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Generate an answer (non-streaming)."""
        user_ctx = _build_user_context(memories or [], user_summary)
        doc_ctx = "Reference documents:\n" + _build_doc_context(retrieved_docs) if retrieved_docs else ""
        system = _RAG_SYSTEM_PROMPT.format(user_context=user_ctx, doc_context=doc_ctx)

        messages = list(history or [])
        messages.append(Message(role="user", content=query))

        answer = await self.llm.chat(system=system, messages=messages, max_tokens=max_tokens)
        logger.info(f"Answer generation complete, length: {len(answer)} chars")
        return answer

    async def stream_generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        history: List[Message] = None,
        memories: List[str] = None,
        user_summary: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream-generate an answer."""
        if retrieved_docs:
            user_ctx = _build_user_context(memories or [], user_summary)
            doc_ctx = "Reference documents:\n" + _build_doc_context(retrieved_docs)
            system = _RAG_SYSTEM_PROMPT.format(user_context=user_ctx, doc_context=doc_ctx)
        else:
            system = "You are a professional assistant. No relevant documents were found in the knowledge base. Honestly inform the user, try to help with your own knowledge, and suggest they upload relevant documents."

        messages = list(history or [])
        messages.append(Message(role="user", content=query))

        async for token in self.llm.stream_chat(
            system=system, messages=messages, max_tokens=max_tokens
        ):
            yield token

    async def generate_no_context(
        self,
        query: str,
        history: List[Message] = None,
    ) -> str:
        """Fallback answer when no documents were retrieved."""
        system = "You are a professional assistant. No relevant documents were found in the knowledge base. Honestly inform the user, try to help with your own knowledge, and suggest they upload relevant documents."
        messages = list(history or [])
        messages.append(Message(role="user", content=query))
        return await self.llm.chat(system=system, messages=messages)
