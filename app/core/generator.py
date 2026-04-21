"""
RAG 生成器：组装检索上下文 → 调用 LLM 生成答案
"""
from typing import List, Dict, Any, AsyncIterator
from loguru import logger
from app.core.llm_client import LLMClient
from app.models.schemas import Message


RAG_SYSTEM_PROMPT = """你是一个专业的知识问答助手。请严格基于以下参考文档回答用户问题。

回答要求：
1. **忠实于文档**：只使用参考文档中的信息，不要添加文档中没有的内容
2. **结构清晰**：使用适当的格式（列表、段落）组织答案
3. **标注来源**：在引用具体信息时，说明来自哪个文档/页面
4. **承认局限**：如果文档中没有相关信息，明确告知用户
5. **语言流畅**：用自然的中文表达，避免机械复述

参考文档：
{context}

---
请基于以上文档回答用户问题。"""


def _build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """将检索结果格式化为上下文字符串"""
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "未知来源")
        page = meta.get("page", "?")
        content = doc.get("content", "")
        parts.append(f"【文档{i}】来源: {source}，第{page}页\n{content}")
    return "\n\n".join(parts)


class RAGGenerator:
    def __init__(self):
        self.llm = LLMClient()

    async def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        history: List[Message] = None,
        max_tokens: int = 2048,
    ) -> str:
        """生成答案（非流式）"""
        context = _build_context(retrieved_docs)
        system = RAG_SYSTEM_PROMPT.format(context=context)

        # 构造消息列表（多轮历史 + 当前问题）
        messages = list(history or [])
        messages.append(Message(role="user", content=query))

        answer = await self.llm.chat(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        logger.info(f"生成答案完成，长度: {len(answer)} 字符")
        return answer

    async def stream_generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        history: List[Message] = None,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """流式生成答案(retrieved_docs 为空时自动切换为无上下文模式）"""
        if retrieved_docs:
            context = _build_context(retrieved_docs)
            system = RAG_SYSTEM_PROMPT.format(context=context)
        else:
            system = "你是一个专业助手。当前知识库中没有找到相关文档，请如实告知用户，并尝试用你自身的知识提供帮助，同时建议用户上传相关文档。"

        messages = list(history or [])
        messages.append(Message(role="user", content=query))

        async for token in self.llm.stream_chat(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        ):
            yield token

    async def generate_no_context(
        self,
        query: str,
        history: List[Message] = None,
    ) -> str:
        """无上下文兜底回答（检索为空时）"""
        system = "你是一个专业助手。当前知识库中没有找到相关文档，请如实告知用户，并尝试用你自身的知识提供帮助，同时建议用户上传相关文档。"
        messages = list(history or [])
        messages.append(Message(role="user", content=query))
        return await self.llm.chat(system=system, messages=messages)
