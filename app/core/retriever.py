"""
RRF (Reciprocal Rank Fusion) 混合检索器

Pipeline（Parent-Child 模式开启时）：
  Query
    → asyncio.gather[ VectorSearch(child), BM25(child) ]   # 并行双路小块召回
    → RRF 融合
    → Cross-encoder Rerank（在子块上精排，文本短、精度更高）
    → Expand to Parent（用 parent_id 换回大块，喂给 LLM 上下文更完整）
"""
import asyncio
from typing import Dict, List, Any, Optional

from loguru import logger

from app.config import settings
from app.core.bm25_store import BM25Store
from app.core.rrf import reciprocal_rank_fusion  # noqa: F401 — re-exported
from app.core.vector_store import VectorStore


class HybridRetriever:
    """混合检索器：向量 + BM25 并行召回 → RRF → Rerank → Parent 扩展"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

        if settings.enable_reranker:
            from app.core.reranker import Reranker
            self.reranker = Reranker()
        else:
            self.reranker = None

        # Parent-Child：检索子块，扩展父块给 LLM
        if settings.enable_parent_child:
            from app.core.parent_store import ParentStore
            self.parent_store = ParentStore()
        else:
            self.parent_store = None

    # ─── 主检索入口 ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        rrf_k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """
        1. 向量检索 + BM25 检索 并行执行（asyncio.gather + to_thread）
        2. RRF 融合两路结果
        3. Cross-encoder rerank（在子块上精排）
        4. 扩展为父块（Parent-Child 模式）
        5. 返回 top_k 个结果
        """
        k = top_k or settings.top_k
        rrf_k_val = rrf_k or settings.rrf_k
        bw = bm25_weight if bm25_weight is not None else settings.bm25_weight
        vw = vector_weight if vector_weight is not None else settings.vector_weight
        recall_k = k * 3

        logger.info(
            f"[Retriever] Query='{query[:50]}', top_k={k}, "
            f"reranker={'on' if self.reranker else 'off'}, "
            f"parent_child={'on' if self.parent_store else 'off'}"
        )

        # ── 1. 并行双路召回 ────────────────────────────────────────────────────
        vector_results, bm25_results = await asyncio.gather(
            asyncio.to_thread(self.vector_store.query, query, recall_k),
            asyncio.to_thread(self.bm25_store.query, query, recall_k),
        )
        logger.debug(f"  向量召回: {len(vector_results)}, BM25召回: {len(bm25_results)}")

        # ── 2. RRF 融合 ────────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            rank_lists=[vector_results, bm25_results],
            k=rrf_k_val,
            weights=[vw, bw],
        )

        # ── 3. Cross-encoder rerank（在子块上精排，文本短效果更好）────────────
        if self.reranker is not None and fused:
            final = await asyncio.to_thread(self.reranker.rerank, query, fused, k)
            logger.info(f"  Rerank后: {len(final)} 条")
        else:
            final = fused[:k]
            logger.info(f"  RRF融合后: {len(final)} 条")

        # ── 4. 扩展为父块（Rerank 之后再扩展，保证精排基于小块）────────────────
        if self.parent_store is not None and final:
            final = self._expand_to_parents(final)
            logger.info(f"  父块扩展后: {len(final)} 条")

        return final

    # ─── 调试用：返回各子检索详情 ──────────────────────────────────────────────

    async def retrieve_with_details(
        self, query: str, top_k: int = None
    ) -> Dict[str, Any]:
        """检索并返回详细信息（含各子检索结果，便于调试）"""
        k = top_k or settings.top_k
        recall_k = k * 3

        vector_results, bm25_results = await asyncio.gather(
            asyncio.to_thread(self.vector_store.query, query, recall_k),
            asyncio.to_thread(self.bm25_store.query, query, recall_k),
        )

        fused = reciprocal_rank_fusion(
            rank_lists=[vector_results, bm25_results],
            k=settings.rrf_k,
            weights=[settings.vector_weight, settings.bm25_weight],
        )

        final = fused[:k]
        if self.reranker is not None and fused:
            final = await asyncio.to_thread(self.reranker.rerank, query, fused, k)

        if self.parent_store is not None and final:
            final = self._expand_to_parents(final)

        return {
            "final": final,
            "vector_results": vector_results[:k],
            "bm25_results": bm25_results[:k],
        }

    # ─── 父块扩展 ──────────────────────────────────────────────────────────────

    def _expand_to_parents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将子块检索结果扩展为对应父块。

        - 同一 parent_id 只保留一条（取分数最高的子块排名）
        - 没有 parent_id 的旧文档原样返回（向后兼容）
        - 父块找不到时（数据不一致）回退到子块内容
        """
        result = []
        seen_parents: set = set()

        for doc in docs:
            parent_id = doc.get("metadata", {}).get("parent_id")

            if not parent_id:
                # 旧文档（无 parent_id），直接使用
                result.append(doc)
                continue

            if parent_id in seen_parents:
                # 同一父块已被更高排名的子块代表，跳过
                continue

            parent = self.parent_store.get(parent_id)
            if parent:
                expanded = {
                    "content": parent["content"],          # 大块内容喂给 LLM
                    "metadata": parent["metadata"],
                    "score": doc["score"],                  # 保留子块的排名分数
                    "rrf_score": doc.get("rrf_score", 0),
                }
                if "rerank_score" in doc:
                    expanded["rerank_score"] = doc["rerank_score"]
                result.append(expanded)
                seen_parents.add(parent_id)
            else:
                # 父块丢失（不应发生），回退到子块
                logger.warning(f"父块 {parent_id} 未找到，回退到子块内容")
                result.append(doc)

        return result
