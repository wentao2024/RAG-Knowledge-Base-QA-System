"""
RRF (Reciprocal Rank Fusion) hybrid retriever.

Pipeline (when Parent-Child mode is enabled):
  Query
    → asyncio.gather[ VectorSearch(child), BM25(child) ]   # parallel dual-path child-chunk recall
    → RRF fusion
    → Cross-encoder Rerank (fine-ranking on child chunks — shorter text, higher precision)
    → Expand to Parent (swap child for parent chunk to give LLM richer context)
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple

from loguru import logger

from app.config import settings
from app.core.bm25_store import BM25Store
from app.core.rrf import reciprocal_rank_fusion  # noqa: F401 — re-exported
from app.core.vector_store import VectorStore


class HybridRetriever:
    """Hybrid retriever: vector + BM25 parallel recall → RRF → Rerank → Parent expansion."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

        if settings.enable_reranker:
            from app.core.reranker import Reranker
            self.reranker = Reranker()
        else:
            self.reranker = None

        # Parent-Child: retrieve child chunks, expand to parent chunks for the LLM
        if settings.enable_parent_child:
            from app.core.parent_store import ParentStore
            self.parent_store = ParentStore()
        else:
            self.parent_store = None

    # ─── Main retrieval entry point ────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        rrf_k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        1. Vector search + BM25 search run in parallel (asyncio.gather + to_thread)
        2. RRF fusion of both result sets
        3. Cross-encoder rerank (fine-ranking on child chunks)
        4. Expand to parent chunks (Parent-Child mode)
        5. Return (top_k results, timing_dict)
        """
        k = top_k or settings.top_k
        rrf_k_val = rrf_k or settings.rrf_k
        bw = bm25_weight if bm25_weight is not None else settings.bm25_weight
        vw = vector_weight if vector_weight is not None else settings.vector_weight
        recall_k = k * 2  # top_k=5 → 10 per path → ~15 after RRF dedup (was k*3=25)

        logger.info(
            f"[Retriever] Query='{query[:50]}', top_k={k}, "
            f"reranker={'on' if self.reranker else 'off'}, "
            f"parent_child={'on' if self.parent_store else 'off'}"
        )

        # ── 1. Parallel dual-path recall ───────────────────────────────────────
        t0 = time.perf_counter()
        vector_results, bm25_results = await asyncio.gather(
            asyncio.to_thread(self.vector_store.query, query, recall_k),
            asyncio.to_thread(self.bm25_store.query, query, recall_k),
        )
        t_recall = time.perf_counter()
        logger.debug(f"  Vector recall: {len(vector_results)}, BM25 recall: {len(bm25_results)}")

        # ── 2. RRF fusion ──────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            rank_lists=[vector_results, bm25_results],
            k=rrf_k_val,
            weights=[vw, bw],
        )
        t_rrf = time.perf_counter()

        # ── 3. Cross-encoder rerank (fine-ranking on child chunks) ─────────────
        if self.reranker is not None and fused:
            final = await asyncio.to_thread(self.reranker.rerank, query, fused, k)
            logger.info(f"  After rerank: {len(final)} results")
        else:
            final = fused[:k]
            logger.info(f"  After RRF fusion: {len(final)} results")
        t_rerank = time.perf_counter()

        # ── 4. Expand to parent chunks (after rerank so fine-ranking uses child chunks) ──
        if self.parent_store is not None and final:
            final = self._expand_to_parents(final)
            logger.info(f"  After parent expansion: {len(final)} results")
        t_expand = time.perf_counter()

        timing = {
            "recall_ms":   round((t_recall - t0)     * 1000, 1),
            "rrf_ms":      round((t_rrf    - t_recall)* 1000, 1),
            "rerank_ms":   round((t_rerank - t_rrf)  * 1000, 1),
            "expand_ms":   round((t_expand - t_rerank)* 1000, 1),
            "retrieve_ms": round((t_expand - t0)     * 1000, 1),
        }
        logger.info(
            f"  Timing: recall={timing['recall_ms']}ms  rrf={timing['rrf_ms']}ms"
            f"  rerank={timing['rerank_ms']}ms  expand={timing['expand_ms']}ms"
            f"  total={timing['retrieve_ms']}ms"
        )
        return final, timing

    # ─── Debug: return details per sub-retriever ───────────────────────────────

    async def retrieve_with_details(
        self, query: str, top_k: int = None
    ) -> Dict[str, Any]:
        """Retrieve and return detailed information (including per-retriever results, for debugging)."""
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

    # ─── Parent chunk expansion ────────────────────────────────────────────────

    def _expand_to_parents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand child-chunk retrieval results to their corresponding parent chunks.

        - Only one entry per parent_id is kept (highest-ranked child chunk wins).
        - Documents without a parent_id (legacy docs) are returned as-is (backwards-compatible).
        - Falls back to child content if the parent is missing (data inconsistency).
        """
        result = []
        seen_parents: set = set()

        for doc in docs:
            parent_id = doc.get("metadata", {}).get("parent_id")

            if not parent_id:
                # Legacy document (no parent_id): use as-is
                result.append(doc)
                continue

            if parent_id in seen_parents:
                # Parent already represented by a higher-ranked child: skip
                continue

            parent = self.parent_store.get(parent_id)
            if parent:
                expanded = {
                    "content": parent["content"],          # large chunk fed to LLM
                    "metadata": parent["metadata"],
                    "score": doc["score"],                  # preserve child chunk's ranking score
                    "rrf_score": doc.get("rrf_score", 0),
                }
                if "rerank_score" in doc:
                    expanded["rerank_score"] = doc["rerank_score"]
                result.append(expanded)
                seen_parents.add(parent_id)
            else:
                # Parent missing (should not happen): fall back to child content
                logger.warning(f"Parent chunk {parent_id} not found, falling back to child content")
                result.append(doc)

        return result
