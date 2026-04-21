"""
RRF (Reciprocal Rank Fusion) 混合检索器
将向量检索（密集）+ BM25 (稀疏)融合排序，提升召回质量
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from app.config import settings
from app.core.vector_store import VectorStore
from app.core.bm25_store import BM25Store


def reciprocal_rank_fusion(
    rank_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    RRF 算法实现
    
    公式:RRF(d) = Σ weight_i / (k + rank_i(d))
    
    Args:
        rank_lists: 多个排序结果列表，每个元素含 'content' 和 'metadata'
        k: RRF 平滑系数（通常取 60)
        weights: 各列表的权重，默认均等
    
    Returns:
        融合后的排序结果列表（含 rrf_score)
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    assert len(weights) == len(rank_lists), "权重数量须与列表数量一致"

    # 用 content 作为 key 聚合分数
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank_list, weight in zip(rank_lists, weights):
        for rank, doc in enumerate(rank_list, start=1):
            key = doc["content"][:200]  # 截断作为 key
            rrf_score = weight / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

    # 按 RRF 分数排序
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = round(scores[key], 6)
        doc["score"] = doc["rrf_score"]  # 统一字段名
        results.append(doc)

    return results


class HybridRetriever:
    """混合检索器：向量 + BM25 → RRF 融合"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        rrf_k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索主入口
        
        1. 并行执行向量检索 + BM25 检索
        2. 对两路结果执行 RRF 融合
        3. 返回 top_k 个融合结果
        """
        k = top_k or settings.top_k
        rrf_k_val = rrf_k or settings.rrf_k
        bw = bm25_weight if bm25_weight is not None else settings.bm25_weight
        vw = vector_weight if vector_weight is not None else settings.vector_weight

        # 扩大召回量，融合后再截断
        recall_k = k * 3

        logger.info(f"[Retriever] Query='{query[:50]}', top_k={k}")

        # 向量检索
        vector_results = self.vector_store.query(query, top_k=recall_k)
        logger.debug(f"  向量检索召回: {len(vector_results)} 条")

        # BM25 检索
        bm25_results = self.bm25_store.query(query, top_k=recall_k)
        logger.debug(f"  BM25 检索召回: {len(bm25_results)} 条")

        # RRF 融合
        fused = reciprocal_rank_fusion(
            rank_lists=[vector_results, bm25_results],
            k=rrf_k_val,
            weights=[vw, bw],
        )

        final = fused[:k]
        logger.info(f"  RRF 融合后返回: {len(final)} 条")
        return final

    def retrieve_with_details(
        self, query: str, top_k: int = None
    ) -> Dict[str, Any]:
        """检索并返回详细信息（含各子检索结果，便于调试）"""
        k = top_k or settings.top_k
        recall_k = k * 3

        vector_results = self.vector_store.query(query, top_k=recall_k)
        bm25_results = self.bm25_store.query(query, top_k=recall_k)

        fused = reciprocal_rank_fusion(
            rank_lists=[vector_results, bm25_results],
            k=settings.rrf_k,
            weights=[settings.vector_weight, settings.bm25_weight],
        )

        return {
            "final": fused[:k],
            "vector_results": vector_results[:k],
            "bm25_results": bm25_results[:k],
        }
