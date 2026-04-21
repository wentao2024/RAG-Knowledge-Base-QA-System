"""
ChromaDB 向量存储：持久化、增删查
"""
import os
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from app.config import settings
from app.core.embedder import Embedder
from app.core.document_processor import Chunk


class VectorStore:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB 初始化完成，集合: {settings.collection_name}，"
            f"现有文档数: {self.collection.count()}"
        )

    # ─── 写入 ──────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """批量写入 chunks 到向量库"""
        if not chunks:
            return
        texts = [c.content for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embeddings = self.embedder.embed_texts(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"写入 {len(chunks)} 个块到向量库")

    # ─── 查询 ──────────────────────────────────────────────────────────────────

    def query(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """向量检索，返回带 score 的结果列表"""
        k = top_k or settings.top_k
        query_emb = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(k, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma cosine distance → similarity
            score = 1.0 - dist
            hits.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "score": float(score),
                }
            )
        return hits

    # ─── 删除 ──────────────────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关 chunk"""
        results = self.collection.get(
            where={"doc_id": doc_id}, include=["documents"]
        )
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info(f"删除 doc_id={doc_id}，共 {len(ids)} 个块")
        return len(ids)

    def get_all_doc_ids(self) -> List[Dict[str, Any]]:
        """列出所有文档元信息（去重）"""
        results = self.collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = meta
        return list(seen.values())

    def count(self) -> int:
        return self.collection.count()
