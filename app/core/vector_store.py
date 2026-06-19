"""
ChromaDB vector store: persistent storage with add, delete, and query operations.
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
            f"ChromaDB initialised, collection: {settings.collection_name}, "
            f"existing documents: {self.collection.count()}"
        )

    # ─── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Batch-write chunks to the vector store."""
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
        logger.info(f"Wrote {len(chunks)} chunks to vector store")

    # ─── Query ─────────────────────────────────────────────────────────────────

    def query(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Vector search; returns a list of results with scores."""
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

    # ─── Delete ────────────────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks associated with a doc_id."""
        results = self.collection.get(
            where={"doc_id": doc_id}, include=["documents"]
        )
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info(f"Deleted doc_id={doc_id}, {len(ids)} chunks removed")
        return len(ids)

    def get_all_doc_ids(self) -> List[Dict[str, Any]]:
        """List all document metadata (deduplicated)."""
        results = self.collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = meta
        return list(seen.values())

    def count(self) -> int:
        return self.collection.count()
