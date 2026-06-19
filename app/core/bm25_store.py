"""
BM25 sparse retrieval using jieba tokenisation + rank_bm25.
Supports incremental updates and persistence.
"""
import os
import pickle
from typing import List, Dict, Any
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi
from loguru import logger
from app.config import settings
from app.core.document_processor import Chunk

BM25_INDEX_PATH = os.path.join(settings.chroma_persist_dir, "bm25_index.pkl")

# Load custom dictionary if available
jieba.setLogLevel("ERROR")


def _tokenize(text: str) -> List[str]:
    """Tokenise text with jieba; remove stopwords and single-character tokens."""
    tokens = jieba.lcut(text)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 1]
    return tokens


class BM25Store:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None
        self._load()

    # ─── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(BM25_INDEX_PATH):
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                self.documents = data["documents"]
                self.tokenized_corpus = data["tokenized_corpus"]
                self._rebuild_index()
                logger.info(f"BM25 index loaded, document count: {len(self.documents)}")
            except Exception as e:
                logger.warning(f"BM25 index load failed, recreating: {e}")
                self._reset()
        else:
            self._reset()

    def _save(self):
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )

    def _reset(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def _rebuild_index(self):
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ─── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            self.documents.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
            )
            self.tokenized_corpus.append(tokens)
        self._rebuild_index()
        self._save()
        logger.info(f"BM25 added {len(chunks)} chunks, total: {len(self.documents)}")

    # ─── Query ─────────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.top_k
        if self.bm25 is None or not self.documents:
            return []
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Take top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in indexed:
            if score > 0:
                doc = self.documents[idx]
                results.append(
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": float(score),
                    }
                )
        return results

    # ─── Delete ────────────────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> int:
        original_len = len(self.documents)
        filtered = [
            (doc, tokens)
            for doc, tokens in zip(self.documents, self.tokenized_corpus)
            if doc["metadata"].get("doc_id") != doc_id
        ]
        if filtered:
            self.documents, self.tokenized_corpus = zip(*filtered)
            self.documents = list(self.documents)
            self.tokenized_corpus = list(self.tokenized_corpus)
        else:
            self.documents = []
            self.tokenized_corpus = []
        self._rebuild_index()
        self._save()
        deleted = original_len - len(self.documents)
        logger.info(f"BM25 deleted doc_id={doc_id}, {deleted} chunks removed")
        return deleted
