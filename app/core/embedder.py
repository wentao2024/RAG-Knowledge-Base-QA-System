"""
Embedding model wrapper: sentence-transformers with singleton caching.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.config import settings


class Embedder:
    _instance = None  # singleton to avoid reloading the model

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {settings.embed_model}")
        self.model = SentenceTransformer(settings.embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(f"Embedding model loaded, dimension: {self.dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Batch-encode texts; returns a numpy array of shape (n, dim)."""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # normalise for cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.embed_texts([query])[0]

    def get_dim(self) -> int:
        return self.dim
