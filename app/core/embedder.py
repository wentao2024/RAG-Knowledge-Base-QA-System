"""
Embedding 模型封装: sentence-transformers 带缓存
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.config import settings


class Embedder:
    _instance = None  # 单例，避免重复加载模型

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"加载 Embedding 模型: {settings.embed_model}")
        self.model = SentenceTransformer(settings.embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(f"Embedding 模型加载完成，维度: {self.dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """批量编码文本，返回 numpy array (n, dim)"""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化，便于余弦相似度计算
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """编码单条查询"""
        return self.embed_texts([query])[0]

    def get_dim(self) -> int:
        return self.dim
