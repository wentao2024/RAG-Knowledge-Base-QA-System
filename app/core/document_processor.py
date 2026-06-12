"""
文档处理器 支持中文PDF解析、文本清洗、智能分块
"""
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from loguru import logger
from app.config import settings


class Chunk:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        child_chunk_size: int = None,
        child_chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.child_chunk_size = child_chunk_size or settings.child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap or settings.child_chunk_overlap

    # ─── PDF解析 ───────────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """逐页提取PDF文本 保留页码元信息"""
        pages = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                pages.append({"page": page_num, "text": text})
        doc.close()
        logger.info(f"提取 {len(pages)} 页文本，来自: {pdf_path}")
        return pages

    # ─── 文本清洗 ──────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """清洗PDF提取的原始文本"""
        # 删除页眉页脚常见模式（页码等）
        text = re.sub(r"\n(\d+)\n", "\n", text)
        # 合并连字符断行（英文）
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # 中文段落：去除段内多余换行
        text = re.sub(r"(?<=[^\n])\n(?=[^\n])", "", text)
        # 多个空行压缩为一个
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 清除特殊控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # 全角标点统一
        text = text.replace("　", " ").strip()
        return text

    # ─── 分块策略 ──────────────────────────────────────────────────────────────

    def split_into_chunks(
        self, pages: List[Dict[str, Any]], filename: str, doc_id: str
    ) -> List[Chunk]:
        """
        智能分块：
        1. 优先按段落分割
        2. 段落过长时再按 chunk_size 滑窗切分
        3. 保留页码、文件名等元数据
        """
        chunks: List[Chunk] = []

        for page_info in pages:
            page_num = page_info["page"]
            text = page_info["text"]

            # 按段落粗分
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) <= self.chunk_size:
                    buffer = (buffer + "\n\n" + para).strip() if buffer else para
                else:
                    if buffer:
                        chunks.append(
                            Chunk(
                                content=buffer,
                                metadata={
                                    "source": filename,
                                    "doc_id": doc_id,
                                    "page": page_num,
                                },
                            )
                        )
                    # 段落本身超长则滑窗切分
                    if len(para) > self.chunk_size:
                        sub_chunks = self._sliding_window(para)
                        for sc in sub_chunks:
                            chunks.append(
                                Chunk(
                                    content=sc,
                                    metadata={
                                        "source": filename,
                                        "doc_id": doc_id,
                                        "page": page_num,
                                    },
                                )
                            )
                        buffer = ""
                    else:
                        buffer = para

            # 剩余 buffer
            if buffer:
                chunks.append(
                    Chunk(
                        content=buffer,
                        metadata={
                            "source": filename,
                            "doc_id": doc_id,
                            "page": page_num,
                        },
                    )
                )

        logger.info(f"文档 {filename} 分块完成，共 {len(chunks)} 个块")
        return chunks

    def _sliding_window(self, text: str) -> List[str]:
        """按字符滑窗切分超长文本"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ─── 一体化处理 ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str, filename: str) -> Tuple[str, List[Chunk]]:
        """完整处理PDF 提取 → 清洗 → 分块，返回 (doc_id, chunks)"""
        doc_id = str(uuid.uuid4())
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(pages, filename, doc_id)
        return doc_id, chunks

    def process_pdf_parent_child(
        self, pdf_path: str, filename: str
    ) -> Tuple[str, List["Chunk"], List["Chunk"]]:
        """
        Parent-Child 模式处理 PDF。

        返回 (doc_id, parent_chunks, child_chunks)：
          - parent_chunks：大块（~chunk_size 字），存入 ParentStore，喂给 LLM
          - child_chunks：小块（~child_chunk_size 字），存入向量库 + BM25，做 Embedding
        """
        doc_id = str(uuid.uuid4())
        pages = self.extract_text_from_pdf(pdf_path)
        parent_chunks = self.split_into_chunks(pages, filename, doc_id)
        child_chunks = []
        for parent in parent_chunks:
            child_chunks.extend(self._split_into_children(parent))
        logger.info(
            f"Parent-Child 分块完成: {len(parent_chunks)} 父块, {len(child_chunks)} 子块"
        )
        return doc_id, parent_chunks, child_chunks

    def _split_into_children(self, parent: "Chunk") -> List["Chunk"]:
        """将一个父 Chunk 拆分为若干小子 Chunk，子 Chunk 携带 parent_id。"""
        sub_texts = self._sliding_window_child(parent.content)
        children = []
        for text in sub_texts:
            child = Chunk(
                content=text,
                metadata={
                    **parent.metadata,
                    "parent_id": parent.id,
                    "chunk_type": "child",
                },
            )
            children.append(child)
        return children

    def _sliding_window_child(self, text: str) -> List[str]:
        """按 child_chunk_size / child_chunk_overlap 滑窗切分。"""
        chunks = []
        start = 0
        size = self.child_chunk_size
        overlap = self.child_chunk_overlap
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks
