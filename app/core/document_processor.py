"""
Document processor: PDF parsing, text cleaning, and intelligent chunking.
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

    # ─── PDF parsing ───────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract PDF text page by page, preserving page number metadata."""
        pages = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                pages.append({"page": page_num, "text": text})
        doc.close()
        logger.info(f"Extracted {len(pages)} pages of text from: {pdf_path}")
        return pages

    # ─── Text cleaning ─────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Clean raw text extracted from a PDF."""
        # Remove common header/footer patterns (page numbers, etc.)
        text = re.sub(r"\n(\d+)\n", "\n", text)
        # Merge hyphenated line breaks (English)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Remove mid-paragraph line breaks
        text = re.sub(r"(?<=[^\n])\n(?=[^\n])", "", text)
        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove special control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalise full-width spaces
        text = text.replace("　", " ").strip()
        return text

    # ─── Chunking strategy ─────────────────────────────────────────────────────

    def split_into_chunks(
        self, pages: List[Dict[str, Any]], filename: str, doc_id: str
    ) -> List[Chunk]:
        """
        Intelligent chunking:
        1. Prefer splitting on paragraph boundaries.
        2. Fall back to sliding-window splitting for overly long paragraphs.
        3. Preserve page number, filename, and other metadata.
        """
        chunks: List[Chunk] = []

        for page_info in pages:
            page_num = page_info["page"]
            text = page_info["text"]

            # Coarse split on paragraph breaks
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
                    # Paragraph itself is too long: apply sliding window
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

            # Flush remaining buffer
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

        logger.info(f"Document {filename} chunked, total {len(chunks)} chunks")
        return chunks

    def _sliding_window(self, text: str) -> List[str]:
        """Sliding-window split for oversized text."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ─── End-to-end processing ─────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str, filename: str) -> Tuple[str, List[Chunk]]:
        """Full PDF pipeline: extract → clean → chunk. Returns (doc_id, chunks)."""
        doc_id = str(uuid.uuid4())
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(pages, filename, doc_id)
        return doc_id, chunks

    def process_pdf_parent_child(
        self, pdf_path: str, filename: str
    ) -> Tuple[str, List["Chunk"], List["Chunk"]]:
        """
        Process a PDF in Parent-Child mode.

        Returns (doc_id, parent_chunks, child_chunks):
          - parent_chunks: large chunks (~chunk_size chars), stored in ParentStore, fed to LLM.
          - child_chunks: small chunks (~child_chunk_size chars), stored in vector store + BM25, used for embedding.
        """
        doc_id = str(uuid.uuid4())
        pages = self.extract_text_from_pdf(pdf_path)
        parent_chunks = self.split_into_chunks(pages, filename, doc_id)
        child_chunks = []
        for parent in parent_chunks:
            child_chunks.extend(self._split_into_children(parent))
        logger.info(
            f"Parent-Child chunking complete: {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks"
        )
        return doc_id, parent_chunks, child_chunks

    def _split_into_children(self, parent: "Chunk") -> List["Chunk"]:
        """Split a parent Chunk into smaller child Chunks, each carrying the parent_id."""
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
        """Sliding-window split using child_chunk_size / child_chunk_overlap."""
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
