"""
File upload API — PDF upload, processing, and indexing.
"""
import os
import uuid
import time
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles
from loguru import logger

from app.config import settings
from app.models.schemas import UploadResponse, ListDocumentsResponse, DocumentInfo
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore
from app.core.bm25_store import BM25Store
from app.core.parent_store import ParentStore

router = APIRouter(prefix="/documents", tags=["Documents"])

os.makedirs(settings.upload_dir, exist_ok=True)

# Document metadata persistence
DOC_META_PATH = os.path.join(settings.chroma_persist_dir, "doc_meta.json")


def _load_doc_meta() -> dict:
    if os.path.exists(DOC_META_PATH):
        with open(DOC_META_PATH) as f:
            return json.load(f)
    return {}


def _save_doc_meta(meta: dict):
    os.makedirs(os.path.dirname(DOC_META_PATH), exist_ok=True)
    with open(DOC_META_PATH, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document.

    Pipeline:
    1. Save the file
    2. Extract text
    3. Smart chunking
    4. Write to vector store + BM25 index
    5. Record metadata
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # File size limit: 50 MB
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File must not exceed 50 MB")

    # Save file
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(settings.upload_dir, safe_filename)
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"File saved: {save_path}, size: {len(content)/1024:.1f} KB")

    try:
        processor = DocumentProcessor()

        if settings.enable_parent_child:
            doc_id, parent_chunks, child_chunks = processor.process_pdf_parent_child(
                save_path, file.filename
            )
            index_chunks = child_chunks   # small chunks go into vector store / BM25
            display_count = len(parent_chunks)
            msg = (
                f"Upload successful! {len(parent_chunks)} parent chunks (fed to LLM), "
                f"{len(child_chunks)} child chunks (used for precise retrieval)"
            )
        else:
            doc_id, chunks = processor.process_pdf(save_path, file.filename)
            index_chunks = chunks
            display_count = len(chunks)
            msg = f"Upload successful! Created {len(chunks)} text chunks"

        if not index_chunks:
            raise HTTPException(status_code=422, detail="PDF content is empty or cannot be parsed")

        # Write to index
        vector_store = VectorStore()
        bm25_store = BM25Store()
        vector_store.add_chunks(index_chunks)
        bm25_store.add_chunks(index_chunks)

        if settings.enable_parent_child:
            ParentStore().add_parents(parent_chunks)

        # Save metadata
        meta = _load_doc_meta()
        meta[doc_id] = {
            "filename": file.filename,
            "chunks_count": display_count,
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": save_path,
        }
        _save_doc_meta(meta)

        logger.info(f"Document indexed: {file.filename}, doc_id={doc_id}, chunks={display_count}")

        return UploadResponse(
            filename=file.filename,
            doc_id=doc_id,
            chunks_count=display_count,
            message=msg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("", response_model=ListDocumentsResponse)
async def list_documents():
    """List all indexed documents."""
    meta = _load_doc_meta()
    docs = []
    for doc_id, info in meta.items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=info.get("filename", "unknown"),
                chunks_count=info.get("chunks_count", 0),
                upload_time=info.get("upload_time", ""),
            )
        )
    return ListDocumentsResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its index."""
    meta = _load_doc_meta()
    if doc_id not in meta:
        raise HTTPException(status_code=404, detail="Document not found")

    info = meta[doc_id]
    filename = info.get("filename", "unknown")

    # Delete index
    vector_store = VectorStore()
    bm25_store = BM25Store()
    v_count = vector_store.delete_by_doc_id(doc_id)
    b_count = bm25_store.delete_by_doc_id(doc_id)
    if settings.enable_parent_child:
        ParentStore().delete_by_doc_id(doc_id)

    # Delete file
    file_path = info.get("file_path", "")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # Update metadata
    del meta[doc_id]
    _save_doc_meta(meta)

    return {
        "message": f"Document '{filename}' deleted",
        "vector_chunks_deleted": v_count,
        "bm25_chunks_deleted": b_count,
    }
