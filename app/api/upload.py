"""
文件上传 API  PDF 上传、处理、索引
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

router = APIRouter(prefix="/documents", tags=["Documents"])

os.makedirs(settings.upload_dir, exist_ok=True)

# 文档元信息持久化
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
    上传 PDF 文档并索引
    
    流程：
    1. 保存文件
    2. 提取文本
    3. 智能分块
    4. 写入向量库 + BM25 索引
    5. 记录元信息
    """
    # 校验文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式")

    # 文件大小限制 50MB
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件不能超过 50MB")

    # 保存文件
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(settings.upload_dir, safe_filename)
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"文件保存: {save_path}，大小: {len(content)/1024:.1f} KB")

    try:
        # 处理文档
        processor = DocumentProcessor()
        doc_id, chunks = processor.process_pdf(save_path, file.filename)

        if not chunks:
            raise HTTPException(status_code=422, detail="PDF 内容为空或无法解析")

        # 写入索引
        vector_store = VectorStore()
        bm25_store = BM25Store()
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)

        # 保存元信息
        meta = _load_doc_meta()
        meta[doc_id] = {
            "filename": file.filename,
            "chunks_count": len(chunks),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": save_path,
        }
        _save_doc_meta(meta)

        logger.info(f"文档索引完成: {file.filename}, doc_id={doc_id}, chunks={len(chunks)}")

        return UploadResponse(
            filename=file.filename,
            doc_id=doc_id,
            chunks_count=len(chunks),
            message=f"上传成功！已创建 {len(chunks)} 个文本块",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("", response_model=ListDocumentsResponse)
async def list_documents():
    """列出所有已索引文档"""
    meta = _load_doc_meta()
    docs = []
    for doc_id, info in meta.items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=info.get("filename", "未知"),
                chunks_count=info.get("chunks_count", 0),
                upload_time=info.get("upload_time", ""),
            )
        )
    return ListDocumentsResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及其索引"""
    meta = _load_doc_meta()
    if doc_id not in meta:
        raise HTTPException(status_code=404, detail="文档不存在")

    info = meta[doc_id]
    filename = info.get("filename", "未知")

    # 删除索引
    vector_store = VectorStore()
    bm25_store = BM25Store()
    v_count = vector_store.delete_by_doc_id(doc_id)
    b_count = bm25_store.delete_by_doc_id(doc_id)

    # 删除文件
    file_path = info.get("file_path", "")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # 更新元信息
    del meta[doc_id]
    _save_doc_meta(meta)

    return {
        "message": f"文档 '{filename}' 已删除",
        "vector_chunks_deleted": v_count,
        "bm25_chunks_deleted": b_count,
    }
