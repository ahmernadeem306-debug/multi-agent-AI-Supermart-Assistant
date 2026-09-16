"""Knowledge-base routes: list, search, upload and re-ingest documents."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.exceptions import RetrievalError
from app.db.base import get_readonly_session
from app.db.repositories.document_repo import DocumentRepository
from app.logging_config import get_logger
from app.rag.ingest import SUPPORTED_SUFFIXES, ingest_directory
from app.rag.retriever import build_retriever

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    doc_type: str | None = None


class ReingestRequest(BaseModel):
    force: bool = False


def _safe_filename(name: str) -> str:
    base = Path(name or "").name  # drop any directory components (traversal guard)
    base = _SAFE_NAME.sub("_", base).strip("._") or "upload"
    return base


@router.get("")
def list_documents() -> dict:
    with get_readonly_session() as session:
        docs = DocumentRepository(session).list_documents()
    return {
        "documents": docs,
        "count": len(docs),
        "total_chunks": sum(d["chunk_count"] for d in docs),
    }


@router.post("/search")
def search_documents(request: SearchRequest) -> dict:
    try:
        result = build_retriever().retrieve(request.query, doc_type=request.doc_type)
    except RetrievalError as exc:
        return {"found": False, "message": exc.message, "citations": []}
    return {
        "found": result.found,
        "citations": [c.model_dump(mode="json") for c in result.citations],
    }


@router.post("/reingest")
def reingest(request: ReingestRequest) -> dict:
    try:
        report = ingest_directory(force=request.force)
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    return {
        "results": [r.__dict__ for r in report.results],
        "total_chunks": report.total_chunks,
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
        )

    knowledge_dir = Path(settings.knowledge_dir).resolve()
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    dest = (knowledge_dir / _safe_filename(file.filename)).resolve()
    if dest.parent != knowledge_dir:
        raise HTTPException(status_code=400, detail="Invalid destination path.")
    dest.write_bytes(data)
    logger.info("document_uploaded", filename=dest.name, bytes=len(data))

    report = ingest_directory(force=False)
    entry = next((r for r in report.results if Path(r.source_path).name == dest.name), None)
    return {
        "filename": dest.name,
        "ingested": entry.__dict__ if entry else None,
        "total_chunks": report.total_chunks,
    }
