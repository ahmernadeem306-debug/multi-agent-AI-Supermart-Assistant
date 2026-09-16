"""Knowledge-base ingestion: load -> clean -> chunk -> embed -> upsert.

Each source file is recorded in the ``documents`` table with a sha256 of its
text, so re-ingesting an unchanged file is a no-op and a changed file has its
old chunks deleted from Chroma before the new ones are added.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.db.base import get_session
from app.db.repositories.document_repo import DocumentRepository
from app.logging_config import get_logger
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingBackend, get_embedding_backend
from app.rag.vector_store import ChromaVectorStore, build_vector_store

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SCRIPT_RE = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)

_DOC_TYPE_HINTS = (
    ("contract", "contract"),
    ("policy", "policy"),
    ("handbook", "handbook"),
    ("sop", "sop"),
)


@dataclass
class IngestResult:
    source_path: str
    title: str
    doc_type: str
    chunk_count: int
    status: str  # "ingested" | "updated" | "unchanged"
    error: str | None = None


@dataclass
class IngestReport:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_doc_type(path: Path) -> str:
    stem = path.stem.lower()
    for hint, value in _DOC_TYPE_HINTS:
        if hint in stem:
            return value
    return "reference"


def _title_from(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").title()


def load_text(path: Path) -> str:
    """Extract plain text from a supported knowledge file."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        raw = "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    elif suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RetrievalError("Reading .docx requires 'python-docx'.") from exc
        raw = "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    else:
        raise RetrievalError(f"Unsupported knowledge file type: {path.suffix}")
    cleaned = _SCRIPT_RE.sub("", raw)
    return _CONTROL_RE.sub("", cleaned).strip()


def ingest_file(
    path: Path,
    *,
    embedding_backend: EmbeddingBackend,
    store: ChromaVectorStore,
    settings: Settings,
    force: bool = False,
) -> IngestResult:
    """Ingest a single file; a no-op if its sha256 is unchanged."""
    path = path.resolve()
    text = load_text(path)
    if not text:
        return IngestResult(str(path), path.stem, infer_doc_type(path), 0, "unchanged", "empty file")

    sha = _sha256(text)
    title = _title_from(path, text)
    doc_type = infer_doc_type(path)
    source_key = str(path)

    with get_session() as session:
        repo = DocumentRepository(session)
        existing = repo.get_by_source_path(source_key)
        if existing and existing["sha256"] == sha and not force:
            return IngestResult(source_key, title, doc_type, existing["chunk_count"], "unchanged")

    chunks = chunk_text(text, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap)
    doc_id = _sha256(source_key)[:16]
    store.delete_by_doc_id(doc_id)

    embeddings = embedding_backend.embed_documents([c.text for c in chunks])
    store.add(
        ids=[f"{doc_id}:{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": doc_id,
                "title": title,
                "doc_type": doc_type,
                "source_path": source_key,
                "chunk_index": c.chunk_index,
                "section": c.section or "",
            }
            for c in chunks
        ],
    )

    with get_session() as session:
        repo = DocumentRepository(session)
        status = "updated" if repo.get_by_source_path(source_key) else "ingested"
        repo.upsert(
            title=title,
            doc_type=doc_type,
            source_path=source_key,
            sha256=sha,
            chunk_count=len(chunks),
        )

    logger.info("document_ingested", title=title, chunks=len(chunks), status=status)
    return IngestResult(source_key, title, doc_type, len(chunks), status)


def ingest_directory(
    directory: str | Path | None = None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> IngestReport:
    """Ingest every supported file in the knowledge directory."""
    settings = settings or get_settings()
    directory = Path(directory or settings.knowledge_dir)
    if not directory.exists():
        raise RetrievalError(f"Knowledge directory not found: {directory}")

    backend = get_embedding_backend(settings)
    store = build_vector_store(backend, settings)

    report = IngestReport()
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        try:
            report.results.append(
                ingest_file(
                    path, embedding_backend=backend, store=store, settings=settings, force=force
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            logger.error("document_ingest_failed", path=str(path), error=str(exc))
            report.results.append(
                IngestResult(str(path), path.stem, infer_doc_type(path), 0, "error", str(exc))
            )
    return report
