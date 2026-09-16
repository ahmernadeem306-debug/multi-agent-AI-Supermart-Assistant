"""RAG ingestion: chunking boundaries/overlap, metadata, sha256 idempotency."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.chunking import chunk_text
from app.rag.ingest import ingest_file
from app.rag.vector_store import ChromaVectorStore


class _FakeEmbeddings:
    name = "fake"
    dimension = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# --------------------------------------------------------------------- chunking
def test_chunk_size_and_overlap_boundaries():
    text = "# Heading\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = chunk_text(text, chunk_size=400, overlap=80)

    assert len(chunks) >= 3
    assert all(c.chunk_index == i for i, c in enumerate(chunks))
    # every chunk stays within a sane bound of the target size
    assert max(len(c.text) for c in chunks) <= 400 + 80 + 120
    # consecutive chunks share an overlap fragment
    for a, b in zip(chunks, chunks[1:]):
        assert a.text[-40:] in b.text


def test_chunk_metadata_tracks_headings():
    text = "# Doc\n\n## Alpha\n\nAlpha body text here.\n\n## Beta\n\nBeta body text here."
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    sections = {c.section for c in chunks}
    assert "Alpha" in sections and "Beta" in sections


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=10) == []


# -------------------------------------------------------------------- ingestion
@pytest.fixture()
def rag_env(tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma"), "test_kb_fake_16", 16)
    doc = tmp_path / "warehouse_sop.md"
    unique = tmp_path.name  # keep content (and its sha256) distinct per test
    doc.write_text(
        f"# Warehouse SOP {unique}\n\n"
        + "\n\n".join(f"Rule {i} for {unique}: " + "detail " * 30 for i in range(8))
    )
    return store, doc, get_settings()


def test_ingest_records_document_and_chunks(rag_env, db_session):
    store, doc, settings = rag_env
    result = ingest_file(doc, embedding_backend=_FakeEmbeddings(), store=store, settings=settings)

    assert result.status == "ingested"
    assert result.doc_type == "sop"
    assert result.chunk_count > 1
    assert store.count() == result.chunk_count

    from app.db.repositories.document_repo import DocumentRepository

    row = DocumentRepository(db_session).get_by_source_path(str(doc.resolve()))
    assert row is not None and row["chunk_count"] == result.chunk_count


def test_reingest_unchanged_file_is_a_noop(rag_env):
    store, doc, settings = rag_env
    first = ingest_file(doc, embedding_backend=_FakeEmbeddings(), store=store, settings=settings)
    count_after_first = store.count()

    second = ingest_file(doc, embedding_backend=_FakeEmbeddings(), store=store, settings=settings)
    assert second.status == "unchanged"
    assert store.count() == count_after_first == first.chunk_count


def test_reingest_modified_file_replaces_old_chunks(rag_env):
    store, doc, settings = rag_env
    ingest_file(doc, embedding_backend=_FakeEmbeddings(), store=store, settings=settings)

    doc.write_text(f"# Warehouse SOP {doc.parent.name} v2\n\nOne short rule only.")
    updated = ingest_file(doc, embedding_backend=_FakeEmbeddings(), store=store, settings=settings)

    assert updated.status == "updated"
    assert updated.chunk_count == 1
    assert store.count() == 1  # old chunks deleted, not accumulated
