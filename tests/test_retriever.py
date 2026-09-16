"""Golden-question retrieval against the real corpus with local embeddings.

Uses the sentence-transformers backend so no API key is needed. Skips if the
model cannot be loaded (offline, no cache).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.rag import ingest as ingest_mod
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "knowledge"

GOLDEN = [
    ("how long do customers have to return frozen goods?", "returns_and_refunds_policy.md"),
    ("what is FreshFarm's standard lead time for produce?", "supplier_contract_freshfarm.md"),
    ("when must a theft shrinkage event be investigated?", "shrinkage_and_loss_prevention_sop.md"),
    ("how often are high velocity SKUs cycle counted?", "inventory_count_and_discrepancy_sop.md"),
    ("what are the FIFO and FEFO rotation rules for shelves?", "shelf_stocking_sop.md"),
]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    try:
        from app.rag.embeddings import LocalEmbeddings

        backend = LocalEmbeddings()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"local embedding model unavailable: {exc}")

    persist = tmp_path_factory.mktemp("chroma")
    store = ChromaVectorStore(str(persist), f"golden_{backend.name}_{backend.dimension}", backend.dimension)
    for path in sorted(_CORPUS.glob("*.md")):
        ingest_mod.ingest_file(
            path, embedding_backend=backend, store=store, settings=get_settings(), force=True
        )
    return Retriever(backend, store, top_k=5, min_score=0.1)


@pytest.mark.parametrize("question,expected_doc", GOLDEN, ids=[q[:32] for q, _ in GOLDEN])
def test_golden_question_returns_expected_document(retriever, question, expected_doc):
    result = retriever.retrieve(question)
    assert result.found
    top3 = [Path(c.source_path).name for c in result.citations[:3]]
    assert expected_doc in top3, f"expected {expected_doc} in top-3, got {top3}"
    assert all(c.snippet and c.doc_title for c in result.citations)


def test_unrelated_query_returns_no_policy(retriever):
    result = retriever.retrieve("what colour is the sky on a spring afternoon in Lisbon?")
    # Either nothing clears threshold, or whatever does is genuinely low-signal.
    assert (not result.found) or result.citations[0].score < 0.4
