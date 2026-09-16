"""Cited retrieval over the Chroma knowledge base.

Top-k cosine similarity with an optional ``doc_type`` filter and a minimum
score threshold. If nothing clears the threshold the result is explicitly
empty (``found=False``) — callers must never fabricate policy text.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.logging_config import get_logger
from app.rag.embeddings import EmbeddingBackend, get_embedding_backend
from app.rag.vector_store import ChromaVectorStore, build_vector_store

logger = get_logger(__name__)

_SNIPPET_CHARS = 320


class Citation(BaseModel):
    doc_title: str
    source_path: str
    chunk_index: int
    section: str | None = None
    score: float
    snippet: str


class RetrievalResult(BaseModel):
    query: str
    found: bool
    citations: list[Citation] = []
    chunks: list[str] = []

    @property
    def context_blocks(self) -> list[str]:
        """Delimited, labelled blocks for prompt-injection-safe prompting."""
        blocks = []
        for i, (chunk, cite) in enumerate(zip(self.chunks, self.citations), start=1):
            blocks.append(
                f"[REFERENCE {i}] title={cite.doc_title!r} section={cite.section or 'n/a'!r} "
                f"score={cite.score:.3f}\n{chunk}"
            )
        return blocks


class Retriever:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: ChromaVectorStore,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> None:
        self._embeddings = embedding_backend
        self._store = vector_store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str, *, doc_type: str | None = None) -> RetrievalResult:
        if self._store.count() == 0:
            raise RetrievalError(
                "The knowledge base is empty. Run `python scripts/ingest_docs.py` first."
            )
        embedding = self._embeddings.embed_query(query)
        where = {"doc_type": doc_type} if doc_type else None
        hits = self._store.query(embedding, self._top_k, where=where)
        kept = [h for h in hits if h["score"] >= self._min_score]
        if not kept:
            logger.info("retrieval_below_threshold", query=query[:120], best=hits[0]["score"] if hits else None)
            return RetrievalResult(query=query, found=False)

        citations = [
            Citation(
                doc_title=h["metadata"].get("title", "Unknown"),
                source_path=h["metadata"].get("source_path", ""),
                chunk_index=int(h["metadata"].get("chunk_index", 0)),
                section=h["metadata"].get("section") or None,
                score=round(h["score"], 4),
                snippet=h["document"][:_SNIPPET_CHARS].strip(),
            )
            for h in kept
        ]
        return RetrievalResult(
            query=query,
            found=True,
            citations=citations,
            chunks=[h["document"] for h in kept],
        )


_retriever_singleton: Retriever | None = None


def build_retriever(settings: Settings | None = None) -> Retriever:
    """Return the process-wide Retriever, constructing it on first use.

    The embedding backend (a loaded sentence-transformers model, or a remote
    LLM client) and the Chroma collection handle are expensive to build; every
    caller (``/query``, ``/rca``, ``/documents/search``) shares one instance
    instead of rebuilding it per request.
    """
    global _retriever_singleton
    if _retriever_singleton is not None:
        return _retriever_singleton
    settings = settings or get_settings()
    backend = get_embedding_backend(settings)
    store = build_vector_store(backend, settings)
    _retriever_singleton = Retriever(
        backend, store, top_k=settings.rag_top_k, min_score=settings.rag_min_score
    )
    return _retriever_singleton
