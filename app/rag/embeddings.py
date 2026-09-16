"""Embedding backends for the RAG knowledge engine.

``EmbeddingBackend`` is the interface; ``LocalEmbeddings`` (default,
sentence-transformers/all-MiniLM-L6-v2) is the supported implementation.
Groq does not expose an embeddings API, so ``EMBEDDING_BACKEND=groq`` fails
fast with a clear error rather than silently degrading. The active backend
is chosen by ``EMBEDDING_BACKEND``. Query embeddings are LRU-cached so
repeated identical queries in one run cost nothing.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from app.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.llm.provider import LLMProvider
from app.logging_config import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 32


class EmbeddingBackend(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class RemoteEmbeddings:
    """Wraps an :class:`LLMProvider` so RAG code never imports the SDK.

    No currently supported LLMProvider implements ``embed`` (Groq has no
    embeddings API), so this backend always raises when used. It is kept
    for providers that do support remote embeddings in the future.
    """

    name = "groq"

    def __init__(self, provider: LLMProvider, dimension: int | None = None) -> None:
        self._provider = provider
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=10), reraise=True)
    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            vectors.extend(self._embed(texts[start : start + _BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return _cached_query_embedding(self, text)


class LocalEmbeddings:
    """Offline fallback using sentence-transformers/all-MiniLM-L6-v2 (384-d)."""

    name = "local"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RetrievalError(
                "EMBEDDING_BACKEND=local requires 'sentence-transformers'. "
                "Install it (pip install sentence-transformers). EMBEDDING_BACKEND=local "
                "is the only supported embedding backend."
            ) from exc
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts, batch_size=_BATCH_SIZE, normalize_embeddings=False)]

    def embed_query(self, text: str) -> list[float]:
        return _cached_query_embedding(self, text)


@lru_cache(maxsize=512)
def _cached_query_embedding(backend: "EmbeddingBackend", text: str) -> list[float]:
    if isinstance(backend, LocalEmbeddings):
        return [float(x) for x in backend._model.encode([text], normalize_embeddings=False)[0]]
    return backend._embed([text])[0]  # type: ignore[attr-defined]


def get_embedding_backend(settings: Settings | None = None, provider: LLMProvider | None = None) -> EmbeddingBackend:
    """Return the configured embedding backend (``local`` or ``groq``)."""
    settings = settings or get_settings()
    choice = settings.embedding_backend.lower()
    if choice == "local":
        return LocalEmbeddings(settings.local_embedding_model)
    if choice == "groq":
        if provider is None:
            from app.llm.groq_client import GroqProvider

            provider = GroqProvider(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model,
                timeout_seconds=settings.groq_timeout_seconds,
            )
        return RemoteEmbeddings(provider)
    raise RetrievalError(f"Unknown EMBEDDING_BACKEND '{settings.embedding_backend}'.")
