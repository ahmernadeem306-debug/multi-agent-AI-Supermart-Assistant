"""ChromaDB persistence for embedded knowledge-base chunks.

The collection name encodes backend + embedding dimension
(e.g. ``bizagent_kb_local_384``) so switching ``EMBEDDING_BACKEND`` opens a
different collection and cannot corrupt an existing one. A dimension recorded
on the collection that disagrees with the active backend raises a clear
"re-ingest" error.
"""
from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.logging_config import get_logger

logger = get_logger(__name__)


def collection_name(prefix: str, backend: str, dimension: int) -> str:
    return f"{prefix}_{backend}_{dimension}"


class ChromaVectorStore:
    def __init__(self, persist_dir: str, name: str, dimension: int) -> None:
        import chromadb

        self._dimension = dimension
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "dimension": dimension},
            embedding_function=None,
        )
        recorded = self._collection.metadata.get("dimension") if self._collection.metadata else None
        if recorded is not None and int(recorded) != dimension:
            raise RetrievalError(
                f"Chroma collection '{name}' was built with dimension {recorded}, "
                f"but the active embedding backend produces {dimension}. "
                "Delete CHROMA_PERSIST_DIR and re-run `python scripts/ingest_docs.py`."
            )

    @property
    def name(self) -> str:
        return self._collection.name

    def count(self) -> int:
        return self._collection.count()

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(
        self, embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i])
            hits.append(
                {
                    "id": chunk_id,
                    "document": docs[i],
                    "metadata": metas[i] or {},
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine distance -> similarity
                }
            )
        return hits


def build_vector_store(
    embedding_backend, settings: Settings | None = None
) -> ChromaVectorStore:
    """Open (or create) the Chroma collection for the given embedding backend."""
    settings = settings or get_settings()
    name = collection_name(
        settings.rag_collection_prefix, embedding_backend.name, embedding_backend.dimension
    )
    return ChromaVectorStore(settings.chroma_persist_dir, name, embedding_backend.dimension)
