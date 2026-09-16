"""RAG knowledge engine: chunking, embeddings, Chroma store, ingest, retriever."""
from app.rag.retriever import Citation, RetrievalResult, Retriever, build_retriever

__all__ = ["Citation", "RetrievalResult", "Retriever", "build_retriever"]
