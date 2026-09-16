"""Ingest the knowledge-base corpus into ChromaDB.

    python scripts/ingest_docs.py            # ingest data/knowledge/, skip unchanged
    python scripts/ingest_docs.py --force    # re-embed every file

Prints a per-document table (title, type, chunks, status) and the total
chunk count now in the collection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.rag.embeddings import get_embedding_backend  # noqa: E402
from app.rag.ingest import ingest_directory  # noqa: E402
from app.rag.vector_store import build_vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge-base documents into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Re-embed even unchanged files.")
    parser.add_argument("--dir", default=None, help="Override the knowledge directory.")
    args = parser.parse_args()

    settings = get_settings()
    report = ingest_directory(args.dir, settings=settings, force=args.force)

    print(f"\n{'Title':<42} {'Type':<10} {'Chunks':>7}  Status")
    print("-" * 78)
    for r in report.results:
        note = f"  ({r.error})" if r.error else ""
        print(f"{r.title[:41]:<42} {r.doc_type:<10} {r.chunk_count:>7}  {r.status}{note}")

    backend = get_embedding_backend(settings)
    store = build_vector_store(backend, settings)
    print("-" * 78)
    print(f"Collection '{store.name}' now holds {store.count()} chunks "
          f"({backend.name} embeddings, dim {backend.dimension}).")
    if any(r.status == "error" for r in report.results):
        sys.exit(1)


if __name__ == "__main__":
    main()
