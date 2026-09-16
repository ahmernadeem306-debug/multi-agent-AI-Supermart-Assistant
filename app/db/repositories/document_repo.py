"""Typed, parameterised data access for the ``documents`` knowledge-base table.

The only code permitted to touch the Document ORM model. Used by the RAG
ingestion pipeline and the /documents route.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document


def _to_dict(d: Document) -> dict:
    return {
        "id": d.id,
        "title": d.title,
        "doc_type": d.doc_type,
        "source_path": d.source_path,
        "sha256": d.sha256,
        "ingested_at": d.ingested_at,
        "chunk_count": d.chunk_count,
    }


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_documents(self) -> list[dict]:
        stmt = select(Document).order_by(Document.title.asc())
        return [_to_dict(d) for d in self.session.scalars(stmt)]

    def get_by_source_path(self, source_path: str) -> dict | None:
        stmt = select(Document).where(Document.source_path == source_path)
        row = self.session.scalars(stmt).first()
        return _to_dict(row) if row else None

    def upsert(
        self, *, title: str, doc_type: str, source_path: str, sha256: str, chunk_count: int
    ) -> dict:
        stmt = select(Document).where(Document.source_path == source_path)
        row = self.session.scalars(stmt).first()
        if row is None:
            row = Document(
                title=title,
                doc_type=doc_type,
                source_path=source_path,
                sha256=sha256,
                ingested_at=dt.datetime.utcnow(),
                chunk_count=chunk_count,
            )
            self.session.add(row)
        else:
            row.title = title
            row.doc_type = doc_type
            row.sha256 = sha256
            row.chunk_count = chunk_count
            row.ingested_at = dt.datetime.utcnow()
        self.session.flush()
        return _to_dict(row)

    def delete_by_source_path(self, source_path: str) -> None:
        row = self.session.scalars(
            select(Document).where(Document.source_path == source_path)
        ).first()
        if row is not None:
            self.session.delete(row)
            self.session.flush()
