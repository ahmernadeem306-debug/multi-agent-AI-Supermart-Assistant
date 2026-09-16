"""SQLAlchemy engine, session factory and declarative base."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.exceptions import ToolExecutionError


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine(database_url: str | None = None):
    """Return the process-wide SQLAlchemy engine, creating it on first use."""
    global _engine, _SessionLocal
    if _engine is None:
        if database_url is None:
            from app.config import get_settings

            database_url = get_settings().database_url
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        _engine = create_engine(database_url, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the process-wide session factory, creating the engine if needed."""
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager yielding a SQLAlchemy session, committing on success."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_readonly_session() -> Iterator[Session]:
    """Yield a session that rejects INSERT/UPDATE/DELETE.

    Used by the MCP tool server, the /metrics and /forecast routes and the RCA
    workflow so that no code path reachable from an agent can mutate the
    database. Enforcement is a ``before_flush`` guard that raises
    :class:`ToolExecutionError` on any pending ORM change — the only way tool
    code ever touches the database. The session also never commits.
    """
    session_factory = get_session_factory()
    session = session_factory()

    @event.listens_for(session, "before_flush")
    def _block_writes(sess: Session, flush_context, instances) -> None:  # noqa: ANN001
        if sess.new or sess.dirty or sess.deleted:
            raise ToolExecutionError("Write attempted on a read-only session.")

    try:
        yield session
    finally:
        session.rollback()
        session.close()


def create_all(database_url: str | None = None) -> None:
    """Create all tables. No migrations (Alembic) in this 5-day project."""
    from app.db import models  # noqa: F401  (ensure models are registered)

    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
