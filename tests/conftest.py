"""Shared pytest fixtures.

Sets test environment variables (a dummy Groq key, a temp-file SQLite
database) before any app module is imported, so tests never touch
data/bizagent.db or require a real GROQ_API_KEY.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.close(_tmp_db_fd)

os.environ.setdefault("GROQ_API_KEY", "test-key-for-pytest-do-not-use")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp_db_path).as_posix()}"
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())

from app.db.base import create_all, get_session_factory  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _setup_database():
    create_all()
    yield


@pytest.fixture()
def db_session():
    """A SQLAlchemy session whose changes are rolled back after the test."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def seeded_db(_setup_database):
    """Run the real synthetic seeder into the temp test database once.

    Services, MCP tools and the /metrics routes are exercised against the
    same deterministic dataset (and the four planted anomalies) the demo
    uses. The real data/anomalies.json is saved and restored so the seeder
    run does not touch the working tree.
    """
    import scripts.seed_db as seeder

    anomalies_path = Path(seeder.__file__).resolve().parent.parent / "data" / "anomalies.json"
    original = anomalies_path.read_text(encoding="utf-8") if anomalies_path.exists() else None
    seeder.seed(reset=True)
    if original is not None:
        anomalies_path.write_text(original, encoding="utf-8")
    yield


@pytest.fixture()
def read_session(seeded_db):
    """A plain read session over the seeded dataset (no rollback needed)."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def planted_anomalies():
    """The four planted anomalies recorded by the seeder, keyed by id."""
    import json

    path = Path(__file__).resolve().parent.parent / "data" / "anomalies.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data["anomalies"]}
