"""Agent decision log: persists one agent_runs row per /query invocation.

Kept as core infrastructure (like logging_config) rather than a repository,
since it is written from the API layer around whatever orchestration runs
today (a direct Groq call) or later (the LangGraph supervisor).
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AgentRun


def _run_to_dict(run: AgentRun) -> dict:
    return {
        "run_id": run.run_id,
        "ts": run.ts.isoformat() if run.ts else None,
        "user_query": run.user_query,
        "route": run.route,
        "agents_invoked": run.agents_invoked or [],
        "tool_calls": run.tool_calls or [],
        "retrieved_docs": run.retrieved_docs or [],
        "final_answer": run.final_answer,
        "confidence": run.confidence,
        "latency_ms": run.latency_ms,
        "status": run.status,
        "error": run.error,
    }


def start_run(session: Session, question: str) -> str:
    """Create a pending agent_runs row and return its run_id."""
    run_id = str(uuid.uuid4())
    run = AgentRun(
        run_id=run_id,
        ts=dt.datetime.utcnow(),
        user_query=question,
        route=None,
        agents_invoked=[],
        tool_calls=[],
        retrieved_docs=[],
        final_answer=None,
        confidence=None,
        latency_ms=None,
        status="pending",
        error=None,
    )
    session.add(run)
    session.flush()
    return run_id


def finish_run(
    session: Session,
    run_id: str,
    *,
    route: str | None = None,
    agents_invoked: list | None = None,
    tool_calls: list | None = None,
    retrieved_docs: list | None = None,
    final_answer: str | None = None,
    confidence: float | None = None,
    latency_ms: int | None = None,
    status: str = "success",
    error: str | None = None,
) -> None:
    """Update the agent_runs row for run_id with the outcome of the run."""
    run = session.get(AgentRun, run_id)
    if run is None:
        return
    run.route = route
    run.agents_invoked = agents_invoked or []
    run.tool_calls = tool_calls or []
    run.retrieved_docs = retrieved_docs or []
    run.final_answer = final_answer
    run.confidence = confidence
    run.latency_ms = latency_ms
    run.status = status
    run.error = error
    session.flush()


def list_runs(session: Session, *, limit: int = 20, offset: int = 0) -> dict:
    """Return a page of decision-log rows, newest first, with a total count."""
    total = int(session.scalar(select(func.count()).select_from(AgentRun)) or 0)
    stmt = select(AgentRun).order_by(AgentRun.ts.desc()).limit(limit).offset(offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [_run_to_dict(r) for r in session.scalars(stmt)],
    }


def get_run(session: Session, run_id: str) -> dict | None:
    """Return one decision-log row by id, or None."""
    run = session.get(AgentRun, run_id)
    return _run_to_dict(run) if run else None
