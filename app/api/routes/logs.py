"""GET /logs and GET /logs/{run_id} — the agent decision log."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.decision_log import get_run, list_runs
from app.core.exceptions import DataNotFoundError
from app.db.base import get_readonly_session

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("")
def read_logs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    with get_readonly_session() as session:
        return list_runs(session, limit=limit, offset=offset)


@router.get("/{run_id}")
def read_log(run_id: str) -> dict:
    with get_readonly_session() as session:
        row = get_run(session, run_id)
    if row is None:
        raise DataNotFoundError(f"No decision-log run with id '{run_id}'.")
    return row
