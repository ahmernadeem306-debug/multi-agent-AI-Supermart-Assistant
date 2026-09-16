"""POST /rca — run the root-cause analysis workflow and log the run."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.agents.rca_schemas import RcaRequest, RootCauseReport
from app.agents.rca_workflow import RcaWorkflow
from app.api.routes.query import (
    _build_retriever_safe,
    _build_tool_provider,
    get_llm_provider,
)
from app.config import Settings, get_settings
from app.core.decision_log import finish_run, start_run
from app.db.base import get_session
from app.logging_config import get_logger

router = APIRouter(tags=["rca"])
logger = get_logger(__name__)

_workflow: RcaWorkflow | None = None

_STATUS_MAP = {"complete": "success", "no_anomaly": "success", "partial": "partial"}


def get_rca_workflow(settings: Settings = Depends(get_settings)) -> RcaWorkflow:
    global _workflow
    if _workflow is None:
        _workflow = RcaWorkflow(
            get_llm_provider(settings),
            _build_tool_provider(settings),
            _build_retriever_safe(),
        )
    return _workflow


@router.post("/rca", response_model=RootCauseReport)
def run_rca(
    request: RcaRequest,
    workflow: RcaWorkflow = Depends(get_rca_workflow),
) -> RootCauseReport:
    target = request.sku or request.aisle
    query_text = f"RCA[{request.anomaly_type}] {target}"
    start = time.monotonic()
    with get_session() as session:
        run_id = start_run(session, query_text)

    report = workflow.run(request)
    latency_ms = int((time.monotonic() - start) * 1000)
    top_confidence = report.ranked_causes[0].confidence if report.ranked_causes else None

    with get_session() as session:
        finish_run(
            session,
            run_id,
            route="rca",
            agents_invoked=["rca_workflow"],
            tool_calls=report.tool_trace,
            retrieved_docs=[pf.citation for pf in report.policy_findings],
            final_answer=report.summary,
            confidence=top_confidence,
            latency_ms=latency_ms,
            status=_STATUS_MAP.get(report.status, "partial"),
            error="; ".join(report.data_gaps) if report.data_gaps else None,
        )

    logger.info(
        "rca_served",
        run_id=run_id,
        target=target,
        status=report.status,
        causes=len(report.ranked_causes),
        llm_calls=report.llm_calls,
        latency_ms=latency_ms,
    )
    return report
