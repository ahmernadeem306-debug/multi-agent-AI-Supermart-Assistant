"""POST /query — served by the LangGraph multi-agent orchestrator.

The graph (supervisor -> specialist fan-out -> synthesis) is compiled once on
first use. Every call writes exactly one agent_runs row, including failures.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.agents.graph import build_graph, run_graph
from app.agents.state import TOOL_CACHE
from app.api.schemas import QueryRequest, QueryResponse
from app.config import Settings, get_settings
from app.core.decision_log import finish_run, start_run
from app.core.exceptions import ToolExecutionError
from app.db.base import get_session
from app.llm.groq_client import GroqProvider
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.mcp_client.adapter import DirectToolProvider, get_tool_provider
from app.rag.retriever import build_retriever

router = APIRouter()
logger = get_logger(__name__)

_provider: LLMProvider | None = None
_graph = None


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Return the process-wide LLMProvider, constructing it on first use."""
    global _provider
    if _provider is None:
        _provider = GroqProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
            timeout_seconds=settings.groq_timeout_seconds,
        )
    return _provider


def _build_tool_provider(settings: Settings):
    try:
        return get_tool_provider(settings)
    except ToolExecutionError as exc:
        logger.error("tool_provider_unavailable_falling_back_to_direct", error=str(exc))
        return DirectToolProvider()


def _build_retriever_safe():
    try:
        return build_retriever()
    except Exception as exc:  # noqa: BLE001 - policy agent degrades if RAG is down
        logger.error("retriever_unavailable", error=str(exc))
        return _NullRetriever()


class _NullRetriever:
    def retrieve(self, query: str, *, doc_type: str | None = None):
        raise ToolExecutionError(
            "The knowledge base is unavailable. Run `python scripts/ingest_docs.py`."
        )


def get_graph(settings: Settings = Depends(get_settings)):
    """Compile the orchestration graph once and cache it."""
    global _graph
    if _graph is None:
        _graph = build_graph(
            get_llm_provider(settings),
            _build_tool_provider(settings),
            _build_retriever_safe(),
            settings,
        )
    return _graph


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> QueryResponse:
    start = time.monotonic()
    with get_session() as session:
        run_id = start_run(session, request.question)

    result = run_graph(graph, run_id, request.question, settings)
    latency_ms = int((time.monotonic() - start) * 1000)

    plan = result.get("plan", {}) or {}
    agents_invoked = list((result.get("agent_outputs") or {}).keys())
    answer = result.get("final_answer") or "No answer was produced."
    citations = result.get("citations", []) or []
    tool_calls = result.get("tool_calls", []) or []
    status = result.get("status", "success")
    errors = result.get("errors", []) or []

    with get_session() as session:
        finish_run(
            session,
            run_id,
            route=result.get("route"),
            agents_invoked=agents_invoked,
            tool_calls=tool_calls,
            retrieved_docs=citations,
            final_answer=answer,
            confidence=result.get("confidence"),
            latency_ms=latency_ms,
            status=status,
            error="; ".join(errors) if errors else None,
        )

    logger.info(
        "query_served",
        run_id=run_id,
        route=result.get("route"),
        agents=agents_invoked,
        latency_ms=latency_ms,
        tool_cache_hit_rate=TOOL_CACHE.hit_rate,
    )
    return QueryResponse(
        run_id=run_id,
        answer=answer,
        route=result.get("route"),
        tool_calls=tool_calls,
        citations=citations,
        latency_ms=latency_ms,
        agents_invoked=agents_invoked,
        plan_reasoning=plan.get("reasoning"),
        confidence=result.get("confidence"),
        status=status,
    )
