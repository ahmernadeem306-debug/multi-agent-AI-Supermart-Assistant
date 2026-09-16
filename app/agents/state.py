"""Shared agent state, result schemas, tool cache and the specialist base class.

``AgentState`` is the LangGraph state object. It is intentionally
extensible — Day 4 adds ``forecast_results`` and ``rca_report`` fields without
touching the graph wiring.
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from pydantic import BaseModel, Field

from app.core.exceptions import BizAgentError
from app.llm.parsing import generate_structured
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.mcp_client.adapter import ToolProvider, ToolResult

logger = get_logger(__name__)


# --------------------------------------------------------------------- schemas
class PlannedToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    reasoning: str = ""


class AgentAnswer(BaseModel):
    answer: str
    confidence: float = 0.5
    used_tool_data: bool = False


class ToolCallRecord(BaseModel):
    agent: str
    tool: str
    arguments: dict[str, Any]
    summary: str = ""
    row_count: int = 0
    is_error: bool = False
    error_code: str | None = None
    duration_ms: int = 0


class AgentOutput(BaseModel):
    agent: str
    answer: str = ""
    confidence: float = 0.0
    used_tool_data: bool = False
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    error: str | None = None


def _merge_outputs(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged.update(right)
    return merged


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    route: str
    plan: dict
    needs_policy_check: bool
    agent_outputs: Annotated[dict[str, dict], _merge_outputs]
    tool_calls: Annotated[list[dict], operator.add]
    citations: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    final_answer: str
    confidence: float
    status: str


# ----------------------------------------------------------------- tool cache
class _TTLCache:
    """Tiny process-wide (tool, args) cache with a short TTL and hit counter."""

    def __init__(self, ttl_seconds: float = 60.0, maxsize: int = 256) -> None:
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._data: dict[str, tuple[float, ToolResult]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(name: str, arguments: dict) -> str:
        return f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def get(self, name: str, arguments: dict) -> ToolResult | None:
        key = self._key(name, arguments)
        entry = self._data.get(key)
        if entry and (time.monotonic() - entry[0]) < self._ttl:
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def put(self, name: str, arguments: dict, result: ToolResult) -> None:
        if len(self._data) >= self._maxsize:
            self._data.pop(next(iter(self._data)))
        self._data[self._key(name, arguments)] = (time.monotonic(), result)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0


TOOL_CACHE = _TTLCache()


def _row_count(data: dict) -> int:
    for key in ("count", "items", "matches", "points"):
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return len(value)
    return 0


# --------------------------------------------------------------- base class
class SpecialistAgent:
    """Plan -> call permitted tools (<= max) -> synthesise a grounded answer.

    Concrete agents set ``name``, ``system``, ``allowed_tools`` and the two
    prompt builders. All failures are caught and reported as an
    :class:`AgentOutput` with ``error`` set — one agent never breaks the graph.
    """

    name: str = "specialist"
    system: str = ""
    allowed_tools: list[str] = []
    plan_prompt: Callable[[str], str]
    answer_prompt: Callable[[str, str], str]

    def __init__(self, llm: LLMProvider, tools: ToolProvider, *, max_tool_calls: int = 3) -> None:
        self.llm = llm
        self.tools = tools
        self.max_tool_calls = max_tool_calls

    # -- steps -------------------------------------------------------------
    def _plan(self, question: str) -> list[PlannedToolCall]:
        plan = generate_structured(
            self.llm, self.plan_prompt(question), self.system, ToolPlan, temperature=0.0
        )
        allowed = set(self.allowed_tools)
        picked = [c for c in plan.tool_calls if c.tool in allowed][: self.max_tool_calls]
        return picked

    def _execute(self, planned: list[PlannedToolCall]) -> tuple[list[ToolCallRecord], str]:
        records: list[ToolCallRecord] = []
        lines: list[str] = []
        for call in planned:
            started = time.monotonic()
            cached = TOOL_CACHE.get(call.tool, call.arguments)
            try:
                result = cached or self.tools.call_tool(call.tool, call.arguments)
            except BizAgentError as exc:
                result = ToolResult(
                    name=call.tool, data={"error": exc.message}, summary=exc.message,
                    is_error=True, error_code=exc.error_code,
                )
            if cached is None and not result.is_error:
                TOOL_CACHE.put(call.tool, call.arguments, result)
            duration_ms = int((time.monotonic() - started) * 1000)
            records.append(
                ToolCallRecord(
                    agent=self.name, tool=call.tool, arguments=call.arguments,
                    summary=result.summary, row_count=_row_count(result.data),
                    is_error=result.is_error, error_code=result.error_code,
                    duration_ms=duration_ms,
                )
            )
            lines.append(
                f"- {call.tool}({json.dumps(call.arguments, default=str)}) -> "
                + (f"ERROR: {result.summary}" if result.is_error else json.dumps(result.data, default=str)[:1500])
            )
        return records, ("\n".join(lines) if lines else "(no tools were called)")

    def _answer(self, question: str, transcript: str) -> AgentAnswer:
        return generate_structured(
            self.llm, self.answer_prompt(question, transcript), self.system,
            AgentAnswer, temperature=0.2,
        )

    # -- entrypoint ------------------------------------------------------
    def run(self, state: AgentState) -> dict:
        question = state["question"]
        try:
            planned = self._plan(question)
            records, transcript = self._execute(planned)
            answer = self._answer(question, transcript)
            output = AgentOutput(
                agent=self.name,
                answer=answer.answer,
                confidence=answer.confidence,
                used_tool_data=answer.used_tool_data,
                tool_calls=records,
            )
        except Exception as exc:  # noqa: BLE001 - isolate agent failure
            logger.error("agent_failed", agent=self.name, error=str(exc))
            output = AgentOutput(
                agent=self.name,
                answer=f"The {self.name} agent could not complete: {exc}",
                error=str(exc),
            )
        return {
            "agent_outputs": {self.name: output.model_dump(mode="json")},
            "tool_calls": [r.model_dump(mode="json") for r in output.tool_calls],
            "errors": [f"{self.name}: {output.error}"] if output.error else [],
        }
