"""Test doubles for the Day 3 multi-agent / RAG stack.

None of these touch the network. ``ScriptedLLM`` returns canned text keyed on
prompt content; ``StubToolProvider`` returns canned tool results and records
calls; ``StubRetriever`` returns a canned retrieval result.
"""
from __future__ import annotations

import json
from typing import Callable

from app.mcp_client.adapter import ToolResult
from app.rag.retriever import Citation, RetrievalResult


class ScriptedLLM:
    """LLMProvider stub. ``responder(prompt, system) -> str`` supplies output."""

    def __init__(self, responder: Callable[[str, str | None], str]) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str | None, float | None]] = []

    def generate(self, prompt: str, system: str | None = None, *, temperature: float | None = None) -> str:
        self.calls.append((prompt, system, temperature))
        return self._responder(prompt, system)

    def generate_structured(self, prompt: str, system: str | None, schema):  # pragma: no cover
        from app.llm.parsing import generate_structured

        return generate_structured(self, prompt, system, schema)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class RaisingLLM:
    """Every call raises — used to exercise fallback paths."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("LLM unavailable")

    def generate(self, *a, **k) -> str:
        raise self._exc

    def generate_structured(self, *a, **k):
        raise self._exc

    def embed(self, texts):
        raise self._exc


class StubToolProvider:
    """ToolProvider stub returning canned results and recording every call."""

    def __init__(self, results: dict[str, dict] | None = None) -> None:
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[str]:
        return sorted(self._results) or ["get_stock_level", "list_low_stock"]

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        payload = self._results.get(name)
        if payload is None:
            return ToolResult(name=name, data={"summary": "", "items": [], "count": 0}, summary="no data")
        if isinstance(payload, ToolResult):
            return payload
        return ToolResult(name=name, data=payload, summary=payload.get("summary", ""))

    def close(self) -> None:
        return None


class StubRetriever:
    def __init__(self, result: RetrievalResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    def retrieve(self, query: str, *, doc_type: str | None = None) -> RetrievalResult:
        if self._exc is not None:
            raise self._exc
        if self._result is not None:
            return self._result
        return RetrievalResult(
            query=query,
            found=True,
            chunks=["Frozen goods may be returned within 7 days if kept frozen."],
            citations=[
                Citation(
                    doc_title="Returns and Refunds Policy",
                    source_path="data/knowledge/returns_and_refunds_policy.md",
                    chunk_index=2,
                    section="Time limits by category",
                    score=0.61,
                    snippet="Frozen | 7 days | Kept frozen, packaging intact",
                )
            ],
        )


def json_responder(route: dict | None = None, tool_calls: list | None = None) -> Callable[[str, str | None], str]:
    """Build a responder that answers routing / planning / answer / synthesis prompts."""

    def _respond(prompt: str, system: str | None) -> str:
        low = prompt.lower()
        if "return json with: route" in low:
            return json.dumps(route or {"route": "inventory", "agents": ["inventory"],
                                        "reasoning": "stub", "needs_policy_check": False})
        if '"tool_calls"' in low and "return an empty list" in low:
            return json.dumps({"tool_calls": tool_calls or [], "reasoning": "stub"})
        if '"used_tool_data"' in low:
            return json.dumps({"answer": "Grounded stub answer.", "confidence": 0.8, "used_tool_data": True})
        if '"final_answer"' in low:
            return json.dumps({"final_answer": "Synthesised stub answer.", "confidence": 0.8})
        return "stub text reply"

    return _respond
