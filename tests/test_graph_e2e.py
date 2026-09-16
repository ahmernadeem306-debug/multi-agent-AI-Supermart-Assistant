"""End-to-end graph runs with mocked LLM + stub tools, through the /query API.

Also holds the static guard that agents never import services, the DB or the
Groq SDK.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.graph import build_graph
from app.api.main import app
from app.api.routes.query import get_graph
from tests.fakes import ScriptedLLM, StubRetriever, StubToolProvider

_AGENTS_DIR = Path(__file__).resolve().parent.parent / "app" / "agents"
_FORBIDDEN_PREFIXES = ("app.services", "app.db", "groq")

_TOOL_BY_SYSTEM = {
    "Inventory & Stock specialist": ("list_low_stock", {"limit": 5}),
    "Sales specialist": ("get_top_sellers", {"days": 30, "limit": 5}),
    "Finance specialist": ("get_margin_report", {"aisle": "Dairy", "days": 30}),
    "Demand Forecasting specialist": ("get_stock_level", {"sku": "SKU-1001"}),
}

_ROUTES = {
    "low on stock": {"route": "inventory", "agents": ["inventory"], "reasoning": "r", "needs_policy_check": False},
    "top sellers": {"route": "sales", "agents": ["sales"], "reasoning": "r", "needs_policy_check": False},
    "return frozen": {"route": "policy", "agents": ["policy"], "reasoning": "r", "needs_policy_check": True},
    "margin drop": {"route": "rca", "agents": ["finance", "inventory"], "reasoning": "r", "needs_policy_check": False},
}


def _responder(prompt: str, system: str | None) -> str:
    low = prompt.lower()
    if "return json with: route" in low:
        for key, plan in _ROUTES.items():
            if key in low:
                return json.dumps(plan)
        return json.dumps(_ROUTES["low on stock"])
    if '"tool_calls":' in prompt and "Return an empty list" in prompt:
        for marker, (tool, args) in _TOOL_BY_SYSTEM.items():
            if system and marker in system:
                return json.dumps({"tool_calls": [{"tool": tool, "arguments": args}], "reasoning": "r"})
        return json.dumps({"tool_calls": [], "reasoning": "none"})
    if '"answer":' in prompt:
        return json.dumps({"answer": "Grounded specialist answer with data.", "confidence": 0.8, "used_tool_data": True})
    if '"final_answer":' in prompt:
        return json.dumps({"final_answer": "Coherent synthesised answer.", "confidence": 0.83})
    return "Hello!"


@pytest.fixture()
def mock_graph_client():
    tools = StubToolProvider(
        {
            "list_low_stock": {"summary": "2 low-stock SKUs", "items": [{"sku": "SKU-1"}, {"sku": "SKU-2"}], "count": 2},
            "get_top_sellers": {"summary": "top sellers", "items": [{"sku": "SKU-9"}], "count": 1},
            "get_margin_report": {"summary": "margin 41%", "report": {"gross_margin_pct": 41.0}},
            "get_stock_level": {"summary": "SKU-1001: 12 on hand", "stock": {"on_hand_qty": 12}},
        }
    )
    graph = build_graph(ScriptedLLM(_responder), tools, StubRetriever())
    app.dependency_overrides[get_graph] = lambda: graph
    yield TestClient(app), tools
    app.dependency_overrides.pop(get_graph, None)


@pytest.mark.parametrize(
    "question,expected_route,expect_tools",
    [
        ("What is low on stock right now?", "inventory", True),
        ("Who are the top sellers this month?", "sales", True),
        ("How long to return frozen goods?", "policy", False),
        ("Why did the dairy margin drop while stock held?", "rca", True),
    ],
)
def test_graph_query_writes_complete_decision_log(mock_graph_client, question, expected_route, expect_tools):
    client, tools = mock_graph_client
    response = client.post("/query", json={"question": question})
    assert response.status_code == 200
    body = response.json()

    assert body["route"] == expected_route
    assert body["answer"]
    assert body["status"] in {"success", "partial"}
    if expect_tools:
        assert body["tool_calls"], "expected at least one tool call in the trace"

    detail = client.get(f"/logs/{body['run_id']}").json()
    assert detail["route"] == expected_route
    assert detail["agents_invoked"] == body["agents_invoked"]
    assert detail["final_answer"] == body["answer"]
    if expect_tools:
        assert detail["tool_calls"][0]["tool"] in {tc[0] for tc in tools.calls}


def test_multi_agent_query_invokes_multiple_specialists(mock_graph_client):
    client, _ = mock_graph_client
    body = client.post("/query", json={"question": "Why did the dairy margin drop while stock held?"}).json()
    assert set(body["agents_invoked"]) >= {"finance", "inventory"}
    assert body["answer"] == "Coherent synthesised answer."


def test_policy_query_returns_citations(mock_graph_client):
    client, _ = mock_graph_client
    body = client.post("/query", json={"question": "How long to return frozen goods?"}).json()
    assert body["route"] == "policy"
    assert body["citations"], "policy answer must carry citations"
    assert body["citations"][0]["doc_title"] == "Returns and Refunds Policy"


def test_agents_package_has_no_forbidden_imports():
    offenders: list[str] = []
    for path in _AGENTS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == p or name.startswith(p + ".") for p in _FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders
