"""GET /logs and GET /logs/{run_id}."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.decision_log import finish_run, start_run
from app.db.base import get_session

client = TestClient(app)


def _seed_run(question: str, route: str) -> str:
    with get_session() as session:
        run_id = start_run(session, question)
        finish_run(
            session,
            run_id,
            route=route,
            agents_invoked=["inventory"],
            tool_calls=[{"agent": "inventory", "tool": "get_stock_level", "arguments": {"sku": "SKU-1001"},
                         "row_count": 1, "duration_ms": 5}],
            retrieved_docs=[],
            final_answer="ok",
            confidence=0.8,
            latency_ms=42,
            status="success",
        )
    return run_id


def test_logs_list_is_paginated_and_newest_first():
    a = _seed_run("first question", "inventory")
    b = _seed_run("second question", "finance")

    body = client.get("/logs", params={"limit": 5}).json()
    assert body["total"] >= 2
    assert body["limit"] == 5
    ids = [r["run_id"] for r in body["runs"]]
    assert ids.index(b) < ids.index(a)  # newest first


def test_logs_detail_has_full_tool_trace():
    run_id = _seed_run("trace me", "inventory")
    detail = client.get(f"/logs/{run_id}").json()
    assert detail["run_id"] == run_id
    assert detail["route"] == "inventory"
    assert detail["agents_invoked"] == ["inventory"]
    assert detail["tool_calls"][0]["tool"] == "get_stock_level"
    assert detail["status"] == "success"


def test_logs_detail_unknown_run_id_is_404():
    response = client.get("/logs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "data_not_found"


def test_logs_pagination_offset():
    for i in range(3):
        _seed_run(f"paged {i}", "sales")
    page1 = client.get("/logs", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/logs", params={"limit": 2, "offset": 2}).json()
    assert len(page1["runs"]) == 2
    assert {r["run_id"] for r in page1["runs"]}.isdisjoint({r["run_id"] for r in page2["runs"]})
