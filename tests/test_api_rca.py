"""TestClient coverage of POST /rca, with a mocked LLM and in-process tools."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.rca import get_rca_workflow
from app.mcp_client.adapter import DirectToolProvider
from app.agents.rca_workflow import RcaWorkflow
from tests.fakes import RaisingLLM, StubRetriever

pytestmark = pytest.mark.usefixtures("seeded_db")

client = TestClient(app)


@pytest.fixture(autouse=True)
def _mock_workflow():
    wf = RcaWorkflow(RaisingLLM(), DirectToolProvider(), StubRetriever())
    app.dependency_overrides[get_rca_workflow] = lambda: wf
    yield
    app.dependency_overrides.pop(get_rca_workflow, None)


def test_rca_returns_a_report_and_logs_the_run():
    r = client.post("/rca", json={"sku": "SKU-1035", "anomaly_type": "auto"})
    assert r.status_code == 200
    body = r.json()
    assert body["anomaly_type"] == "auto"
    assert body["ranked_causes"][0]["category"] == "supply_delay"
    assert body["ranked_causes"][0]["evidence_ids"]
    assert body["status"] in {"complete", "partial"}
    assert body["policy_findings"]  # StubRetriever returns a citation

    logs = client.get("/logs", params={"limit": 50}).json()
    rca_runs = [row for row in logs["runs"] if row["route"] == "rca"]
    assert rca_runs
    assert any(row["final_answer"] == body["summary"] for row in rca_runs)
    assert all(row["status"] in {"success", "partial"} for row in rca_runs)


def test_rca_requires_sku_or_aisle():
    assert client.post("/rca", json={"anomaly_type": "auto"}).status_code == 422


def test_rca_rejects_reversed_date_range():
    r = client.post(
        "/rca",
        json={"sku": "SKU-1035", "start_date": "2026-02-01", "end_date": "2026-01-01"},
    )
    assert r.status_code == 422


def test_rca_on_sku_with_no_anomaly():
    body = client.post("/rca", json={"sku": "SKU-1010", "anomaly_type": "auto"}).json()
    assert body["status"] == "no_anomaly"
    assert body["ranked_causes"] == []
    assert body["checks_performed"]
