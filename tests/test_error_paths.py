"""Fault-injection matrix: every failure mode must degrade, never crash."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.rca import get_rca_workflow
from app.config import get_settings
from app.core.exceptions import ForecastError, RateLimitError, RetrievalError, ToolExecutionError
from app.db.repositories.sales_repo import SalesRepository
from app.mcp_client.adapter import DirectToolProvider
from app.services.forecasting_service import ForecastingService
from app.agents.rca_schemas import RcaRequest
from app.agents.rca_workflow import RcaWorkflow
from tests.fakes import RaisingLLM, StubRetriever

pytestmark = pytest.mark.usefixtures("seeded_db")

client = TestClient(app)


# 1. Groq down -----------------------------------------------------------
def test_groq_down_rca_returns_partial_report():
    wf = RcaWorkflow(RaisingLLM(), DirectToolProvider(), StubRetriever())
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    assert report.status == "partial"
    assert report.llm_calls == 0
    assert report.ranked_causes[0].category == "supply_delay"


# 2. Groq 429 ----------------------------------------------------------
def test_groq_rate_limited_rca_still_completes():
    wf = RcaWorkflow(RaisingLLM(RateLimitError("429 quota")), DirectToolProvider(), StubRetriever())
    report = wf.run(RcaRequest(sku="SKU-1002", anomaly_type="auto"))
    assert report.status == "partial"
    assert report.ranked_causes[0].category == "shrinkage_theft"


# 3. MCP server killed -------------------------------------------------
def test_mcp_tools_unavailable_rca_records_data_gaps():
    class DeadTools(DirectToolProvider):
        def call_tool(self, name, arguments):
            raise ToolExecutionError("MCP stdio handshake failed")

    wf = RcaWorkflow(RaisingLLM(), DeadTools(), StubRetriever())
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    assert report.status in {"partial", "no_anomaly"}
    assert report.data_gaps  # every evidence step logged its failure
    # never raised


# 4. Chroma directory deleted ---------------------------------------
def test_chroma_missing_search_returns_no_result_not_500(tmp_path):
    from app.rag import retriever as retriever_mod

    settings = get_settings()
    original = settings.chroma_persist_dir
    original_singleton = retriever_mod._retriever_singleton
    try:
        # tmp_path is empty and pytest-managed, so nothing is left behind in
        # the working tree (simulates the Chroma persist dir being deleted).
        settings.chroma_persist_dir = str(tmp_path / "missing_chroma")
        retriever_mod._retriever_singleton = None  # force a rebuild against the new (empty) dir
        r = client.post("/documents/search", json={"query": "return policy for frozen goods"})
        assert r.status_code == 200
        assert r.json()["found"] is False
    finally:
        settings.chroma_persist_dir = original
        retriever_mod._retriever_singleton = original_singleton
        retriever_mod._retriever_singleton = original_singleton  # restore the shared instance


# 5. Model file missing -------------------------------------------------
def test_missing_model_falls_back_to_seasonal_naive(read_session, tmp_path):
    settings = get_settings().model_copy(
        update={"forecast_backend": "xgboost", "model_dir": str(tmp_path / "empty")}
    )
    svc = ForecastingService(read_session, settings)
    fc = svc.forecast("SKU-1035", 7)
    assert svc.backend_name == "seasonal_naive"
    assert fc["backend"] == "seasonal_naive"
    assert len(fc["forecast"]) == 7


# 6. Database locked / data-layer failure ---------------------------
def test_database_error_surfaces_as_forecast_error(read_session, monkeypatch):
    from sqlalchemy.exc import OperationalError

    def boom(*a, **k):
        raise OperationalError("SELECT 1", {}, Exception("database is locked"))

    monkeypatch.setattr(SalesRepository, "list_transactions", boom)
    with pytest.raises(ForecastError):
        ForecastingService(read_session, get_settings().model_copy(update={"forecast_backend": "seasonal_naive"})).forecast("SKU-1035")


# 7. Empty result set -------------------------------------------------
def test_zero_sales_sku_forecast_is_safe(db_session):
    from tests.test_forecast_service import _seed_sku, _settings_naive

    _seed_sku(db_session, "SKU-EP1", [0] * 15, on_hand=5, reorder_point=20)
    fc = ForecastingService(db_session, _settings_naive()).forecast("SKU-EP1")
    assert fc["avg_daily_forecast"] == 0.0
    assert fc["warnings"]


# 8. Malformed user input -------------------------------------------
def test_malformed_api_input_is_rejected_with_422():
    assert client.get("/forecast/SKU-1035", params={"horizon_days": -5}).status_code == 422
    assert client.post("/rca", json={}).status_code == 422
    assert client.post(
        "/rca", json={"sku": "X", "start_date": "2026-05-01", "end_date": "2026-01-01"}
    ).status_code == 422


# 9. Oversized / unsafe upload -------------------------------------
def test_oversized_upload_is_rejected():
    big = b"x" * (get_settings().max_upload_bytes + 1)
    r = client.post("/documents/upload", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413


def test_disallowed_upload_extension_is_rejected():
    r = client.post("/documents/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_filename_is_sanitised_no_path_traversal(tmp_path):
    from pathlib import Path

    knowledge_dir = Path(get_settings().knowledge_dir).resolve()
    r = client.post(
        "/documents/upload",
        files={"file": ("../../etc/passwd.md", b"# Traversal test\n\nHarmless content.", "text/markdown")},
    )
    assert r.status_code == 200
    landed = Path(r.json()["filename"])
    assert landed.parent == Path(".")  # bare filename, no directories
    assert (knowledge_dir / landed.name).exists()
    (knowledge_dir / landed.name).unlink(missing_ok=True)
