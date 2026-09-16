"""TestClient coverage of every /metrics route, including validation failures."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

pytestmark = pytest.mark.usefixtures("seeded_db")

client = TestClient(app)


def test_aisles_metrics():
    response = client.get("/metrics/aisles", params={"days": 30})
    assert response.status_code == 200
    body = response.json()
    assert len(body["aisles"]) == 8
    assert {"revenue", "out_of_stock_count", "avg_margin_pct"} <= set(body["aisles"][0])


def test_single_aisle_metrics_and_unknown():
    ok = client.get("/metrics/aisle/Produce")
    assert ok.status_code == 200
    assert ok.json()["aisle"] == "Produce"

    missing = client.get("/metrics/aisle/NoSuchAisle")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "data_not_found"


def test_low_stock_route_and_bad_limit():
    ok = client.get("/metrics/low-stock", params={"limit": 50})
    assert ok.status_code == 200
    assert ok.json()["count"] == len(ok.json()["items"])

    bad = client.get("/metrics/low-stock", params={"limit": 100000})
    assert bad.status_code == 422


def test_expiring_route_and_bad_days():
    ok = client.get("/metrics/expiring", params={"days_ahead": 7})
    assert ok.status_code == 200
    assert "items" in ok.json()

    bad = client.get("/metrics/expiring", params={"days_ahead": -1})
    assert bad.status_code == 422


def test_suppliers_route():
    response = client.get("/metrics/suppliers")
    assert response.status_code == 200
    suppliers = response.json()["suppliers"]
    assert len(suppliers) == 6
    assert "on_time_delivery_rate" in suppliers[0]


def test_anomalies_route_all_and_single_kind():
    all_kinds = client.get("/metrics/anomalies", params={"days": 90})
    assert all_kinds.status_code == 200
    assert set(all_kinds.json()["results"]) == {
        "shrinkage",
        "late_delivery",
        "demand_shift",
        "stockout",
    }

    single = client.get("/metrics/anomalies", params={"kind": "stockout", "days": 90})
    assert single.status_code == 200
    assert list(single.json()["results"]) == ["stockout"]

    bad = client.get("/metrics/anomalies", params={"kind": "bogus"})
    assert bad.status_code == 422


def test_readonly_session_blocks_writes():
    """The read-only session guard must reject any ORM write."""
    from app.core.exceptions import ToolExecutionError
    from app.db.base import get_readonly_session
    from app.db.models import ShrinkageEvent

    with pytest.raises(ToolExecutionError):
        with get_readonly_session() as session:
            session.add(
                ShrinkageEvent(
                    sku="SKU-1001",
                    event_date=__import__("datetime").date.today(),
                    qty=1,
                    reason="theft",
                    notes="should not persist",
                )
            )
            session.flush()
