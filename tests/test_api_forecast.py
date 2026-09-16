"""TestClient coverage of the /forecast routes, including validation errors."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

pytestmark = pytest.mark.usefixtures("seeded_db")

client = TestClient(app)


def test_list_skus():
    body = client.get("/forecast/skus").json()
    assert len(body["skus"]) == 60
    assert {"sku", "name", "aisle", "is_perishable"} <= set(body["skus"][0])


def test_sku_forecast_has_series_and_risk():
    r = client.get("/forecast/SKU-1035", params={"horizon_days": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 10
    assert body["backend"] in {"xgboost", "seasonal_naive"}
    assert {"date", "predicted_units", "lower", "upper"} <= set(body["forecast"][0])
    assert body["stockout_risk"]["risk_level"] in {"critical", "high", "medium", "low"}
    assert "recommended_reorder_point" in body["reorder_recommendation"]
    assert isinstance(body["expiry_risk"], list)


def test_forecast_horizon_bounds_are_validated():
    assert client.get("/forecast/SKU-1035", params={"horizon_days": 0}).status_code == 422
    assert client.get("/forecast/SKU-1035", params={"horizon_days": 99}).status_code == 422


def test_forecast_unknown_sku_is_404():
    r = client.get("/forecast/SKU-DOES-NOT-EXIST")
    assert r.status_code == 404
    assert r.json()["error_code"] == "data_not_found"


def test_alerts_and_reorder_recommendations():
    alerts = client.get("/forecast/alerts", params={"limit": 10})
    assert alerts.status_code == 200
    assert alerts.json()["count"] >= 0
    assert len(alerts.json()["alerts"]) <= 10

    recs = client.get("/forecast/reorder-recommendations", params={"limit": 10})
    assert recs.status_code == 200
    assert "recommendations" in recs.json()

    assert client.get("/forecast/alerts", params={"limit": 9999}).status_code == 422
