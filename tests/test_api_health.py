from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_returns_200_with_expected_shape():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "version" in body
    assert isinstance(body["db_connected"], bool)
    assert isinstance(body["groq_configured"], bool)


def test_health_reports_db_connected():
    response = client.get("/health")

    body = response.json()
    assert body["db_connected"] is True


def test_health_reports_groq_configured():
    response = client.get("/health")

    body = response.json()
    assert body["groq_configured"] is True
