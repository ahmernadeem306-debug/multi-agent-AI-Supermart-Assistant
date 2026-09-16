"""GET /health — app version, DB connectivity and Groq configuration status."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.db.base import get_engine

router = APIRouter()

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_connected = False
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    groq_configured = False
    tool_provider = "mcp"
    forecast_backend = "xgboost"
    model_trained = False
    try:
        settings = get_settings()
        groq_configured = bool(settings.groq_api_key.get_secret_value())
        tool_provider = settings.tool_provider
        forecast_backend = settings.forecast_backend
        model_trained = (Path(settings.model_dir) / "forecast_xgb.joblib").exists()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_connected else "degraded",
        version=APP_VERSION,
        db_connected=db_connected,
        groq_configured=groq_configured,
        tool_provider=tool_provider,
        forecast_backend=forecast_backend,
        forecast_model_trained=model_trained,
    )
