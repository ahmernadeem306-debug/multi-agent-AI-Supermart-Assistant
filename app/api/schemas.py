"""Pydantic request/response models for the API boundary."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    groq_configured: bool
    tool_provider: str = "mcp"
    forecast_backend: str = "xgboost"
    forecast_model_trained: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    run_id: str | None = None


class QueryResponse(BaseModel):
    run_id: str
    answer: str
    route: str | None = None
    tool_calls: list = Field(default_factory=list)
    citations: list = Field(default_factory=list)
    latency_ms: int
    # Day 3 additions (backward compatible)
    agents_invoked: list[str] = Field(default_factory=list)
    plan_reasoning: str | None = None
    confidence: float | None = None
    status: str = "success"


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    run_id: str | None = None
