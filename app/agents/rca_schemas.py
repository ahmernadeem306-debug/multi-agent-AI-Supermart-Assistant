"""Pydantic schemas for the root-cause analysis workflow (Workflow W2)."""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AnomalyType = Literal["stockout", "shrinkage", "supply_delay", "margin_drop", "auto"]
CauseCategory = Literal[
    "supply_delay",
    "shrinkage_theft",
    "shrinkage_damage",
    "shrinkage_admin_error",
    "expiry_writeoff",
    "demand_shift",
    "reorder_point",
    "forecast_miss",
    "data_quality",
    "other",
]
Urgency = Literal["low", "medium", "high", "critical"]
ReportStatus = Literal["complete", "partial", "no_anomaly"]


class RcaRequest(BaseModel):
    sku: str | None = None
    aisle: str | None = None
    anomaly_type: AnomalyType = "auto"
    start_date: dt.date | None = None
    end_date: dt.date | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RcaRequest":
        if not self.sku and not self.aisle:
            raise ValueError("Provide either 'sku' or 'aisle'.")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date.")
        return self


class EvidenceItem(BaseModel):
    id: str
    source: str  # "tool:<name>" | "detector:<kind>" | "document"
    title: str
    detail: str
    data: dict = Field(default_factory=dict)


class RankedCause(BaseModel):
    cause: str
    category: CauseCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    contributing_factor: bool = False


class TimelineEvent(BaseModel):
    date: str
    event: str


class PolicyFinding(BaseModel):
    clause: str
    citation: dict
    relevance: str


class RecommendedAction(BaseModel):
    action: str
    owner_role: str
    urgency: Urgency
    expected_impact: str


class RootCauseReport(BaseModel):
    sku: str | None = None
    aisle: str | None = None
    anomaly_type: AnomalyType
    status: ReportStatus
    summary: str
    ranked_causes: list[RankedCause] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    policy_findings: list[PolicyFinding] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
    generated_at: str
    llm_calls: int = 0


class LlmRanking(BaseModel):
    """What the LLM is asked to return; merged and validated by the workflow."""

    summary: str = ""
    ranked_causes: list[RankedCause] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
