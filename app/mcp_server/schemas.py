"""Pydantic input/output models for every MCP tool.

Each tool has an explicit input model (argument validation, result caps) and
an output model that always carries a human-readable ``summary`` string
alongside its structured payload. Models are shared by the FastMCP server and
by the in-process :class:`DirectToolProvider`.
"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_LIMIT = 100
MAX_LIMIT = 500

Granularity = Literal["daily", "weekly"]
AnomalyKind = Literal["shrinkage", "late_delivery", "demand_shift", "stockout"]


# --------------------------------------------------------------------- inputs
class GetProductInfoInput(BaseModel):
    sku: str | None = None
    name_query: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "GetProductInfoInput":
        if not self.sku and not self.name_query:
            raise ValueError("Provide either 'sku' or 'name_query'.")
        return self


class GetStockLevelInput(BaseModel):
    sku: str
    include_batches: bool = False


class ListLowStockInput(BaseModel):
    aisle: str | None = None
    limit: int = Field(default=50, ge=1, le=MAX_LIMIT)


class GetSalesHistoryInput(BaseModel):
    sku: str
    start_date: dt.date
    end_date: dt.date
    granularity: Granularity = "daily"

    @model_validator(mode="after")
    def _ordered(self) -> "GetSalesHistoryInput":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date.")
        return self


class GetTopSellersInput(BaseModel):
    aisle: str | None = None
    days: int = Field(default=30, ge=1, le=730)
    limit: int = Field(default=10, ge=1, le=MAX_LIMIT)


class GetShrinkageReportInput(BaseModel):
    sku: str | None = None
    aisle: str | None = None
    days: int = Field(default=90, ge=1, le=730)


class GetExpiringBatchesInput(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=365)
    aisle: str | None = None


class GetSupplierStatusInput(BaseModel):
    supplier_id: int | None = None
    supplier_name: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "GetSupplierStatusInput":
        if self.supplier_id is None and not self.supplier_name:
            raise ValueError("Provide either 'supplier_id' or 'supplier_name'.")
        return self


class GetOpenPurchaseOrdersInput(BaseModel):
    sku: str | None = None
    supplier: str | None = None
    late_only: bool = False


class GetMarginReportInput(BaseModel):
    sku: str | None = None
    aisle: str | None = None
    days: int = Field(default=30, ge=1, le=730)


class GetAisleMetricsInput(BaseModel):
    aisle: str
    days: int = Field(default=30, ge=1, le=730)


class DetectAnomaliesInput(BaseModel):
    kind: AnomalyKind
    days: int = Field(default=90, ge=1, le=730)


class GetDemandForecastInput(BaseModel):
    sku: str
    horizon_days: int = Field(default=14, ge=1, le=30)
    include_risk: bool = True


# -------------------------------------------------------------------- outputs
class ToolOutput(BaseModel):
    """Base output: a summary line plus (subclass-specific) structured fields."""

    summary: str


class ProductMatchesOutput(ToolOutput):
    matches: list[dict] = Field(default_factory=list)


class StockLevelOutput(ToolOutput):
    stock: dict | None = None
    batches: list[dict] = Field(default_factory=list)


class ItemListOutput(ToolOutput):
    items: list[dict] = Field(default_factory=list)
    count: int = 0


class SalesHistoryOutput(ToolOutput):
    sku: str
    granularity: Granularity
    points: list[dict] = Field(default_factory=list)
    total_units: int = 0
    total_revenue: float = 0.0


class ShrinkageReportOutput(ToolOutput):
    days: int
    total_qty: int = 0
    total_cost: float = 0.0
    by_reason: dict = Field(default_factory=dict)


class SupplierStatusOutput(ToolOutput):
    supplier: dict


class ReportOutput(ToolOutput):
    report: dict = Field(default_factory=dict)


class AisleMetricsOutput(ToolOutput):
    metrics: dict = Field(default_factory=dict)


class AnomaliesOutput(ToolOutput):
    kind: AnomalyKind
    items: list[dict] = Field(default_factory=list)
    count: int = 0


class ForecastOutput(ToolOutput):
    sku: str
    backend: str
    horizon_days: int
    forecast: list[dict] = Field(default_factory=list)
    avg_daily_forecast: float = 0.0
    stockout_risk: dict | None = None
    reorder_recommendation: dict | None = None
    expiry_risk: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
