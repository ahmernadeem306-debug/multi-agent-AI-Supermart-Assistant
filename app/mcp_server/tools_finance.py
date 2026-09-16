"""Finance- and aisle-metric MCP tool functions."""
from __future__ import annotations

from app.db.base import get_readonly_session
from app.mcp_server.schemas import (
    GetAisleMetricsInput,
    GetMarginReportInput,
    GetShrinkageReportInput,
    AisleMetricsOutput,
    ReportOutput,
    ShrinkageReportOutput,
)
from app.services.analytics_service import AnalyticsService
from app.services.finance_service import FinanceService


def get_shrinkage_report(
    sku: str | None = None, aisle: str | None = None, days: int = 90
) -> ShrinkageReportOutput:
    """Summarise recorded shrinkage cost over the trailing ``days`` window.

    Filter by ``sku`` or ``aisle``. The result breaks the loss down by reason
    (damage, theft, expiry, admin_error) with quantity and valued cost.
    """
    params = GetShrinkageReportInput(sku=sku, aisle=aisle, days=days)
    with get_readonly_session() as session:
        data = FinanceService(session).shrinkage_valuation(
            sku=params.sku, aisle=params.aisle, days=params.days
        )
    return ShrinkageReportOutput(
        days=params.days,
        total_qty=data["total_qty"],
        total_cost=data["total_cost"],
        by_reason=data["by_reason"],
        summary=(
            f"{data['total_qty']} units lost / {data['total_cost']:.2f} cost "
            f"over {params.days} day(s)."
        ),
    )


def get_margin_report(
    sku: str | None = None, aisle: str | None = None, days: int = 30
) -> ReportOutput:
    """Report gross margin over the trailing ``days`` window.

    With ``sku`` -> single-SKU margin; with ``aisle`` -> aisle aggregate;
    with neither -> store-wide revenue, COGS and gross-margin percentage.
    """
    params = GetMarginReportInput(sku=sku, aisle=aisle, days=days)
    with get_readonly_session() as session:
        svc = FinanceService(session)
        if params.sku:
            report = svc.sku_margin(params.sku, days=params.days)
        elif params.aisle:
            report = svc.aisle_margin(params.aisle, days=params.days)
        else:
            report = svc.revenue_and_cogs(days=params.days)
    return ReportOutput(
        report=report,
        summary=(
            f"Gross margin {report['gross_margin']:.2f} "
            f"({report['gross_margin_pct']:.1f}%) over {params.days} day(s)."
        ),
    )


def get_aisle_metrics(aisle: str, days: int = 30) -> AisleMetricsOutput:
    """Return dashboard metrics for one aisle over the trailing ``days`` window.

    Includes SKU count, revenue, units sold, out-of-stock and low-stock
    counts, shrinkage value, average margin percentage and expiring-batch
    count.
    """
    params = GetAisleMetricsInput(aisle=aisle, days=days)
    with get_readonly_session() as session:
        metrics = AnalyticsService(session).aisle_metrics(params.aisle, days=params.days)
    return AisleMetricsOutput(
        metrics=metrics,
        summary=(
            f"{params.aisle}: {metrics['revenue']:.0f} revenue, "
            f"{metrics['out_of_stock_count']} out of stock, "
            f"{metrics['avg_margin_pct']:.1f}% margin."
        ),
    )
