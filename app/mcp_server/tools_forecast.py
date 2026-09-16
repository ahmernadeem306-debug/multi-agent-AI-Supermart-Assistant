"""Demand-forecast MCP tool (tool 13)."""
from __future__ import annotations

from app.db.base import get_readonly_session
from app.mcp_server.schemas import ForecastOutput, GetDemandForecastInput
from app.services.forecasting_service import ForecastingService


def get_demand_forecast(
    sku: str, horizon_days: int = 14, include_risk: bool = True
) -> ForecastOutput:
    """Forecast daily demand for a SKU and, by default, its derived risks.

    Returns the per-day forecast series for the next ``horizon_days`` (1-30).
    With ``include_risk=true`` (default) it also returns the projected stockout
    date and risk level, a reorder-point recommendation versus the current
    setting, and per-batch perishable expiry risk. Uses the trained XGBoost
    model, falling back to a seasonal-naive baseline if no model is available.
    """
    params = GetDemandForecastInput(sku=sku, horizon_days=horizon_days, include_risk=include_risk)
    with get_readonly_session() as session:
        svc = ForecastingService(session)
        fc = svc.forecast(params.sku, params.horizon_days)
        stockout = reorder = None
        expiry: list[dict] = []
        if params.include_risk:
            stockout = svc.stockout_risk(params.sku)
            reorder = svc.reorder_point(params.sku)
            expiry = svc.expiry_risk(params.sku)

    parts = [
        f"{params.sku}: ~{fc['avg_daily_forecast']:.1f} units/day forecast ({fc['backend']})"
    ]
    if stockout:
        parts.append(f"stockout risk {stockout['risk_level']} (cover {stockout['days_of_cover']})")
    if reorder and reorder["delta"]:
        parts.append(f"reorder point {reorder['current_reorder_point']}->{reorder['recommended_reorder_point']}")
    if expiry:
        parts.append(f"{sum(e['projected_units_expiring'] for e in expiry)} units may expire")

    return ForecastOutput(
        sku=params.sku,
        backend=fc["backend"],
        horizon_days=fc["horizon_days"],
        forecast=fc["forecast"],
        avg_daily_forecast=fc["avg_daily_forecast"],
        stockout_risk=stockout,
        reorder_recommendation=reorder,
        expiry_risk=expiry,
        warnings=fc["warnings"],
        summary="; ".join(parts) + ".",
    )
