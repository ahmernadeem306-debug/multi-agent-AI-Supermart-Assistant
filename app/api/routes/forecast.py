"""Read-only /forecast routes backed by the forecasting service."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.db.base import get_readonly_session
from app.db.repositories.inventory_repo import InventoryRepository
from app.logging_config import get_logger
from app.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["forecast"])
logger = get_logger(__name__)


@router.get("/skus")
def list_skus() -> dict:
    with get_readonly_session() as session:
        products = InventoryRepository(session).list_products()
    return {
        "skus": [
            {
                "sku": p["sku"],
                "name": p["name"],
                "aisle": p["aisle"],
                "is_perishable": p["is_perishable"],
            }
            for p in sorted(products, key=lambda p: p["sku"])
        ]
    }


@router.get("/alerts")
def alerts(limit: int = Query(default=25, ge=1, le=200)) -> dict:
    with get_readonly_session() as session:
        result = ForecastingService(session).store_alerts(limit=limit)
    logger.info("forecast_alerts_served", count=result["count"])
    return result


@router.get("/reorder-recommendations")
def reorder_recommendations(limit: int = Query(default=25, ge=1, le=200)) -> dict:
    with get_readonly_session() as session:
        return ForecastingService(session).reorder_recommendations(limit=limit)


@router.get("/{sku}")
def sku_forecast(
    sku: str,
    horizon_days: int = Query(default=14, ge=1, le=30),
    include_risk: bool = Query(default=True),
) -> dict:
    with get_readonly_session() as session:
        svc = ForecastingService(session)
        payload = svc.forecast(sku, horizon_days)
        if include_risk:
            payload["stockout_risk"] = svc.stockout_risk(sku)
            payload["reorder_recommendation"] = svc.reorder_point(sku)
            payload["expiry_risk"] = svc.expiry_risk(sku)
    logger.info("forecast_served", sku=sku, backend=payload["backend"], horizon=horizon_days)
    return payload
