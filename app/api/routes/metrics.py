"""Read-only /metrics routes backed entirely by the service layer.

No ORM or SQL here; every handler opens a read-only session and delegates to
a service. Unknown aisle/supplier -> DataNotFoundError -> 404 via the app's
exception handler.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.db.base import get_readonly_session
from app.services.analytics_service import AnalyticsService
from app.services.inventory_service import InventoryService
from app.services.supplier_service import SupplierService

router = APIRouter(prefix="/metrics", tags=["metrics"])

AnomalyKind = Literal["shrinkage", "late_delivery", "demand_shift", "stockout"]


@router.get("/aisles")
def aisle_metrics_all(days: int = Query(default=30, ge=1, le=730)) -> dict:
    with get_readonly_session() as session:
        return {"days": days, "aisles": AnalyticsService(session).all_aisle_metrics(days=days)}


@router.get("/aisle/{aisle}")
def aisle_metrics_one(aisle: str, days: int = Query(default=30, ge=1, le=730)) -> dict:
    with get_readonly_session() as session:
        return AnalyticsService(session).aisle_metrics(aisle, days=days)


@router.get("/low-stock")
def low_stock(
    aisle: str | None = None, limit: int = Query(default=50, ge=1, le=500)
) -> dict:
    with get_readonly_session() as session:
        items = InventoryService(session).low_stock(aisle=aisle, limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/expiring")
def expiring_batches(
    days_ahead: int = Query(default=7, ge=1, le=365), aisle: str | None = None
) -> dict:
    with get_readonly_session() as session:
        items = InventoryService(session).expiring_batches(days_ahead=days_ahead, aisle=aisle)
    return {"items": items, "count": len(items)}


@router.get("/suppliers")
def supplier_status() -> dict:
    with get_readonly_session() as session:
        return {"suppliers": SupplierService(session).list_supplier_status()}


@router.get("/anomalies")
def anomalies(
    kind: AnomalyKind | None = None, days: int = Query(default=90, ge=1, le=730)
) -> dict:
    kinds = [kind] if kind else ["shrinkage", "late_delivery", "demand_shift", "stockout"]
    with get_readonly_session() as session:
        svc = AnalyticsService(session)
        results = {k: svc.detect(k, days=days) for k in kinds}
    return {"days": days, "results": results}
