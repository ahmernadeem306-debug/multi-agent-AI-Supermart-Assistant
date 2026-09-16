"""Inventory- and anomaly-oriented MCP tool functions.

Every function is a plain importable callable with Pydantic-validated
arguments and a read-only database session. The FastMCP server registers
them; :class:`DirectToolProvider` imports and calls the same functions.
"""
from __future__ import annotations

import datetime as dt

from app.db.base import get_readonly_session
from app.mcp_server.schemas import (
    DetectAnomaliesInput,
    GetExpiringBatchesInput,
    GetProductInfoInput,
    GetStockLevelInput,
    ListLowStockInput,
    MAX_LIMIT,
    AnomaliesOutput,
    ItemListOutput,
    ProductMatchesOutput,
    StockLevelOutput,
)
from app.services.analytics_service import AnalyticsService
from app.services.inventory_service import InventoryService
from app.db.repositories.inventory_repo import InventoryRepository


def get_product_info(sku: str | None = None, name_query: str | None = None) -> ProductMatchesOutput:
    """Look up a product by exact SKU or by a partial name search.

    Use this to resolve a shopper- or manager-supplied product name into a
    canonical SKU before calling other tools. Pass ``sku`` for an exact
    lookup, or ``name_query`` for a case-insensitive partial-name match that
    returns candidate products.
    """
    params = GetProductInfoInput(sku=sku, name_query=name_query)
    with get_readonly_session() as session:
        repo = InventoryRepository(session)
        if params.sku:
            product = repo.get_product(params.sku)
            matches = [product] if product else []
        else:
            matches = repo.find_products_by_name(params.name_query, limit=MAX_LIMIT)
    if not matches:
        return ProductMatchesOutput(matches=[], summary="No matching product found.")
    if params.sku and matches:
        return ProductMatchesOutput(matches=matches, summary=f"{matches[0]['name']} ({matches[0]['sku']}).")
    return ProductMatchesOutput(
        matches=matches,
        summary=f"{len(matches)} product(s) match '{params.name_query}'.",
    )


def get_stock_level(sku: str, include_batches: bool = False) -> StockLevelOutput:
    """Return the current shelf, backroom and on-hand quantity for a SKU.

    Set ``include_batches=true`` to also list that SKU's perishable batches
    with their expiry dates. Raises a not-found error for an unknown SKU.
    """
    params = GetStockLevelInput(sku=sku, include_batches=include_batches)
    with get_readonly_session() as session:
        svc = InventoryService(session)
        stock = svc.current_stock(params.sku)
        batches = svc.batches(params.sku) if params.include_batches else []
    state = "below reorder point" if stock["below_reorder_point"] else "healthy"
    return StockLevelOutput(
        stock=stock,
        batches=batches,
        summary=f"{params.sku}: {stock['on_hand_qty']} on hand ({state}).",
    )


def list_low_stock(aisle: str | None = None, limit: int = 50) -> ItemListOutput:
    """List SKUs at or below their reorder point, worst shortfall first.

    Optionally restrict to a single ``aisle``. ``limit`` caps the number of
    rows returned (default 50, max 500).
    """
    params = ListLowStockInput(aisle=aisle, limit=limit)
    with get_readonly_session() as session:
        items = InventoryService(session).low_stock(aisle=params.aisle, limit=params.limit)
    scope = f" in {params.aisle}" if params.aisle else ""
    return ItemListOutput(items=items, count=len(items), summary=f"{len(items)} low-stock SKU(s){scope}.")


def get_expiring_batches(days_ahead: int = 7, aisle: str | None = None) -> ItemListOutput:
    """List perishable batches expiring within ``days_ahead`` days.

    Optionally restrict to a single ``aisle``. Results are ordered by the
    soonest expiry date first.
    """
    params = GetExpiringBatchesInput(days_ahead=days_ahead, aisle=aisle)
    with get_readonly_session() as session:
        items = InventoryService(session).expiring_batches(
            days_ahead=params.days_ahead, aisle=params.aisle
        )
    items = items[:MAX_LIMIT]
    return ItemListOutput(
        items=items,
        count=len(items),
        summary=f"{len(items)} batch(es) expiring within {params.days_ahead} day(s).",
    )


def detect_anomalies(kind: str, days: int = 90) -> AnomaliesOutput:
    """Run one operational anomaly detector over the trailing window.

    ``kind`` selects the detector: ``shrinkage`` (abnormal loss volume),
    ``late_delivery`` (purchase orders received after the promised date),
    ``demand_shift`` (sales stepping up well above baseline) or ``stockout``
    (recorded zero on-hand days). Returns evidence records, not prose.
    """
    params = DetectAnomaliesInput(kind=kind, days=days)
    with get_readonly_session() as session:
        items = AnalyticsService(session).detect(params.kind, days=params.days)
    items = items[:MAX_LIMIT]
    return AnomaliesOutput(
        kind=params.kind,
        items=items,
        count=len(items),
        summary=f"{len(items)} '{params.kind}' anomaly candidate(s) over ~{params.days} day(s).",
    )
