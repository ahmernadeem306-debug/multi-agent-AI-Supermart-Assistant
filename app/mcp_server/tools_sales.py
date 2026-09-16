"""Sales-oriented MCP tool functions."""
from __future__ import annotations

import datetime as dt

from app.db.base import get_readonly_session
from app.mcp_server.schemas import (
    GetSalesHistoryInput,
    GetTopSellersInput,
    MAX_LIMIT,
    ItemListOutput,
    SalesHistoryOutput,
)
from app.services.sales_service import SalesService


def get_sales_history(
    sku: str,
    start_date: str,
    end_date: str,
    granularity: str = "daily",
) -> SalesHistoryOutput:
    """Return units sold and revenue for a SKU between two dates.

    ``start_date`` and ``end_date`` are ISO dates (YYYY-MM-DD). ``granularity``
    is ``daily`` or ``weekly``. Use this to inspect sales velocity and trend
    for a single product.
    """
    params = GetSalesHistoryInput(
        sku=sku, start_date=start_date, end_date=end_date, granularity=granularity
    )
    with get_readonly_session() as session:
        data = SalesService(session).sales_history(
            params.sku, params.start_date, params.end_date, params.granularity
        )
    points = data["points"][:MAX_LIMIT]
    return SalesHistoryOutput(
        sku=params.sku,
        granularity=params.granularity,
        points=points,
        total_units=data["total_units"],
        total_revenue=data["total_revenue"],
        summary=(
            f"{params.sku}: {data['total_units']} units / "
            f"{data['total_revenue']:.2f} revenue over {len(points)} {params.granularity} bucket(s)."
        ),
    )


def get_top_sellers(aisle: str | None = None, days: int = 30, limit: int = 10) -> ItemListOutput:
    """List the best-selling SKUs by units over the trailing ``days`` window.

    Optionally restrict to a single ``aisle``. ``limit`` caps the number of
    rows (default 10, max 500).
    """
    # "all" / "any" / "store-wide" are natural ways an agent asks for no filter;
    # treat them as unset so the query spans every aisle instead of matching none.
    if (aisle or "").strip().lower() in {"", "all", "any", "store", "store-wide", "storewide"}:
        aisle = None
    params = GetTopSellersInput(aisle=aisle, days=days, limit=limit)
    with get_readonly_session() as session:
        items = SalesService(session).top_sellers(
            aisle=params.aisle, days=params.days, limit=params.limit
        )
    scope = f" in {params.aisle}" if params.aisle else ""
    return ItemListOutput(
        items=items,
        count=len(items),
        summary=f"Top {len(items)} seller(s){scope} over {params.days} day(s).",
    )
