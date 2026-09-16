"""Sales business logic: history, top sellers, velocity, demand-shift detection."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy.orm import Session

from app.core.exceptions import DataNotFoundError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository


def _today() -> dt.date:
    return dt.date.today()


def _span(start: dt.date, end: dt.date) -> tuple[dt.datetime, dt.datetime]:
    return dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)


class SalesService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.sales = SalesRepository(session)
        self.inv = InventoryRepository(session)

    def _require_product(self, sku: str) -> dict:
        product = self.inv.get_product(sku)
        if product is None:
            raise DataNotFoundError(f"Unknown SKU '{sku}'.")
        return product

    def sales_history(
        self,
        sku: str,
        start_date: dt.date,
        end_date: dt.date,
        granularity: str = "daily",
    ) -> dict:
        """Units and revenue for one SKU, bucketed daily or weekly."""
        if granularity not in {"daily", "weekly"}:
            raise ValueError("granularity must be 'daily' or 'weekly'")
        self._require_product(sku)
        start_ts, end_ts = _span(start_date, end_date)
        txns = self.sales.list_transactions(sku, start_ts, end_ts)
        points: list[dict] = []
        if txns:
            frame = pd.DataFrame(txns)
            frame["revenue"] = frame["qty"] * frame["unit_price"] - frame["discount"]
            frame["day"] = pd.to_datetime(frame["ts"]).dt.date
            if granularity == "daily":
                grouped = frame.groupby("day").agg(units=("qty", "sum"), revenue=("revenue", "sum"))
                points = [
                    {"period": str(idx), "units": int(row.units), "revenue": round(float(row.revenue), 2)}
                    for idx, row in grouped.iterrows()
                ]
            else:
                iso = pd.to_datetime(frame["ts"]).dt.isocalendar()
                frame["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
                grouped = frame.groupby("week").agg(units=("qty", "sum"), revenue=("revenue", "sum"))
                points = [
                    {"period": idx, "units": int(row.units), "revenue": round(float(row.revenue), 2)}
                    for idx, row in grouped.iterrows()
                ]
        return {
            "sku": sku,
            "granularity": granularity,
            "start_date": start_date,
            "end_date": end_date,
            "points": points,
            "total_units": sum(p["units"] for p in points),
            "total_revenue": round(sum(p["revenue"] for p in points), 2),
        }

    def top_sellers(self, aisle: str | None = None, days: int = 30, limit: int = 10) -> list[dict]:
        """Best-selling SKUs by units over the trailing window, optionally by aisle."""
        end = _today()
        start_ts, end_ts = _span(end - dt.timedelta(days=days), end)
        rows = []
        for product in self.inv.list_products(aisle=aisle):
            txns = self.sales.list_transactions(product["sku"], start_ts, end_ts)
            units = sum(t["qty"] for t in txns)
            revenue = sum(t["qty"] * t["unit_price"] - t["discount"] for t in txns)
            if units:
                rows.append(
                    {
                        "sku": product["sku"],
                        "name": product["name"],
                        "aisle": product["aisle"],
                        "units": int(units),
                        "revenue": round(float(revenue), 2),
                    }
                )
        rows.sort(key=lambda r: r["units"], reverse=True)
        return rows[:limit]

    def velocity(self, sku: str, window_days: int = 30) -> dict:
        """Mean daily units sold for a SKU over the trailing window."""
        self._require_product(sku)
        end = _today()
        start_ts, end_ts = _span(end - dt.timedelta(days=window_days), end)
        units = self.sales.total_units_sold(sku, start_ts, end_ts)
        return {
            "sku": sku,
            "window_days": window_days,
            "total_units": units,
            "avg_daily_units": round(units / window_days, 2) if window_days else 0.0,
        }

    def demand_shift(self, sku: str, recent_days: int = 30, baseline_days: int = 90) -> dict:
        """Compare recent mean daily units against an earlier baseline window."""
        self._require_product(sku)
        end = _today()
        recent_start = end - dt.timedelta(days=recent_days)
        baseline_end = recent_start
        baseline_start = baseline_end - dt.timedelta(days=baseline_days)

        recent_units = self.sales.total_units_sold(sku, *_span(recent_start, end))
        baseline_units = self.sales.total_units_sold(sku, *_span(baseline_start, baseline_end))
        recent_avg = recent_units / recent_days if recent_days else 0.0
        baseline_avg = baseline_units / baseline_days if baseline_days else 0.0
        ratio = round(recent_avg / baseline_avg, 2) if baseline_avg > 0 else None
        return {
            "sku": sku,
            "recent_days": recent_days,
            "baseline_days": baseline_days,
            "recent_avg_daily": round(recent_avg, 2),
            "baseline_avg_daily": round(baseline_avg, 2),
            "ratio": ratio,
            "shifted": ratio is not None and ratio >= 1.5,
        }
