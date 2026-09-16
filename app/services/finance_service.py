"""Finance business logic: gross margin, revenue/COGS, shrinkage valuation."""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.core.exceptions import DataNotFoundError
from app.db.repositories.finance_repo import FinanceRepository
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository


def _today() -> dt.date:
    return dt.date.today()


def _span(start: dt.date, end: dt.date) -> tuple[dt.datetime, dt.datetime]:
    return dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)


def _margin_pct(revenue: float, cogs: float) -> float:
    return round((revenue - cogs) / revenue * 100, 2) if revenue > 0 else 0.0


class FinanceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.finance = FinanceRepository(session)
        self.inv = InventoryRepository(session)
        self.sales = SalesRepository(session)

    def _require_product(self, sku: str) -> dict:
        product = self.inv.get_product(sku)
        if product is None:
            raise DataNotFoundError(f"Unknown SKU '{sku}'.")
        return product

    def _sku_revenue_cogs(self, product: dict, start: dt.date, end: dt.date) -> tuple[float, float, int]:
        txns = self.sales.list_transactions(product["sku"], *_span(start, end))
        units = sum(t["qty"] for t in txns)
        revenue = sum(t["qty"] * t["unit_price"] - t["discount"] for t in txns)
        cogs = units * product["unit_cost"]
        return round(float(revenue), 2), round(float(cogs), 2), int(units)

    def sku_margin(self, sku: str, days: int = 30) -> dict:
        """Revenue, COGS and gross margin for one SKU over the trailing window."""
        product = self._require_product(sku)
        end = _today()
        revenue, cogs, units = self._sku_revenue_cogs(product, end - dt.timedelta(days=days), end)
        return {
            "sku": sku,
            "name": product["name"],
            "aisle": product["aisle"],
            "days": days,
            "units": units,
            "revenue": revenue,
            "cogs": cogs,
            "gross_margin": round(revenue - cogs, 2),
            "gross_margin_pct": _margin_pct(revenue, cogs),
        }

    def aisle_margin(self, aisle: str, days: int = 30) -> dict:
        """Aggregated revenue, COGS and gross margin for a whole aisle."""
        products = self.inv.list_products(aisle=aisle)
        if not products:
            raise DataNotFoundError(f"Unknown aisle '{aisle}'.")
        end = _today()
        start = end - dt.timedelta(days=days)
        revenue = cogs = 0.0
        for product in products:
            r, c, _ = self._sku_revenue_cogs(product, start, end)
            revenue += r
            cogs += c
        return {
            "aisle": aisle,
            "days": days,
            "sku_count": len(products),
            "revenue": round(revenue, 2),
            "cogs": round(cogs, 2),
            "gross_margin": round(revenue - cogs, 2),
            "gross_margin_pct": _margin_pct(revenue, cogs),
        }

    def revenue_and_cogs(self, days: int = 30, aisle: str | None = None) -> dict:
        """Store-wide (or single-aisle) revenue, COGS and gross margin."""
        end = _today()
        start = end - dt.timedelta(days=days)
        revenue = cogs = 0.0
        for product in self.inv.list_products(aisle=aisle):
            r, c, _ = self._sku_revenue_cogs(product, start, end)
            revenue += r
            cogs += c
        return {
            "days": days,
            "aisle": aisle,
            "revenue": round(revenue, 2),
            "cogs": round(cogs, 2),
            "gross_margin": round(revenue - cogs, 2),
            "gross_margin_pct": _margin_pct(revenue, cogs),
        }

    def shrinkage_valuation(
        self, sku: str | None = None, aisle: str | None = None, days: int = 90
    ) -> dict:
        """Cost of recorded shrinkage over the window, broken down by reason."""
        end = _today()
        start = end - dt.timedelta(days=days)
        products = {p["sku"]: p for p in self.inv.list_products(aisle=aisle)}
        if sku is not None:
            self._require_product(sku)
            products = {sku: products.get(sku) or self.inv.get_product(sku)}

        events = self.finance.list_shrinkage_events(sku=sku, start_date=start, end_date=end)
        total_qty = 0
        total_cost = 0.0
        by_reason: dict[str, dict] = {}
        for e in events:
            product = products.get(e["sku"])
            if aisle is not None and product is None:
                continue
            unit_cost = product["unit_cost"] if product else 0.0
            cost = e["qty"] * unit_cost
            total_qty += e["qty"]
            total_cost += cost
            bucket = by_reason.setdefault(e["reason"], {"qty": 0, "cost": 0.0})
            bucket["qty"] += e["qty"]
            bucket["cost"] = round(bucket["cost"] + cost, 2)
        return {
            "days": days,
            "sku": sku,
            "aisle": aisle,
            "event_count": len([e for e in events if aisle is None or products.get(e["sku"])]),
            "total_qty": total_qty,
            "total_cost": round(total_cost, 2),
            "by_reason": by_reason,
        }

    def margin_erosion_ranking(self, days: int = 30, limit: int = 10) -> list[dict]:
        """SKUs ranked by how much shrinkage cost erodes their gross margin."""
        end = _today()
        start = end - dt.timedelta(days=days)
        rows = []
        for product in self.inv.list_products():
            revenue, cogs, _ = self._sku_revenue_cogs(product, start, end)
            gross_margin = revenue - cogs
            events = self.finance.list_shrinkage_events(
                sku=product["sku"], start_date=start, end_date=end
            )
            shrinkage_cost = sum(e["qty"] for e in events) * product["unit_cost"]
            if shrinkage_cost <= 0:
                continue
            rows.append(
                {
                    "sku": product["sku"],
                    "name": product["name"],
                    "aisle": product["aisle"],
                    "gross_margin": round(gross_margin, 2),
                    "shrinkage_cost": round(float(shrinkage_cost), 2),
                    "net_margin": round(gross_margin - shrinkage_cost, 2),
                    "erosion_pct": round(shrinkage_cost / gross_margin * 100, 2)
                    if gross_margin > 0
                    else None,
                }
            )
        rows.sort(key=lambda r: r["shrinkage_cost"], reverse=True)
        return rows[:limit]
