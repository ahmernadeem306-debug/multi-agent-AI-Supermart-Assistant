"""Inventory business logic: stock positions, low-stock, cover, batches, expiry.

Services depend only on repositories, never on the ORM, and contain no LLM
code so they are fully testable without an API key.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.core.exceptions import DataNotFoundError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository


def _today() -> dt.date:
    return dt.date.today()


class InventoryService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.inv = InventoryRepository(session)
        self.sales = SalesRepository(session)

    def _require_product(self, sku: str) -> dict:
        product = self.inv.get_product(sku)
        if product is None:
            matches = self.inv.find_products_by_name(sku, limit=5)
            hint = (
                f" Did you mean: {', '.join(m['sku'] for m in matches)}?" if matches else ""
            )
            raise DataNotFoundError(f"Unknown SKU '{sku}'.{hint}")
        return product

    def current_stock(self, sku: str) -> dict:
        """Return the latest shelf / backroom / on-hand position for one SKU."""
        product = self._require_product(sku)
        level = self.inv.get_latest_stock_level(sku)
        shelf = level["shelf_qty"] if level else 0
        backroom = level["backroom_qty"] if level else 0
        on_hand = level["on_hand_qty"] if level else 0
        return {
            "sku": sku,
            "name": product["name"],
            "aisle": product["aisle"],
            "snapshot_date": level["snapshot_date"] if level else None,
            "shelf_qty": shelf,
            "backroom_qty": backroom,
            "on_hand_qty": on_hand,
            "reorder_point": product["reorder_point"],
            "safety_stock": product["safety_stock"],
            "below_reorder_point": on_hand <= product["reorder_point"],
        }

    def low_stock(self, aisle: str | None = None, limit: int = 50) -> list[dict]:
        """Products whose latest on-hand quantity is at or below their reorder point."""
        rows: list[dict] = []
        for product in self.inv.list_products(aisle=aisle):
            level = self.inv.get_latest_stock_level(product["sku"])
            on_hand = level["on_hand_qty"] if level else 0
            if on_hand <= product["reorder_point"]:
                rows.append(
                    {
                        "sku": product["sku"],
                        "name": product["name"],
                        "aisle": product["aisle"],
                        "on_hand_qty": on_hand,
                        "reorder_point": product["reorder_point"],
                        "safety_stock": product["safety_stock"],
                        "shortfall": product["reorder_point"] - on_hand,
                    }
                )
        rows.sort(key=lambda r: r["shortfall"], reverse=True)
        return rows[:limit]

    def days_of_cover(self, sku: str, window_days: int = 30) -> dict:
        """On-hand quantity divided by mean daily units sold over the window."""
        product = self._require_product(sku)
        level = self.inv.get_latest_stock_level(sku)
        on_hand = level["on_hand_qty"] if level else 0
        end = _today()
        start = end - dt.timedelta(days=window_days)
        units = self.sales.total_units_sold(
            sku, dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)
        )
        avg_daily = units / window_days if window_days else 0.0
        cover = round(on_hand / avg_daily, 1) if avg_daily > 0 else None
        return {
            "sku": sku,
            "name": product["name"],
            "on_hand_qty": on_hand,
            "window_days": window_days,
            "avg_daily_units": round(avg_daily, 2),
            "days_of_cover": cover,
        }

    def batches(self, sku: str) -> list[dict]:
        """All tracked batches for a perishable SKU, earliest expiry first."""
        self._require_product(sku)
        today = _today()
        out = []
        for b in self.inv.list_batches(sku):
            out.append({**b, "days_until_expiry": (b["expiry_date"] - today).days})
        return out

    def expiring_batches(self, days_ahead: int = 7, aisle: str | None = None) -> list[dict]:
        """Batches expiring within ``days_ahead`` days, optionally filtered by aisle."""
        today = _today()
        products = {p["sku"]: p for p in self.inv.list_products()}
        rows = []
        for b in self.inv.list_expiring_batches(as_of=today, days_ahead=days_ahead):
            product = products.get(b["sku"])
            if product is None:
                continue
            if aisle is not None and product["aisle"] != aisle:
                continue
            rows.append(
                {
                    "sku": b["sku"],
                    "name": product["name"],
                    "aisle": product["aisle"],
                    "batch_no": b["batch_no"],
                    "expiry_date": b["expiry_date"],
                    "qty_remaining": b["qty_remaining"],
                    "days_until_expiry": (b["expiry_date"] - today).days,
                }
            )
        return rows

    def shelf_backroom_discrepancy(self, sku: str | None = None) -> list[dict]:
        """SKUs whose latest snapshot has shelf + backroom != recorded on-hand."""
        skus = [sku] if sku else [p["sku"] for p in self.inv.list_products()]
        if sku:
            self._require_product(sku)
        rows = []
        for s in skus:
            level = self.inv.get_latest_stock_level(s)
            if level is None:
                continue
            expected = level["shelf_qty"] + level["backroom_qty"]
            if expected != level["on_hand_qty"]:
                rows.append(
                    {
                        "sku": s,
                        "snapshot_date": level["snapshot_date"],
                        "shelf_qty": level["shelf_qty"],
                        "backroom_qty": level["backroom_qty"],
                        "on_hand_qty": level["on_hand_qty"],
                        "expected_on_hand": expected,
                        "discrepancy": level["on_hand_qty"] - expected,
                    }
                )
        return rows
