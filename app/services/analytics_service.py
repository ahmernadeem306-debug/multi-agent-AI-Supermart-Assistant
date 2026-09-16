"""Cross-cutting analytics: aisle-level metrics and anomaly detectors.

The four detectors return typed, evidence-carrying records (SKUs, dates,
magnitudes, baselines) rather than prose. They are consumed by the Day 4
root-cause-analysis workflow and by the ``detect_anomalies`` MCP tool.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.core.exceptions import DataNotFoundError
from app.db.repositories.finance_repo import FinanceRepository
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.db.repositories.supplier_repo import SupplierRepository
from app.services.finance_service import FinanceService
from app.services.inventory_service import InventoryService
from app.services.sales_service import SalesService

ANOMALY_KINDS = ("shrinkage", "late_delivery", "demand_shift", "stockout")

_SHRINKAGE_MIN_QTY = 30
_SHRINKAGE_RATIO = 2.0
_DEMAND_SHIFT_RATIO = 1.5
_LATE_DELIVERY_DEFAULT_DAYS = 400


def _today() -> dt.date:
    return dt.date.today()


class AnalyticsService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.inv_repo = InventoryRepository(session)
        self.finance_repo = FinanceRepository(session)
        self.sales_repo = SalesRepository(session)
        self.supplier_repo = SupplierRepository(session)
        self.inventory = InventoryService(session)
        self.sales = SalesService(session)
        self.finance = FinanceService(session)

    # ------------------------------------------------------------------ metrics
    def aisle_metrics(self, aisle: str, days: int = 30) -> dict:
        """Dashboard metrics for a single aisle over the trailing window."""
        products = self.inv_repo.list_products(aisle=aisle)
        if not products:
            raise DataNotFoundError(f"Unknown aisle '{aisle}'.")
        end = _today()
        start = end, end - dt.timedelta(days=days)

        units_sold = 0
        out_of_stock = 0
        low_stock = 0
        for product in products:
            level = self.inv_repo.get_latest_stock_level(product["sku"])
            on_hand = level["on_hand_qty"] if level else 0
            if on_hand == 0:
                out_of_stock += 1
            elif on_hand <= product["reorder_point"]:
                low_stock += 1
            units_sold += self.sales_repo.total_units_sold(
                product["sku"],
                dt.datetime.combine(start[1], dt.time.min),
                dt.datetime.combine(end, dt.time.max),
            )

        margin = self.finance.aisle_margin(aisle, days=days)
        shrinkage = self.finance.shrinkage_valuation(aisle=aisle, days=max(days, 90))
        expiring = self.inventory.expiring_batches(days_ahead=7, aisle=aisle)
        return {
            "aisle": aisle,
            "days": days,
            "sku_count": len(products),
            "revenue": margin["revenue"],
            "units_sold": int(units_sold),
            "out_of_stock_count": out_of_stock,
            "low_stock_count": low_stock,
            "shrinkage_value": shrinkage["total_cost"],
            "avg_margin_pct": margin["gross_margin_pct"],
            "expiring_batch_count": len(expiring),
        }

    def all_aisle_metrics(self, days: int = 30) -> list[dict]:
        return [self.aisle_metrics(aisle, days=days) for aisle in self.inv_repo.list_aisles()]

    # ---------------------------------------------------------------- detectors
    def detect_shrinkage_spikes(self, days: int = 90) -> list[dict]:
        """SKUs whose recent shrinkage volume dwarfs their earlier baseline."""
        end = _today()
        recent_start = end - dt.timedelta(days=days)
        baseline_start = recent_start - dt.timedelta(days=days)
        results = []
        for product in self.inv_repo.list_products():
            sku = product["sku"]
            recent = self.finance_repo.list_shrinkage_events(
                sku=sku, start_date=recent_start, end_date=end
            )
            baseline = self.finance_repo.list_shrinkage_events(
                sku=sku, start_date=baseline_start, end_date=recent_start
            )
            recent_qty = sum(e["qty"] for e in recent)
            baseline_qty = sum(e["qty"] for e in baseline)
            if recent_qty < _SHRINKAGE_MIN_QTY:
                continue
            if baseline_qty > 0 and recent_qty / baseline_qty < _SHRINKAGE_RATIO:
                continue
            reasons: dict[str, int] = {}
            for e in recent:
                reasons[e["reason"]] = reasons.get(e["reason"], 0) + e["qty"]
            dominant = max(reasons, key=reasons.get)
            results.append(
                {
                    "sku": sku,
                    "aisle": product["aisle"],
                    "recent_qty": recent_qty,
                    "baseline_qty": baseline_qty,
                    "ratio": round(recent_qty / baseline_qty, 2) if baseline_qty else None,
                    "dominant_reason": dominant,
                    "event_count": len(recent),
                    "first_event": min(e["event_date"] for e in recent),
                    "last_event": max(e["event_date"] for e in recent),
                }
            )
        return results

    def detect_late_deliveries(self, days: int = _LATE_DELIVERY_DEFAULT_DAYS) -> list[dict]:
        """Purchase orders ordered within the window that arrived after promise."""
        cutoff = _today() - dt.timedelta(days=days)
        results = []
        for po in self.supplier_repo.list_purchase_orders(late_only=True):
            if po["order_date"] < cutoff:
                continue
            results.append(
                {
                    "sku": po["sku"],
                    "po_id": po["po_id"],
                    "supplier_id": po["supplier_id"],
                    "order_date": po["order_date"],
                    "promised_date": po["promised_date"],
                    "received_date": po["received_date"],
                    "delay_days": (po["received_date"] - po["promised_date"]).days,
                }
            )
        results.sort(key=lambda r: r["delay_days"], reverse=True)
        return results

    def detect_demand_shifts(self, days: int = 60, baseline_days: int = 90) -> list[dict]:
        """SKUs whose recent mean daily sales are >= 1.5x their earlier baseline."""
        results = []
        for product in self.inv_repo.list_products():
            shift = self.sales.demand_shift(
                product["sku"], recent_days=days, baseline_days=baseline_days
            )
            if shift["ratio"] is not None and shift["ratio"] >= _DEMAND_SHIFT_RATIO:
                results.append(
                    {
                        "sku": product["sku"],
                        "aisle": product["aisle"],
                        "recent_avg_daily": shift["recent_avg_daily"],
                        "baseline_avg_daily": shift["baseline_avg_daily"],
                        "ratio": shift["ratio"],
                        "reorder_point": product["reorder_point"],
                    }
                )
        results.sort(key=lambda r: r["ratio"], reverse=True)
        return results

    def detect_stockout_events(self, days: int = 90) -> list[dict]:
        """SKUs that recorded one or more zero on-hand days within the window."""
        end = _today()
        start = end - dt.timedelta(days=days)
        results = []
        for product in self.inv_repo.list_products():
            levels = self.inv_repo.list_stock_levels(product["sku"], start, end)
            zero_days = [lvl["snapshot_date"] for lvl in levels if lvl["on_hand_qty"] == 0]
            if not zero_days:
                continue
            results.append(
                {
                    "sku": product["sku"],
                    "aisle": product["aisle"],
                    "stockout_days": len(zero_days),
                    "first_stockout": min(zero_days),
                    "last_stockout": max(zero_days),
                    "reorder_point": product["reorder_point"],
                }
            )
        results.sort(key=lambda r: r["stockout_days"], reverse=True)
        return results

    def detect(self, kind: str, days: int = 90) -> list[dict]:
        """Dispatch to a single detector by kind (used by the MCP tool)."""
        if kind == "shrinkage":
            return self.detect_shrinkage_spikes(days=days)
        if kind == "late_delivery":
            return self.detect_late_deliveries(days=max(days, _LATE_DELIVERY_DEFAULT_DAYS))
        if kind == "demand_shift":
            return self.detect_demand_shifts(days=days)
        if kind == "stockout":
            return self.detect_stockout_events(days=days)
        raise ValueError(f"Unknown anomaly kind '{kind}'. Expected one of {ANOMALY_KINDS}.")
