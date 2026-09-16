"""Supplier business logic.

Shaped like an external supplier-API client (Assumption A-03): the public
methods return plain dicts and hide all data access, so the body can later be
swapped for real HTTP calls without touching callers.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.core.exceptions import DataNotFoundError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.supplier_repo import SupplierRepository


class SupplierService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.suppliers = SupplierRepository(session)
        self.inv = InventoryRepository(session)

    def _resolve_supplier(
        self, supplier_id: int | None = None, supplier_name: str | None = None
    ) -> dict:
        if supplier_id is not None:
            supplier = self.suppliers.get_supplier(supplier_id)
            if supplier is None:
                raise DataNotFoundError(f"Unknown supplier id {supplier_id}.")
            return supplier
        if supplier_name:
            matches = self.suppliers.find_suppliers_by_name(supplier_name)
            if not matches:
                raise DataNotFoundError(f"No supplier matching '{supplier_name}'.")
            return matches[0]
        raise ValueError("Provide either supplier_id or supplier_name.")

    @staticmethod
    def _delay_days(po: dict) -> int | None:
        if po["received_date"] is None:
            return None
        return (po["received_date"] - po["promised_date"]).days

    def on_time_delivery_rate(self, supplier_id: int) -> float:
        """Fraction of delivered POs that arrived on or before the promised date."""
        pos = [
            po
            for po in self.suppliers.list_purchase_orders(supplier_id=supplier_id)
            if po["received_date"] is not None
        ]
        if not pos:
            return 0.0
        on_time = sum(1 for po in pos if self._delay_days(po) <= 0)
        return round(on_time / len(pos), 3)

    def average_delay_days(self, supplier_id: int) -> float:
        """Mean positive delay (days late) across late deliveries; 0 if none."""
        delays = [
            self._delay_days(po)
            for po in self.suppliers.list_purchase_orders(supplier_id=supplier_id)
            if po["received_date"] is not None and self._delay_days(po) > 0
        ]
        return round(sum(delays) / len(delays), 2) if delays else 0.0

    def open_purchase_orders(
        self, sku: str | None = None, supplier_id: int | None = None
    ) -> list[dict]:
        """Purchase orders that have not yet been received."""
        return [
            po
            for po in self.suppliers.list_purchase_orders(sku=sku, supplier_id=supplier_id)
            if po["received_date"] is None
        ]

    def late_purchase_orders(
        self, sku: str | None = None, supplier_id: int | None = None
    ) -> list[dict]:
        """Received purchase orders that arrived after the promised date."""
        rows = []
        for po in self.suppliers.list_purchase_orders(sku=sku, supplier_id=supplier_id, late_only=True):
            rows.append({**po, "delay_days": self._delay_days(po)})
        return rows

    def supplier_profile(
        self, supplier_id: int | None = None, supplier_name: str | None = None
    ) -> dict:
        """Full supplier record plus on-time rate, average delay and PO counts."""
        supplier = self._resolve_supplier(supplier_id, supplier_name)
        sid = supplier["id"]
        return {
            **supplier,
            "on_time_delivery_rate": self.on_time_delivery_rate(sid),
            "avg_delay_days": self.average_delay_days(sid),
            "open_po_count": len(self.open_purchase_orders(supplier_id=sid)),
            "late_po_count": len(self.late_purchase_orders(supplier_id=sid)),
        }

    def sku_supplier_reliability(self, sku: str) -> dict:
        """Reliability metrics for the supplier that provides a given SKU."""
        product = self.inv.get_product(sku)
        if product is None:
            raise DataNotFoundError(f"Unknown SKU '{sku}'.")
        supplier = self.suppliers.get_supplier(product["supplier_id"])
        if supplier is None:
            raise DataNotFoundError(f"SKU '{sku}' has no supplier on record.")
        sid = supplier["id"]
        return {
            "sku": sku,
            "supplier_id": sid,
            "supplier_name": supplier["name"],
            "reliability_score": supplier["reliability_score"],
            "on_time_delivery_rate": self.on_time_delivery_rate(sid),
            "avg_delay_days": self.average_delay_days(sid),
        }

    def list_supplier_status(self) -> list[dict]:
        """Profile summary for every supplier (used by the /metrics dashboard)."""
        return [self.supplier_profile(supplier_id=s["id"]) for s in self.suppliers.list_suppliers()]
