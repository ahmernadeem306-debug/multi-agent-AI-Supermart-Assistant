"""Typed, parameterised data access for suppliers and purchase orders."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import PurchaseOrder, Supplier


def _supplier_to_dict(s: Supplier) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "lead_time_days": s.lead_time_days,
        "reliability_score": s.reliability_score,
        "contact_email": s.contact_email,
        "contract_ref": s.contract_ref,
    }


def _po_to_dict(po: PurchaseOrder) -> dict:
    return {
        "po_id": po.po_id,
        "supplier_id": po.supplier_id,
        "sku": po.sku,
        "qty_ordered": po.qty_ordered,
        "order_date": po.order_date,
        "promised_date": po.promised_date,
        "received_date": po.received_date,
        "qty_received": po.qty_received,
        "status": po.status,
    }


class SupplierRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_supplier(self, **fields) -> dict:
        supplier = Supplier(**fields)
        self.session.add(supplier)
        self.session.flush()
        return _supplier_to_dict(supplier)

    def get_supplier(self, supplier_id: int) -> dict | None:
        supplier = self.session.get(Supplier, supplier_id)
        return _supplier_to_dict(supplier) if supplier else None

    def list_suppliers(self) -> list[dict]:
        stmt = select(Supplier)
        return [_supplier_to_dict(s) for s in self.session.scalars(stmt)]

    def find_suppliers_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Case-insensitive partial match on supplier name (for human input resolution)."""
        pattern = f"%{query}%"
        stmt = select(Supplier).where(Supplier.name.ilike(pattern)).limit(limit)
        return [_supplier_to_dict(s) for s in self.session.scalars(stmt)]

    def add_purchase_order(self, **fields) -> dict:
        po = PurchaseOrder(**fields)
        self.session.add(po)
        self.session.flush()
        return _po_to_dict(po)

    def list_purchase_orders(
        self,
        sku: str | None = None,
        supplier_id: int | None = None,
        late_only: bool = False,
    ) -> list[dict]:
        stmt = select(PurchaseOrder)
        if sku is not None:
            stmt = stmt.where(PurchaseOrder.sku == sku)
        if supplier_id is not None:
            stmt = stmt.where(PurchaseOrder.supplier_id == supplier_id)
        orders = [_po_to_dict(po) for po in self.session.scalars(stmt)]
        if late_only:
            orders = [
                po
                for po in orders
                if po["received_date"] is not None and po["received_date"] > po["promised_date"]
            ]
        return orders

    def list_late_deliveries(self, sku: str | None = None) -> list[dict]:
        return self.list_purchase_orders(sku=sku, late_only=True)
