"""Typed, parameterised data access for products, stock levels and batches.

This is the only layer permitted to touch app.db.models / the ORM for
inventory data. All methods return plain dicts, never ORM instances.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, StockBatch, StockLevel


def _product_to_dict(p: Product) -> dict:
    return {
        "sku": p.sku,
        "name": p.name,
        "category": p.category,
        "aisle": p.aisle,
        "unit_cost": p.unit_cost,
        "unit_price": p.unit_price,
        "is_perishable": p.is_perishable,
        "shelf_life_days": p.shelf_life_days,
        "reorder_point": p.reorder_point,
        "safety_stock": p.safety_stock,
        "supplier_id": p.supplier_id,
    }


def _stock_level_to_dict(s: StockLevel) -> dict:
    return {
        "id": s.id,
        "sku": s.sku,
        "snapshot_date": s.snapshot_date,
        "shelf_qty": s.shelf_qty,
        "backroom_qty": s.backroom_qty,
        "on_hand_qty": s.on_hand_qty,
    }


def _batch_to_dict(b: StockBatch) -> dict:
    return {
        "id": b.id,
        "sku": b.sku,
        "batch_no": b.batch_no,
        "received_date": b.received_date,
        "expiry_date": b.expiry_date,
        "qty_received": b.qty_received,
        "qty_remaining": b.qty_remaining,
    }


class InventoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_product(self, **fields) -> dict:
        product = Product(**fields)
        self.session.add(product)
        self.session.flush()
        return _product_to_dict(product)

    def get_product(self, sku: str) -> dict | None:
        product = self.session.get(Product, sku)
        return _product_to_dict(product) if product else None

    def list_products(self, aisle: str | None = None) -> list[dict]:
        stmt = select(Product)
        if aisle is not None:
            stmt = stmt.where(Product.aisle == aisle)
        return [_product_to_dict(p) for p in self.session.scalars(stmt)]

    def find_products_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Case-insensitive partial match on product name (for human input resolution)."""
        pattern = f"%{query}%"
        stmt = select(Product).where(Product.name.ilike(pattern)).limit(limit)
        return [_product_to_dict(p) for p in self.session.scalars(stmt)]

    def list_aisles(self) -> list[str]:
        stmt = select(Product.aisle).distinct().order_by(Product.aisle.asc())
        return [row for row in self.session.scalars(stmt)]

    def add_stock_level(self, **fields) -> dict:
        level = StockLevel(**fields)
        self.session.add(level)
        self.session.flush()
        return _stock_level_to_dict(level)

    def bulk_add_stock_levels(self, rows: list[dict]) -> None:
        """Bulk-insert many stock level snapshots at once (used by the seeder)."""
        if rows:
            self.session.execute(StockLevel.__table__.insert(), rows)

    def get_latest_stock_level(self, sku: str, as_of: dt.date | None = None) -> dict | None:
        stmt = select(StockLevel).where(StockLevel.sku == sku)
        if as_of is not None:
            stmt = stmt.where(StockLevel.snapshot_date <= as_of)
        stmt = stmt.order_by(StockLevel.snapshot_date.desc()).limit(1)
        level = self.session.scalars(stmt).first()
        return _stock_level_to_dict(level) if level else None

    def list_stock_levels(self, sku: str, start_date: dt.date, end_date: dt.date) -> list[dict]:
        stmt = (
            select(StockLevel)
            .where(StockLevel.sku == sku)
            .where(StockLevel.snapshot_date >= start_date)
            .where(StockLevel.snapshot_date <= end_date)
            .order_by(StockLevel.snapshot_date.asc())
        )
        return [_stock_level_to_dict(s) for s in self.session.scalars(stmt)]

    def add_stock_batch(self, **fields) -> dict:
        batch = StockBatch(**fields)
        self.session.add(batch)
        self.session.flush()
        return _batch_to_dict(batch)

    def list_batches(self, sku: str) -> list[dict]:
        stmt = select(StockBatch).where(StockBatch.sku == sku).order_by(StockBatch.expiry_date.asc())
        return [_batch_to_dict(b) for b in self.session.scalars(stmt)]

    def list_expiring_batches(self, as_of: dt.date, days_ahead: int) -> list[dict]:
        horizon = as_of + dt.timedelta(days=days_ahead)
        stmt = (
            select(StockBatch)
            .where(StockBatch.expiry_date >= as_of)
            .where(StockBatch.expiry_date <= horizon)
            .where(StockBatch.qty_remaining > 0)
            .order_by(StockBatch.expiry_date.asc())
        )
        return [_batch_to_dict(b) for b in self.session.scalars(stmt)]
