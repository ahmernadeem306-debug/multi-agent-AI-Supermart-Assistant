"""SQLAlchemy 2.x ORM models for the 9 tables in the BizAgent data model.

See BIZAGENT_BUILD_PLAN.md Section 1.10 for the authoritative schema.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import ForeignKey, Index, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Supplier(Base):
    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    lead_time_days: Mapped[int] = mapped_column(nullable=False)
    reliability_score: Mapped[float] = mapped_column(nullable=False)
    contact_email: Mapped[str] = mapped_column(nullable=False)
    contract_ref: Mapped[str] = mapped_column(nullable=False)

    products: Mapped[list["Product"]] = relationship(back_populates="supplier")


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    category: Mapped[str] = mapped_column(nullable=False)
    aisle: Mapped[str] = mapped_column(nullable=False, index=True)
    unit_cost: Mapped[float] = mapped_column(nullable=False)
    unit_price: Mapped[float] = mapped_column(nullable=False)
    is_perishable: Mapped[bool] = mapped_column(nullable=False, default=False)
    shelf_life_days: Mapped[int | None] = mapped_column(nullable=True)
    reorder_point: Mapped[int] = mapped_column(nullable=False)
    safety_stock: Mapped[int] = mapped_column(nullable=False)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), nullable=False)

    supplier: Mapped["Supplier"] = relationship(back_populates="products")


class StockLevel(Base):
    __tablename__ = "stock_levels"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), nullable=False)
    snapshot_date: Mapped[dt.date] = mapped_column(nullable=False)
    shelf_qty: Mapped[int] = mapped_column(nullable=False)
    backroom_qty: Mapped[int] = mapped_column(nullable=False)
    on_hand_qty: Mapped[int] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_stock_levels_sku_snapshot_date", "sku", "snapshot_date"),
    )


class StockBatch(Base):
    __tablename__ = "stock_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), nullable=False, index=True)
    batch_no: Mapped[str] = mapped_column(nullable=False)
    received_date: Mapped[dt.date] = mapped_column(nullable=False)
    expiry_date: Mapped[dt.date] = mapped_column(nullable=False, index=True)
    qty_received: Mapped[int] = mapped_column(nullable=False)
    qty_remaining: Mapped[int] = mapped_column(nullable=False)


class SalesTransaction(Base):
    __tablename__ = "sales_transactions"

    txn_id: Mapped[str] = mapped_column(primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(nullable=False)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), nullable=False)
    qty: Mapped[int] = mapped_column(nullable=False)
    unit_price: Mapped[float] = mapped_column(nullable=False)
    discount: Mapped[float] = mapped_column(nullable=False, default=0.0)
    register_id: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_sales_transactions_sku_ts", "sku", "ts"),
    )


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    po_id: Mapped[str] = mapped_column(primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), nullable=False)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), nullable=False)
    qty_ordered: Mapped[int] = mapped_column(nullable=False)
    order_date: Mapped[dt.date] = mapped_column(nullable=False)
    promised_date: Mapped[dt.date] = mapped_column(nullable=False)
    received_date: Mapped[dt.date | None] = mapped_column(nullable=True)
    qty_received: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_purchase_orders_sku_status", "sku", "status"),
    )


class ShrinkageEvent(Base):
    __tablename__ = "shrinkage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), nullable=False)
    event_date: Mapped[dt.date] = mapped_column(nullable=False)
    qty: Mapped[int] = mapped_column(nullable=False)
    reason: Mapped[str] = mapped_column(nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_shrinkage_events_sku_event_date", "sku", "event_date"),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(nullable=False)
    doc_type: Mapped[str] = mapped_column(nullable=False)
    source_path: Mapped[str] = mapped_column(nullable=False)
    sha256: Mapped[str] = mapped_column(nullable=False, unique=True)
    ingested_at: Mapped[dt.datetime] = mapped_column(nullable=False)
    chunk_count: Mapped[int] = mapped_column(nullable=False, default=0)


class AgentRun(Base):
    """Agent decision log — one row per /query (or later, per agent run)."""

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str | None] = mapped_column(nullable=True)
    agents_invoked: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    retrieved_docs: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
