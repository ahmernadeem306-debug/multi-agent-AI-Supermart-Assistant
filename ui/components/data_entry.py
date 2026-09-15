"""Manual + CSV data entry for a real store's own data.

This talks to the *same* SQLite database the API, agents and seed script
already use (``app.db.base.get_session`` / the existing repository layer —
the exact same write path ``scripts/seed_db.py`` calls). Nothing here
creates a new database or touches the FastAPI app, the LangGraph agents or
the MCP tool server: agents keep using ``get_readonly_session`` everywhere
else, so this admin page is the only place that writes, and it writes
through the pre-existing, already-tested repository methods.
"""
from __future__ import annotations

import datetime as dt
import sys
import uuid
from pathlib import Path

import pandas as pd

# Make ``app.*`` importable: this file lives at BizAgent/ui/components/, so
# parents[2] is the project root that contains the ``app`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db.base import get_session  # noqa: E402
from app.db.repositories.finance_repo import FinanceRepository  # noqa: E402
from app.db.repositories.inventory_repo import InventoryRepository  # noqa: E402
from app.db.repositories.sales_repo import SalesRepository  # noqa: E402
from app.db.repositories.supplier_repo import SupplierRepository  # noqa: E402


# ---------------------------------------------------------------------------
# Lookups for dropdowns
# ---------------------------------------------------------------------------
def list_suppliers() -> list[dict]:
    with get_session() as session:
        return SupplierRepository(session).list_suppliers()


def list_products() -> list[dict]:
    with get_session() as session:
        return InventoryRepository(session).list_products()


def list_aisles() -> list[str]:
    with get_session() as session:
        aisles = InventoryRepository(session).list_aisles()
    return aisles or ["Produce", "Dairy", "Bakery", "Frozen", "Beverages", "Snacks",
                       "Household", "Personal Care"]


# ---------------------------------------------------------------------------
# Manual single-row entry (used by the "Manual Entry" forms)
# ---------------------------------------------------------------------------
def add_supplier(**fields) -> dict:
    with get_session() as session:
        return SupplierRepository(session).add_supplier(**fields)


def add_product(**fields) -> dict:
    with get_session() as session:
        return InventoryRepository(session).add_product(**fields)


def add_stock_level(**fields) -> dict:
    with get_session() as session:
        return InventoryRepository(session).add_stock_level(**fields)


def add_stock_batch(**fields) -> dict:
    with get_session() as session:
        return InventoryRepository(session).add_stock_batch(**fields)


def add_transaction(**fields) -> dict:
    fields.setdefault("txn_id", f"TXN-{uuid.uuid4().hex[:10]}")
    with get_session() as session:
        return SalesRepository(session).add_transaction(**fields)


def add_shrinkage_event(**fields) -> dict:
    with get_session() as session:
        return FinanceRepository(session).add_shrinkage_event(**fields)


# ---------------------------------------------------------------------------
# CSV bulk import — one row failing (e.g. duplicate SKU) is isolated with a
# SAVEPOINT so it does not roll back the rest of the file.
# ---------------------------------------------------------------------------
ENTITY_TEMPLATES: dict[str, dict] = {
    "suppliers": {
        "columns": ["name", "lead_time_days", "reliability_score", "contact_email", "contract_ref"],
        "example": ["Acme Foods Ltd", 7, 0.95, "orders@acmefoods.test", "CTR-2001"],
        "required": ["name", "lead_time_days", "reliability_score", "contact_email", "contract_ref"],
    },
    "products": {
        "columns": ["sku", "name", "category", "aisle", "unit_cost", "unit_price",
                    "is_perishable", "shelf_life_days", "reorder_point", "safety_stock", "supplier_id"],
        "example": ["SKU-2001", "Whole Milk 1L", "Milk", "Dairy", 1.20, 1.99,
                    True, 14, 20, 10, 1],
        "required": ["sku", "name", "category", "aisle", "unit_cost", "unit_price",
                     "reorder_point", "safety_stock", "supplier_id"],
    },
    "stock_levels": {
        "columns": ["sku", "snapshot_date", "shelf_qty", "backroom_qty", "on_hand_qty"],
        "example": ["SKU-2001", "2026-09-10", 30, 50, 80],
        "required": ["sku", "snapshot_date", "shelf_qty", "backroom_qty", "on_hand_qty"],
    },
    "stock_batches": {
        "columns": ["sku", "batch_no", "received_date", "expiry_date", "qty_received", "qty_remaining"],
        "example": ["SKU-2001", "B-2026-0910-1", "2026-09-10", "2026-09-24", 100, 100],
        "required": ["sku", "batch_no", "received_date", "expiry_date", "qty_received", "qty_remaining"],
    },
    "sales_transactions": {
        "columns": ["txn_id", "ts", "sku", "qty", "unit_price", "discount", "register_id"],
        "example": ["TXN-0001", "2026-09-10 14:30:00", "SKU-2001", 2, 1.99, 0.0, "REG-1"],
        "required": ["ts", "sku", "qty", "unit_price", "register_id"],
    },
    "shrinkage_events": {
        "columns": ["sku", "event_date", "qty", "reason", "notes"],
        "example": ["SKU-2001", "2026-09-10", 3, "damage", "Dropped crate in aisle"],
        "required": ["sku", "event_date", "qty", "reason"],
    },
}

_DATE_COLS = {
    "suppliers": [],
    "products": [],
    "stock_levels": ["snapshot_date"],
    "stock_batches": ["received_date", "expiry_date"],
    "sales_transactions": ["ts"],
    "shrinkage_events": ["event_date"],
}


def csv_template(entity: str) -> bytes:
    spec = ENTITY_TEMPLATES[entity]
    frame = pd.DataFrame([spec["example"]], columns=spec["columns"])
    return frame.to_csv(index=False).encode("utf-8")


def validate_columns(entity: str, frame: pd.DataFrame) -> list[str]:
    """Return a list of missing required columns (empty = OK to import)."""
    required = ENTITY_TEMPLATES[entity]["required"]
    return [c for c in required if c not in frame.columns]


def _coerce_row(entity: str, row: dict) -> dict:
    row = {k: v for k, v in row.items() if pd.notna(v)}
    for col in _DATE_COLS.get(entity, []):
        if col in row:
            parsed = pd.to_datetime(row[col])
            row[col] = parsed.date() if entity != "sales_transactions" else parsed.to_pydatetime()
    if entity == "products" and "is_perishable" in row:
        row["is_perishable"] = bool(row["is_perishable"])
    if entity == "products":
        row["supplier_id"] = int(row["supplier_id"])
    if entity in ("stock_levels", "stock_batches", "shrinkage_events", "products"):
        for int_col in ("shelf_qty", "backroom_qty", "on_hand_qty", "qty_received",
                        "qty_remaining", "qty", "reorder_point", "safety_stock",
                        "shelf_life_days", "lead_time_days"):
            if int_col in row:
                row[int_col] = int(row[int_col])
    if entity == "sales_transactions":
        if "txn_id" not in row or not row["txn_id"]:
            row["txn_id"] = f"TXN-{uuid.uuid4().hex[:10]}"
        row.setdefault("discount", 0.0)
    return row


_REPO_ADD = {
    "suppliers": ("SupplierRepository", "add_supplier"),
    "products": ("InventoryRepository", "add_product"),
    "stock_levels": ("InventoryRepository", "add_stock_level"),
    "stock_batches": ("InventoryRepository", "add_stock_batch"),
    "sales_transactions": ("SalesRepository", "add_transaction"),
    "shrinkage_events": ("FinanceRepository", "add_shrinkage_event"),
}

_REPO_CLASSES = {
    "SupplierRepository": SupplierRepository,
    "InventoryRepository": InventoryRepository,
    "SalesRepository": SalesRepository,
    "FinanceRepository": FinanceRepository,
}


def import_csv(entity: str, frame: pd.DataFrame) -> tuple[int, list[str]]:
    """Import every row of ``frame`` for ``entity``.

    Each row is inserted in its own SAVEPOINT, so one bad row (e.g. a
    duplicate SKU or a missing supplier) is skipped and reported instead of
    discarding the whole file. Returns (success_count, list_of_error_strings).
    """
    repo_cls_name, method_name = _REPO_ADD[entity]
    repo_cls = _REPO_CLASSES[repo_cls_name]
    successes = 0
    errors: list[str] = []
    with get_session() as session:
        repo = repo_cls(session)
        for i, row in enumerate(frame.to_dict(orient="records"), start=2):  # row 2 = first data row
            try:
                fields = _coerce_row(entity, row)
                with session.begin_nested():
                    getattr(repo, method_name)(**fields)
                successes += 1
            except Exception as exc:  # noqa: BLE001 - report and keep going
                errors.append(f"Row {i}: {exc}")
    return successes, errors
