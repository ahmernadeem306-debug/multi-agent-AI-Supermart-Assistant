"""Supplier- and purchase-order MCP tool functions."""
from __future__ import annotations

from app.core.exceptions import DataNotFoundError
from app.db.base import get_readonly_session
from app.db.repositories.supplier_repo import SupplierRepository
from app.mcp_server.schemas import (
    GetOpenPurchaseOrdersInput,
    GetSupplierStatusInput,
    MAX_LIMIT,
    ItemListOutput,
    SupplierStatusOutput,
)
from app.services.supplier_service import SupplierService


def get_supplier_status(
    supplier_id: int | None = None, supplier_name: str | None = None
) -> SupplierStatusOutput:
    """Return a supplier's reliability profile.

    Identify the supplier by ``supplier_id`` or a partial ``supplier_name``.
    The profile includes on-time delivery rate, average delay days, and open
    / late purchase-order counts.
    """
    params = GetSupplierStatusInput(supplier_id=supplier_id, supplier_name=supplier_name)
    with get_readonly_session() as session:
        profile = SupplierService(session).supplier_profile(
            supplier_id=params.supplier_id, supplier_name=params.supplier_name
        )
    return SupplierStatusOutput(
        supplier=profile,
        summary=(
            f"{profile['name']}: {profile['on_time_delivery_rate'] * 100:.0f}% on time, "
            f"avg delay {profile['avg_delay_days']} day(s), "
            f"{profile['late_po_count']} late PO(s)."
        ),
    )


def _resolve_supplier_id(session, supplier: str | None) -> int | None:
    if not supplier:
        return None
    repo = SupplierRepository(session)
    if supplier.isdigit():
        found = repo.get_supplier(int(supplier))
        if found is None:
            raise DataNotFoundError(f"Unknown supplier id {supplier}.")
        return found["id"]
    matches = repo.find_suppliers_by_name(supplier)
    if not matches:
        raise DataNotFoundError(f"No supplier matching '{supplier}'.")
    return matches[0]["id"]


def get_open_purchase_orders(
    sku: str | None = None, supplier: str | None = None, late_only: bool = False
) -> ItemListOutput:
    """List purchase orders that are still open, or (``late_only=true``) that
    were received after their promised date.

    Filter by ``sku`` and/or ``supplier`` (id or partial name).
    """
    params = GetOpenPurchaseOrdersInput(sku=sku, supplier=supplier, late_only=late_only)
    with get_readonly_session() as session:
        supplier_id = _resolve_supplier_id(session, params.supplier)
        svc = SupplierService(session)
        if params.late_only:
            items = svc.late_purchase_orders(sku=params.sku, supplier_id=supplier_id)
        else:
            items = svc.open_purchase_orders(sku=params.sku, supplier_id=supplier_id)
    items = items[:MAX_LIMIT]
    kind = "late" if params.late_only else "open"
    return ItemListOutput(
        items=items, count=len(items), summary=f"{len(items)} {kind} purchase order(s)."
    )
