from __future__ import annotations

import datetime as dt

import pytest

from app.db.repositories.finance_repo import FinanceRepository
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.db.repositories.supplier_repo import SupplierRepository


@pytest.fixture()
def fixture_data(db_session):
    supplier_repo = SupplierRepository(db_session)
    inv_repo = InventoryRepository(db_session)
    sales_repo = SalesRepository(db_session)
    finance_repo = FinanceRepository(db_session)

    supplier = supplier_repo.add_supplier(
        name="Test Supplier",
        lead_time_days=5,
        reliability_score=0.9,
        contact_email="test@example.test",
        contract_ref="CTR-TEST",
    )
    product = inv_repo.add_product(
        sku="SKU-TEST-1",
        name="Test Widget",
        category="Widgets",
        aisle="Household",
        unit_cost=2.0,
        unit_price=4.0,
        is_perishable=False,
        shelf_life_days=None,
        reorder_point=10,
        safety_stock=5,
        supplier_id=supplier["id"],
    )
    return {
        "supplier_repo": supplier_repo,
        "inv_repo": inv_repo,
        "sales_repo": sales_repo,
        "finance_repo": finance_repo,
        "supplier": supplier,
        "product": product,
    }


def test_inventory_repo_get_and_list_product(fixture_data):
    inv_repo = fixture_data["inv_repo"]
    product = fixture_data["product"]

    fetched = inv_repo.get_product(product["sku"])
    assert fetched is not None
    assert fetched["sku"] == "SKU-TEST-1"
    assert fetched["unit_price"] == 4.0

    listed = inv_repo.list_products(aisle="Household")
    assert any(p["sku"] == "SKU-TEST-1" for p in listed)

    assert inv_repo.get_product("SKU-DOES-NOT-EXIST") is None


def test_inventory_repo_stock_levels_and_batches(fixture_data):
    inv_repo = fixture_data["inv_repo"]
    sku = fixture_data["product"]["sku"]
    today = dt.date.today()

    inv_repo.add_stock_level(sku=sku, snapshot_date=today, shelf_qty=8, backroom_qty=12, on_hand_qty=20)
    latest = inv_repo.get_latest_stock_level(sku)
    assert latest["on_hand_qty"] == 20

    inv_repo.add_stock_batch(
        sku=sku,
        batch_no="B1",
        received_date=today,
        expiry_date=today + dt.timedelta(days=3),
        qty_received=50,
        qty_remaining=30,
    )
    expiring = inv_repo.list_expiring_batches(as_of=today, days_ahead=7)
    assert any(b["batch_no"] == "B1" for b in expiring)

    not_expiring = inv_repo.list_expiring_batches(as_of=today, days_ahead=1)
    assert all(b["batch_no"] != "far-future" for b in not_expiring)


def test_sales_repo_total_units_sold(fixture_data):
    sales_repo = fixture_data["sales_repo"]
    sku = fixture_data["product"]["sku"]
    start = dt.datetime(2026, 1, 1)

    sales_repo.add_transaction(
        txn_id="T1", ts=start, sku=sku, qty=3, unit_price=4.0, discount=0.0, register_id="REG-1"
    )
    sales_repo.add_transaction(
        txn_id="T2", ts=start + dt.timedelta(days=1), sku=sku, qty=5, unit_price=4.0, discount=0.0, register_id="REG-1"
    )

    total = sales_repo.total_units_sold(sku, start, start + dt.timedelta(days=5))
    assert total == 8

    transactions = sales_repo.list_transactions(sku, start, start + dt.timedelta(days=5))
    assert len(transactions) == 2


def test_supplier_repo_purchase_orders(fixture_data):
    supplier_repo = fixture_data["supplier_repo"]
    supplier = fixture_data["supplier"]
    sku = fixture_data["product"]["sku"]

    order_date = dt.date(2026, 1, 1)
    promised = order_date + dt.timedelta(days=5)
    late_received = promised + dt.timedelta(days=9)

    supplier_repo.add_purchase_order(
        po_id="PO-1",
        supplier_id=supplier["id"],
        sku=sku,
        qty_ordered=20,
        order_date=order_date,
        promised_date=promised,
        received_date=late_received,
        qty_received=20,
        status="late",
    )
    supplier_repo.add_purchase_order(
        po_id="PO-2",
        supplier_id=supplier["id"],
        sku=sku,
        qty_ordered=20,
        order_date=order_date,
        promised_date=promised,
        received_date=promised,
        qty_received=20,
        status="received",
    )

    all_orders = supplier_repo.list_purchase_orders(sku=sku)
    assert len(all_orders) == 2

    late_orders = supplier_repo.list_late_deliveries(sku=sku)
    assert len(late_orders) == 1
    assert late_orders[0]["po_id"] == "PO-1"


def test_finance_repo_shrinkage_events(fixture_data):
    finance_repo = fixture_data["finance_repo"]
    sku = fixture_data["product"]["sku"]
    event_date = dt.date(2026, 2, 1)

    finance_repo.add_shrinkage_event(sku=sku, event_date=event_date, qty=4, reason="theft", notes=None)
    finance_repo.add_shrinkage_event(
        sku=sku, event_date=event_date + dt.timedelta(days=1), qty=2, reason="damage", notes=None
    )

    events = finance_repo.list_shrinkage_events(sku=sku)
    assert len(events) == 2

    theft_only = [e for e in events if e["reason"] == "theft"]
    assert len(theft_only) == 1
