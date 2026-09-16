"""Service-layer tests against a small hand-computed fixture dataset."""
from __future__ import annotations

import datetime as dt

import pytest

from app.core.exceptions import DataNotFoundError
from app.db.repositories.finance_repo import FinanceRepository
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.db.repositories.supplier_repo import SupplierRepository
from app.services.finance_service import FinanceService
from app.services.inventory_service import InventoryService
from app.services.sales_service import SalesService
from app.services.supplier_service import SupplierService

TODAY = dt.date.today()


@pytest.fixture()
def data(db_session):
    supplier_repo = SupplierRepository(db_session)
    inv_repo = InventoryRepository(db_session)
    sales_repo = SalesRepository(db_session)
    finance_repo = FinanceRepository(db_session)

    supplier = supplier_repo.add_supplier(
        name="Acme Distribution",
        lead_time_days=5,
        reliability_score=0.9,
        contact_email="acme@example.test",
        contract_ref="CTR-ACME",
    )
    widget = inv_repo.add_product(
        sku="SKU-W",
        name="Widget",
        category="Widgets",
        aisle="TestAisle",
        unit_cost=2.0,
        unit_price=5.0,
        is_perishable=False,
        shelf_life_days=None,
        reorder_point=20,
        safety_stock=10,
        supplier_id=supplier["id"],
    )
    gadget = inv_repo.add_product(
        sku="SKU-G",
        name="Gadget",
        category="Gadgets",
        aisle="TestAisle",
        unit_cost=4.0,
        unit_price=10.0,
        is_perishable=True,
        shelf_life_days=10,
        reorder_point=15,
        safety_stock=5,
        supplier_id=supplier["id"],
    )

    # Widget: 10 units/day for 3 days ending today. Gadget: no sales at all.
    for offset in range(3):
        day = TODAY - dt.timedelta(days=offset)
        sales_repo.add_transaction(
            txn_id=f"T-W-{offset}",
            ts=dt.datetime.combine(day, dt.time(12, 0)),
            sku="SKU-W",
            qty=10,
            unit_price=5.0,
            discount=0.0,
            register_id="REG-1",
        )

    inv_repo.add_stock_level(
        sku="SKU-W", snapshot_date=TODAY, shelf_qty=8, backroom_qty=4, on_hand_qty=12
    )
    inv_repo.add_stock_level(
        sku="SKU-G", snapshot_date=TODAY, shelf_qty=0, backroom_qty=0, on_hand_qty=0
    )
    inv_repo.add_stock_batch(
        sku="SKU-G",
        batch_no="G-1",
        received_date=TODAY - dt.timedelta(days=8),
        expiry_date=TODAY + dt.timedelta(days=2),
        qty_received=40,
        qty_remaining=25,
    )

    supplier_repo.add_purchase_order(
        po_id="PO-ON-TIME",
        supplier_id=supplier["id"],
        sku="SKU-W",
        qty_ordered=40,
        order_date=TODAY - dt.timedelta(days=20),
        promised_date=TODAY - dt.timedelta(days=15),
        received_date=TODAY - dt.timedelta(days=15),
        qty_received=40,
        status="received",
    )
    supplier_repo.add_purchase_order(
        po_id="PO-LATE",
        supplier_id=supplier["id"],
        sku="SKU-W",
        qty_ordered=40,
        order_date=TODAY - dt.timedelta(days=12),
        promised_date=TODAY - dt.timedelta(days=8),
        received_date=TODAY - dt.timedelta(days=4),
        qty_received=40,
        status="late",
    )
    supplier_repo.add_purchase_order(
        po_id="PO-OPEN",
        supplier_id=supplier["id"],
        sku="SKU-G",
        qty_ordered=30,
        order_date=TODAY - dt.timedelta(days=2),
        promised_date=TODAY + dt.timedelta(days=3),
        received_date=None,
        qty_received=None,
        status="open",
    )

    finance_repo.add_shrinkage_event(
        sku="SKU-W", event_date=TODAY - dt.timedelta(days=5), qty=6, reason="theft", notes=None
    )
    db_session.flush()
    return {"session": db_session, "supplier": supplier}


# ------------------------------------------------------------------- inventory
def test_current_stock_and_unknown_sku(data):
    svc = InventoryService(data["session"])
    stock = svc.current_stock("SKU-W")
    assert stock["on_hand_qty"] == 12
    assert stock["below_reorder_point"] is True  # 12 <= 20

    with pytest.raises(DataNotFoundError):
        svc.current_stock("SKU-MISSING")


def test_low_stock_and_days_of_cover(data):
    svc = InventoryService(data["session"])
    low = svc.low_stock(aisle="TestAisle", limit=50)
    skus = {row["sku"] for row in low}
    assert skus == {"SKU-W", "SKU-G"}  # 12<=20 and 0<=15

    cover = svc.days_of_cover("SKU-W", window_days=3)
    assert cover["avg_daily_units"] == pytest.approx(10.0)
    assert cover["days_of_cover"] == pytest.approx(1.2)  # 12 / 10

    # SKU with no sales -> undefined cover, not a crash
    assert svc.days_of_cover("SKU-G", window_days=30)["days_of_cover"] is None


def test_expiring_batches_and_discrepancy(data):
    svc = InventoryService(data["session"])
    expiring = svc.expiring_batches(days_ahead=7, aisle="TestAisle")
    assert [b["batch_no"] for b in expiring] == ["G-1"]
    assert expiring[0]["days_until_expiry"] == 2

    assert svc.expiring_batches(days_ahead=1, aisle="TestAisle") == []  # boundary
    assert svc.shelf_backroom_discrepancy("SKU-W") == []  # 8 + 4 == 12


# ---------------------------------------------------------------------- sales
def test_sales_history_daily_weekly_and_empty(data):
    svc = SalesService(data["session"])
    daily = svc.sales_history("SKU-W", TODAY - dt.timedelta(days=2), TODAY, "daily")
    assert daily["total_units"] == 30
    assert daily["total_revenue"] == pytest.approx(150.0)
    assert len(daily["points"]) == 3

    weekly = svc.sales_history("SKU-W", TODAY - dt.timedelta(days=6), TODAY, "weekly")
    assert weekly["total_units"] == 30

    single = svc.sales_history("SKU-W", TODAY, TODAY, "daily")
    assert single["total_units"] == 10

    empty = svc.sales_history("SKU-G", TODAY - dt.timedelta(days=10), TODAY, "daily")
    assert empty["total_units"] == 0 and empty["points"] == []


def test_velocity_and_demand_shift(data):
    svc = SalesService(data["session"])
    assert svc.velocity("SKU-W", window_days=3)["avg_daily_units"] == pytest.approx(10.0)

    shift = svc.demand_shift("SKU-W", recent_days=3, baseline_days=30)
    assert shift["baseline_avg_daily"] == 0.0
    assert shift["ratio"] is None  # no baseline sales -> undefined, not a crash


def test_top_sellers(data):
    svc = SalesService(data["session"])
    top = svc.top_sellers(aisle="TestAisle", days=7, limit=10)
    assert top[0]["sku"] == "SKU-W"
    assert top[0]["units"] == 30
    assert all(row["sku"] != "SKU-G" for row in top)  # no sales -> excluded


# -------------------------------------------------------------------- finance
def test_sku_and_aisle_margin(data):
    svc = FinanceService(data["session"])
    sku = svc.sku_margin("SKU-W", days=7)
    assert sku["revenue"] == pytest.approx(150.0)
    assert sku["cogs"] == pytest.approx(60.0)  # 30 units * 2.0
    assert sku["gross_margin"] == pytest.approx(90.0)
    assert sku["gross_margin_pct"] == pytest.approx(60.0)

    aisle = svc.aisle_margin("TestAisle", days=7)
    assert aisle["revenue"] == pytest.approx(150.0)
    assert aisle["sku_count"] == 2

    with pytest.raises(DataNotFoundError):
        svc.aisle_margin("NoSuchAisle")


def test_shrinkage_valuation_and_erosion(data):
    svc = FinanceService(data["session"])
    val = svc.shrinkage_valuation(sku="SKU-W", days=90)
    assert val["total_qty"] == 6
    assert val["total_cost"] == pytest.approx(12.0)  # 6 * 2.0
    assert val["by_reason"]["theft"]["qty"] == 6

    ranking = {row["sku"]: row for row in svc.margin_erosion_ranking(days=90, limit=500)}
    assert "SKU-W" in ranking  # only SKUs with shrinkage cost appear
    assert ranking["SKU-W"]["shrinkage_cost"] == pytest.approx(12.0)
    assert "SKU-G" not in ranking  # no shrinkage -> excluded

    assert svc.shrinkage_valuation(sku="SKU-G", days=90)["total_cost"] == 0.0  # boundary: none


# ------------------------------------------------------------------- supplier
def test_supplier_profile_metrics(data):
    svc = SupplierService(data["session"])
    sid = data["supplier"]["id"]

    assert svc.on_time_delivery_rate(sid) == pytest.approx(0.5)  # 1 of 2 received on time
    assert svc.average_delay_days(sid) == pytest.approx(4.0)  # PO-LATE: 4 days late

    assert [po["po_id"] for po in svc.open_purchase_orders(supplier_id=sid)] == ["PO-OPEN"]
    late = svc.late_purchase_orders(supplier_id=sid)
    assert [po["po_id"] for po in late] == ["PO-LATE"]
    assert late[0]["delay_days"] == 4

    profile = svc.supplier_profile(supplier_name="Acme")
    assert profile["late_po_count"] == 1 and profile["open_po_count"] == 1

    with pytest.raises(DataNotFoundError):
        svc.supplier_profile(supplier_name="Nonexistent Vendor")


def test_sku_supplier_reliability(data):
    svc = SupplierService(data["session"])
    rel = svc.sku_supplier_reliability("SKU-W")
    assert rel["supplier_name"] == "Acme Distribution"
    assert rel["on_time_delivery_rate"] == pytest.approx(0.5)
