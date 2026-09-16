"""Forecasting service: stockout risk, reorder arithmetic, expiry risk, boundaries."""
from __future__ import annotations

import datetime as dt

import pytest

from app.config import get_settings
from app.core.exceptions import DataNotFoundError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.db.repositories.supplier_repo import SupplierRepository
from app.services.forecasting_service import ForecastingService

TODAY = dt.date.today()


def _settings_naive():
    return get_settings().model_copy(update={"forecast_backend": "seasonal_naive"})


def _seed_sku(session, sku, daily_units, *, on_hand, reorder_point=50, lead_time=5,
              is_perishable=False, shelf_life=None):
    sup = SupplierRepository(session).add_supplier(
        name=f"S-{sku}", lead_time_days=lead_time, reliability_score=0.9,
        contact_email="s@x.test", contract_ref="C",
    )
    inv = InventoryRepository(session)
    inv.add_product(
        sku=sku, name=sku, category="X", aisle="Beverages", unit_cost=2.0, unit_price=5.0,
        is_perishable=is_perishable, shelf_life_days=shelf_life, reorder_point=reorder_point,
        safety_stock=reorder_point // 2, supplier_id=sup["id"],
    )
    rows = [
        {
            "txn_id": f"T-{sku}-{i}", "ts": dt.datetime.combine(TODAY - dt.timedelta(days=i + 1), dt.time(12)),
            "sku": sku, "qty": q, "unit_price": 5.0, "discount": 0.0, "register_id": "R1",
        }
        for i, q in enumerate(reversed(daily_units))
    ]
    SalesRepository(session).bulk_add_transactions(rows)
    inv.add_stock_level(sku=sku, snapshot_date=TODAY - dt.timedelta(days=1),
                        shelf_qty=min(on_hand, reorder_point), backroom_qty=max(0, on_hand - reorder_point),
                        on_hand_qty=on_hand)
    session.flush()


def test_stockout_risk_on_a_known_trajectory(db_session):
    _seed_sku(db_session, "SKU-FS1", [10] * 28, on_hand=25, reorder_point=50)
    svc = ForecastingService(db_session, _settings_naive())
    risk = svc.stockout_risk("SKU-FS1")
    # ~10 units/day, 25 on hand -> runs out in ~2-3 days
    assert risk["risk_level"] == "critical"
    assert isinstance(risk["days_of_cover"], int) and risk["days_of_cover"] <= 3
    assert risk["projected_stockout_date"] is not None
    assert risk["assumptions"]["no_replenishment_assumed"] is True


def test_stockout_risk_levels_span_the_boundaries(db_session):
    for i, (on_hand, expected) in enumerate(
        [(15, "critical"), (45, "high"), (100, "medium"), (400, "low")]
    ):
        _seed_sku(db_session, f"SKU-B{i}", [10] * 28, on_hand=on_hand, reorder_point=999)
        risk = ForecastingService(db_session, _settings_naive()).stockout_risk(f"SKU-B{i}")
        assert risk["risk_level"] == expected


def test_reorder_point_arithmetic(db_session):
    _seed_sku(db_session, "SKU-FS2", [10] * 28, on_hand=30, reorder_point=40, lead_time=5)
    rec = ForecastingService(db_session, _settings_naive()).reorder_point("SKU-FS2")
    # ~10/day over a 5-day lead time -> ~50 lead-time demand + safety stock
    assert rec["forecast_lead_time_demand"] == pytest.approx(50, abs=15)
    assert rec["recommended_reorder_point"] > rec["forecast_lead_time_demand"] - 1
    assert rec["current_reorder_point"] == 40
    assert rec["delta"] == rec["recommended_reorder_point"] - 40
    assert rec["recommended_order_qty"] >= 0


def test_expiry_risk_against_a_slow_moving_batch(db_session):
    _seed_sku(db_session, "SKU-FS3", [1] * 28, on_hand=5, reorder_point=20,
              is_perishable=True, shelf_life=14)
    InventoryRepository(db_session).add_stock_batch(
        sku="SKU-FS3", batch_no="B-1", received_date=TODAY - dt.timedelta(days=10),
        expiry_date=TODAY + dt.timedelta(days=4), qty_received=100, qty_remaining=100,
    )
    db_session.flush()
    risk = ForecastingService(db_session, _settings_naive()).expiry_risk("SKU-FS3")
    assert len(risk) == 1
    assert risk[0]["projected_units_expiring"] > 50  # ~1/day can't clear 100 in 4 days
    assert risk[0]["estimated_write_off_value"] == pytest.approx(risk[0]["projected_units_expiring"] * 2.0, abs=2)
    assert risk[0]["recommended_action"] == "markdown"


def test_expiry_risk_empty_for_non_perishable(db_session):
    _seed_sku(db_session, "SKU-FS4", [5] * 20, on_hand=50, reorder_point=20)
    assert ForecastingService(db_session, _settings_naive()).expiry_risk("SKU-FS4") == []


def test_zero_sales_sku_does_not_divide_by_zero(db_session):
    _seed_sku(db_session, "SKU-FS5", [0] * 20, on_hand=10, reorder_point=20)
    svc = ForecastingService(db_session, _settings_naive())
    fc = svc.forecast("SKU-FS5")
    assert fc["avg_daily_forecast"] == 0.0
    assert "no recorded sales" in " ".join(fc["warnings"]).lower()
    assert svc.stockout_risk("SKU-FS5")["risk_level"] == "low"  # never projected to stock out


def test_unknown_sku_raises_data_not_found(db_session):
    with pytest.raises(DataNotFoundError):
        ForecastingService(db_session, _settings_naive()).forecast("SKU-NOPE")
