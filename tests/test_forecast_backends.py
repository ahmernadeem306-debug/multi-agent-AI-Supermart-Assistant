"""Forecast backends: seasonal-naive correctness and XGBoost train/predict."""
from __future__ import annotations

import datetime as dt

import pytest

from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.ml.backends import SeasonalNaiveBackend, XGBoostBackend

TODAY = dt.date.today()


def _make_sku(session, sku: str, daily: list[int], *, aisle: str = "Beverages") -> None:
    from app.db.repositories.supplier_repo import SupplierRepository

    inv, sales = InventoryRepository(session), SalesRepository(session)
    if inv.get_product(sku) is None:
        sup = SupplierRepository(session).add_supplier(
            name=f"S-{sku}", lead_time_days=5, reliability_score=0.9,
            contact_email="s@x.test", contract_ref="C",
        )
        inv.add_product(
            sku=sku, name=sku, category="Water", aisle=aisle, unit_cost=1.0, unit_price=2.0,
            is_perishable=False, shelf_life_days=None, reorder_point=20, safety_stock=8,
            supplier_id=sup["id"],
        )
    rows = []
    for offset, qty in enumerate(reversed(daily)):
        day = TODAY - dt.timedelta(days=offset + 1)
        rows.append(
            {
                "txn_id": f"T-{sku}-{offset}",
                "ts": dt.datetime.combine(day, dt.time(12, 0)),
                "sku": sku,
                "qty": qty,
                "unit_price": 2.0,
                "discount": 0.0,
                "register_id": "R1",
            }
        )
    sales.bulk_add_transactions(rows)
    inv.add_stock_level(sku=sku, snapshot_date=TODAY - dt.timedelta(days=1),
                        shelf_qty=10, backroom_qty=10, on_hand_qty=20)
    session.flush()


def test_seasonal_naive_shape_and_horizon(db_session):
    _make_sku(db_session, "SKU-SN1", [10] * 28)
    backend = SeasonalNaiveBackend(db_session)
    for horizon in (1, 7, 30):
        preds = backend.predict("SKU-SN1", horizon)
        assert len(preds) == horizon
        assert preds[0].date == TODAY + dt.timedelta(days=1)
        assert all(p.predicted_units >= 0 for p in preds)
        assert all(p.lower <= p.predicted_units <= p.upper for p in preds)


def test_seasonal_naive_learns_weekday_pattern(db_session):
    # 4 weeks: weekends (Sat/Sun) ~30, weekdays ~5
    daily = []
    for i in range(28):
        d = TODAY - dt.timedelta(days=28 - i)
        daily.append(30 if d.weekday() >= 5 else 5)
    _make_sku(db_session, "SKU-SN2", daily)
    preds = {p.date.weekday(): p.predicted_units for p in SeasonalNaiveBackend(db_session).predict("SKU-SN2", 14)}
    weekend = max(preds[5], preds[6])
    weekday = min(preds[1], preds[2], preds[3])
    assert weekend > weekday * 2


def test_zero_sales_sku_returns_zeros(db_session):
    _make_sku(db_session, "SKU-SN0", [0] * 20)
    preds = SeasonalNaiveBackend(db_session).predict("SKU-SN0", 10)
    assert len(preds) == 10
    assert all(p.predicted_units == 0 for p in preds)


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory, seeded_db):
    from app.db.base import get_session_factory
    from app.ml.train import train_global_model

    model_dir = tmp_path_factory.mktemp("models")
    session = get_session_factory()()
    try:
        train_global_model(session, valid_days=14, model_dir=str(model_dir))
    finally:
        session.close()
    return str(model_dir)


def test_xgboost_trains_and_predicts(trained_model_dir, read_session):
    backend = XGBoostBackend(read_session, trained_model_dir)
    assert backend.name == "xgboost"
    for horizon in (1, 14, 30):
        preds = backend.predict("SKU-1035", horizon)
        assert len(preds) == horizon
        assert all(p.predicted_units >= 0 for p in preds)
        assert all(p.predicted_units == p.predicted_units for p in preds)  # not NaN
        assert preds[0].date == TODAY + dt.timedelta(days=1)


def test_xgboost_metadata_reports_baseline_comparison(trained_model_dir):
    import json
    from pathlib import Path

    meta = json.loads((Path(trained_model_dir) / "forecast_xgb.meta.json").read_text())
    assert "xgboost" in meta["metrics"] and "seasonal_naive" in meta["metrics"]
    assert meta["metrics"]["xgboost"]["mae"] is not None
    assert "verdict" in meta
