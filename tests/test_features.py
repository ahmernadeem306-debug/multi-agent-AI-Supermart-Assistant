"""Feature-engineering correctness and an explicit no-target-leakage assertion."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.ml import features as F


def _panel(units: list[int], sku: str = "SKU-X", aisle: str = "Beverages") -> pd.DataFrame:
    start = dt.date(2025, 1, 1)
    rows = [
        {"date": pd.Timestamp(start + dt.timedelta(days=i)), "sku": sku, "aisle": aisle, "units": u}
        for i, u in enumerate(units)
    ]
    return pd.DataFrame(rows)


def test_daily_units_frame_fills_gaps():
    rows = [
        {"date": "2025-01-01", "sku": "A", "units": 3, "aisle": "Snacks"},
        {"date": "2025-01-04", "sku": "A", "units": 5, "aisle": "Snacks"},
    ]
    panel = F.daily_units_frame(rows)
    assert list(panel["date"].dt.day) == [1, 2, 3, 4]
    assert list(panel["units"]) == [3, 0, 0, 5]


def test_lag_and_rolling_values_are_correct():
    units = list(range(1, 41))  # 1..40
    feats = F.build_features(_panel(units))
    row = feats.iloc[35]  # 36th day, units == 36
    assert row["lag_1"] == 35
    assert row["lag_7"] == 29
    assert row["rmean_7"] == np.mean(units[28:35])  # days 29..35, i.e. before day 36
    assert row["is_weekend"] in (0, 1)
    assert row["day_of_week"] == feats.iloc[35]["date"].dayofweek


def test_promo_flag_fires_on_a_demand_step_up():
    units = [5] * 40 + [20] * 10  # sharp step-up
    feats = F.build_features(_panel(units))
    assert feats.iloc[30]["promo_flag"] == 0
    assert feats.iloc[-1]["promo_flag"] == 1


def test_days_since_stockout_uses_only_prior_dates():
    units = [4] * 20
    stockout_day = dt.date(2025, 1, 10)
    feats = F.build_features(_panel(units), stockouts={"SKU-X": [stockout_day]})
    before = feats[feats["date"] < pd.Timestamp(stockout_day)]
    after = feats[feats["date"] > pd.Timestamp(stockout_day)]
    assert (before["days_since_stockout"] == F._NO_STOCKOUT_SENTINEL).all()
    assert after.iloc[0]["days_since_stockout"] == 1


def test_no_target_leakage():
    """Changing the target on the last day must not change any feature row —
    every feature is derived strictly from earlier observations."""
    units = list(np.random.default_rng(0).integers(1, 20, size=45))
    base = F.build_features(_panel(list(units)))

    tampered_units = list(units)
    tampered_units[-1] = 9999  # blow up the final target
    tampered = F.build_features(_panel(tampered_units))

    feature_cols = [c for c in F.NUMERIC_FEATURES]
    pd.testing.assert_frame_equal(
        base[feature_cols].reset_index(drop=True),
        tampered[feature_cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_training_matrix_drops_rows_without_full_lags():
    feats = F.build_features(_panel(list(range(1, 60))))
    usable = F.training_matrix(feats)
    assert len(usable) < len(feats)
    assert usable[F.LAG_COLUMNS].notna().all().all()
