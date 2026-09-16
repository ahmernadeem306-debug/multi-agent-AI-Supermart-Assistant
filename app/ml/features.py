"""Feature engineering for per-SKU daily demand forecasting.

Every feature is computed from information strictly *before* the row's date:
lags and rolling statistics are shifted by at least one day, calendar
features come from the date itself, and the promotion flag and
days-since-stockout look only at prior observations. ``tests/test_features.py``
asserts there is no target leakage.
"""
from __future__ import annotations

import bisect
import datetime as dt

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 14, 28)
_NO_STOCKOUT_SENTINEL = 999
_PROMO_RATIO = 1.5

CATEGORICAL_COLUMNS = ["sku", "aisle"]
LAG_COLUMNS = [f"lag_{k}" for k in LAGS]
ROLL_COLUMNS = [f"rmean_{w}" for w in ROLL_WINDOWS] + [f"rstd_{w}" for w in ROLL_WINDOWS]
CALENDAR_COLUMNS = ["day_of_week", "week_of_year", "month", "is_weekend"]
EXTRA_COLUMNS = ["days_since_stockout", "promo_flag"]
NUMERIC_FEATURES = LAG_COLUMNS + ROLL_COLUMNS + CALENDAR_COLUMNS + EXTRA_COLUMNS
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_FEATURES
TARGET = "units"


def daily_units_frame(
    rows: list[dict], start: dt.date | None = None, end: dt.date | None = None
) -> pd.DataFrame:
    """Turn ``[{date, sku, units, aisle}]`` into a gap-free daily panel per SKU."""
    if not rows:
        return pd.DataFrame(columns=["date", "sku", "aisle", "units"])
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.groupby(["sku", "date"], as_index=False).agg(
        units=("units", "sum"), aisle=("aisle", "first")
    )
    lo = pd.Timestamp(start) if start else frame["date"].min()
    hi = pd.Timestamp(end) if end else frame["date"].max()
    full_index = pd.date_range(lo, hi, freq="D")

    panels = []
    for sku, grp in frame.groupby("sku"):
        aisle = grp["aisle"].dropna().iloc[0] if grp["aisle"].notna().any() else None
        reindexed = (
            grp.set_index("date")["units"]
            .reindex(full_index, fill_value=0)
            .rename_axis("date")
            .rename("units")
        )
        panel = reindexed.reset_index()
        panel["sku"] = sku
        panel["aisle"] = aisle
        panels.append(panel)
    return pd.concat(panels, ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["day_of_week"] = d.dayofweek
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["month"] = d.month
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    return df


def _add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("sku")["units"]
    for k in LAGS:
        df[f"lag_{k}"] = grp.shift(k)
    shifted = grp.shift(1)
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean()
        df[f"rstd_{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).std()
    return df


def _add_promo_flag(df: pd.DataFrame) -> pd.DataFrame:
    short = df["rmean_7"]
    long = df[f"rmean_{ROLL_WINDOWS[-1]}"]
    df["promo_flag"] = ((long > 0) & (short >= _PROMO_RATIO * long)).astype(int)
    return df


def _add_days_since_stockout(
    df: pd.DataFrame, stockouts: dict[str, list[dt.date]] | None
) -> pd.DataFrame:
    stockouts = stockouts or {}
    values = np.full(len(df), _NO_STOCKOUT_SENTINEL, dtype=float)
    dates_col = df["date"].tolist()
    skus_col = df["sku"].tolist()
    per_sku = {
        sku: sorted(pd.Timestamp(d) for d in dates) for sku, dates in stockouts.items() if dates
    }
    for i, (sku, row_date) in enumerate(zip(skus_col, dates_col)):
        sorted_dates = per_sku.get(sku)
        if not sorted_dates:
            continue
        pos = bisect.bisect_left(sorted_dates, row_date)  # strictly before -> no leakage
        if pos > 0:
            values[i] = (row_date - sorted_dates[pos - 1]).days
    df["days_since_stockout"] = values
    return df


def build_features(
    panel: pd.DataFrame, *, stockouts: dict[str, list[dt.date]] | None = None
) -> pd.DataFrame:
    """Add every feature column to a gap-free daily panel (see :func:`daily_units_frame`)."""
    if panel.empty:
        return pd.DataFrame(columns=["date", *FEATURE_COLUMNS, TARGET])
    df = panel.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = _add_calendar(df)
    df = _add_lags_and_rolls(df)
    df = _add_promo_flag(df)
    df = _add_days_since_stockout(df, stockouts)
    return df


def training_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for supervised training (all lag/rolling features present)."""
    required = LAG_COLUMNS + [f"rmean_{w}" for w in ROLL_WINDOWS]
    return features.dropna(subset=required).reset_index(drop=True)
