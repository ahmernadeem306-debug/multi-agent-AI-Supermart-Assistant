"""Forecast backends.

``SeasonalNaiveBackend`` (day-of-week average of the last four weeks) is the
always-available baseline. ``XGBoostBackend`` loads a trained global model
from ``MODEL_DIR`` and forecasts recursively. Both implement the
``ForecastBackend`` protocol: ``name`` and ``predict(sku, horizon_days)``.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.core.exceptions import DataNotFoundError, ForecastError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.logging_config import get_logger
from app.ml import features as F

logger = get_logger(__name__)

_INTERVAL_Z = 1.28  # ~80% prediction interval
_RECENT_WINDOW = 28
_MODEL_BASENAME = "forecast_xgb"
_bundle_cache: dict[str, "dict"] = {}


class DailyForecast(BaseModel):
    date: dt.date
    predicted_units: float
    lower: float
    upper: float


class ForecastBackend(Protocol):
    name: str

    def predict(self, sku: str, horizon_days: int) -> list[DailyForecast]:
        """Forecast daily units for ``horizon_days`` starting tomorrow."""


# --------------------------------------------------------------------- helpers
def _load_history_rows(session, sku: str) -> tuple[pd.DataFrame, dict]:
    inv = InventoryRepository(session)
    product = inv.get_product(sku)
    if product is None:
        raise DataNotFoundError(f"Unknown SKU '{sku}'.")
    end = dt.date.today()
    start = end - dt.timedelta(days=730)
    txns = SalesRepository(session).list_transactions(
        sku, dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)
    )
    rows = [
        {"date": t["ts"], "sku": sku, "units": t["qty"], "aisle": product["aisle"]}
        for t in txns
    ]
    panel = F.daily_units_frame(rows, start=start, end=end - dt.timedelta(days=1))
    return panel, product


def _last_stockout_date(session, sku: str) -> dt.date | None:
    inv = InventoryRepository(session)
    end = dt.date.today()
    start = end - dt.timedelta(days=730)
    zero_days = [
        lvl["snapshot_date"]
        for lvl in inv.list_stock_levels(sku, start, end)
        if lvl["on_hand_qty"] == 0
    ]
    return max(zero_days) if zero_days else None


def _future_dates(horizon_days: int) -> list[dt.date]:
    today = dt.date.today()
    return [today + dt.timedelta(days=i) for i in range(1, horizon_days + 1)]


def _flat_forecast(dates: list[dt.date], value: float, std: float) -> list[DailyForecast]:
    spread = _INTERVAL_Z * max(std, 1.0)
    return [
        DailyForecast(
            date=d,
            predicted_units=round(max(value, 0.0), 2),
            lower=round(max(value - spread, 0.0), 2),
            upper=round(value + spread, 2),
        )
        for d in dates
    ]


# ------------------------------------------------------------ seasonal naive
class SeasonalNaiveBackend:
    name = "seasonal_naive"

    def __init__(self, session) -> None:
        self._session = session

    def _recent(self, sku: str) -> tuple[pd.Series, dict]:
        panel, product = _load_history_rows(self._session, sku)
        series = panel.set_index("date")["units"].astype(float)
        return series.tail(_RECENT_WINDOW), product

    def daily_std(self, sku: str) -> float:
        recent, _ = self._recent(sku)
        std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
        return round(std, 3)

    def history(self, sku: str) -> pd.Series:
        panel, _ = _load_history_rows(self._session, sku)
        return panel.set_index("date")["units"].astype(float)

    def predict(self, sku: str, horizon_days: int) -> list[DailyForecast]:
        recent, _ = self._recent(sku)
        dates = _future_dates(horizon_days)
        if recent.empty or recent.sum() == 0:
            return _flat_forecast(dates, 0.0, 1.0)

        by_weekday_mean = recent.groupby(recent.index.dayofweek).mean()
        by_weekday_std = recent.groupby(recent.index.dayofweek).std(ddof=0).fillna(0.0)
        overall_mean = float(recent.mean())
        overall_std = float(recent.std(ddof=0))

        out: list[DailyForecast] = []
        for d in dates:
            wd = d.weekday()
            value = float(by_weekday_mean.get(wd, overall_mean))
            std = float(by_weekday_std.get(wd, overall_std))
            spread = _INTERVAL_Z * max(std, 1.0)
            out.append(
                DailyForecast(
                    date=d,
                    predicted_units=round(max(value, 0.0), 2),
                    lower=round(max(value - spread, 0.0), 2),
                    upper=round(value + spread, 2),
                )
            )
        return out


# ------------------------------------------------------------------ xgboost
def _bundle_path(model_dir: str) -> Path:
    return Path(model_dir) / f"{_MODEL_BASENAME}.joblib"


def load_bundle(model_dir: str) -> dict:
    """Load (and cache by path+mtime) the trained model bundle, or raise."""
    path = _bundle_path(model_dir)
    if not path.exists():
        raise ForecastError(
            f"No trained forecast model at {path}. Run `python scripts/train_forecast.py` "
            "or set FORECAST_BACKEND=seasonal_naive."
        )
    key = f"{path.resolve()}::{os.path.getmtime(path)}"
    if key not in _bundle_cache:
        import joblib

        _bundle_cache[key] = joblib.load(path)
        logger.info("forecast_model_loaded", path=str(path))
    return _bundle_cache[key]


class XGBoostBackend:
    name = "xgboost"

    def __init__(self, session, model_dir: str) -> None:
        self._session = session
        self._bundle = load_bundle(model_dir)

    def history(self, sku: str) -> pd.Series:
        panel, _ = _load_history_rows(self._session, sku)
        return panel.set_index("date")["units"].astype(float)

    def daily_std(self, sku: str) -> float:
        residual = float(self._bundle.get("residual_std", 1.0))
        panel, _ = _load_history_rows(self._session, sku)
        recent = panel.set_index("date")["units"].astype(float).tail(_RECENT_WINDOW)
        recent_std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
        return round(max(residual, recent_std), 3)

    def predict(self, sku: str, horizon_days: int) -> list[DailyForecast]:
        panel, product = _load_history_rows(self._session, sku)
        if panel.empty or panel["units"].sum() == 0:
            return _flat_forecast(_future_dates(horizon_days), 0.0, 1.0)

        model = self._bundle["model"]
        encoder = self._bundle["encoder"]
        feature_columns = self._bundle["feature_columns"]
        residual_std = float(self._bundle.get("residual_std", 1.0))
        spread = _INTERVAL_Z * max(residual_std, 1.0)

        # Recursive one-SKU forecast without per-step pandas groupby: keep a
        # growing units list and derive lag / rolling features from its tail.
        units = [float(u) for u in panel["units"].tolist()]
        cat_codes = encoder.transform([[sku, product["aisle"]]])[0].tolist()
        last_stockout = _last_stockout_date(self._session, sku)
        max_win = max(F.ROLL_WINDOWS)

        def _tail_stats(seq: list[float], w: int) -> tuple[float, float]:
            window = np.asarray(seq[-w:], dtype=float)
            if window.size < 2:
                return (float(window.mean()) if window.size else 0.0), 0.0
            return float(window.mean()), float(window.std(ddof=1))

        preds: list[DailyForecast] = []
        for d in _future_dates(horizon_days):
            row: dict[str, float] = dict(zip(F.CATEGORICAL_COLUMNS, cat_codes))
            for k in F.LAGS:
                row[f"lag_{k}"] = units[-k] if len(units) >= k else 0.0
            rmeans: dict[int, float] = {}
            for w in F.ROLL_WINDOWS:
                mean_w, std_w = _tail_stats(units, w)
                row[f"rmean_{w}"] = mean_w
                row[f"rstd_{w}"] = std_w
                rmeans[w] = mean_w
            row["day_of_week"] = d.weekday()
            row["week_of_year"] = int(d.isocalendar()[1])
            row["month"] = d.month
            row["is_weekend"] = 1 if d.weekday() >= 5 else 0
            long_mean = rmeans[max_win]
            row["promo_flag"] = 1 if long_mean > 0 and rmeans[7] >= F._PROMO_RATIO * long_mean else 0
            row["days_since_stockout"] = (
                (d - last_stockout).days if last_stockout else F._NO_STOCKOUT_SENTINEL
            )

            X = np.array([[row[c] for c in feature_columns]], dtype=float)
            yhat = float(max(model.predict(X)[0], 0.0))
            units.append(yhat)
            preds.append(
                DailyForecast(
                    date=d,
                    predicted_units=round(yhat, 2),
                    lower=round(max(yhat - spread, 0.0), 2),
                    upper=round(yhat + spread, 2),
                )
            )
        return preds
