"""Train the global XGBoost demand model with a time-based holdout.

Never shuffles the series and never splits randomly: the last ``valid_days``
of history are the validation window and everything before is training. The
model is evaluated against the seasonal-naive baseline on that window and the
result is reported honestly, whichever wins.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.logging_config import get_logger
from app.ml import evaluate as E
from app.ml import features as F
from app.ml.backends import _MODEL_BASENAME

logger = get_logger(__name__)

_XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=4,
    objective="reg:squarederror",
)


def _load_panel_and_stockouts(session) -> tuple[pd.DataFrame, dict[str, list[dt.date]]]:
    inv = InventoryRepository(session)
    sales = SalesRepository(session)
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=730)

    rows: list[dict] = []
    stockouts: dict[str, list[dt.date]] = {}
    for product in inv.list_products():
        sku = product["sku"]
        for t in sales.list_transactions(
            sku,
            dt.datetime.combine(start, dt.time.min),
            dt.datetime.combine(end, dt.time.max),
        ):
            rows.append({"date": t["ts"], "sku": sku, "units": t["qty"], "aisle": product["aisle"]})
        levels = inv.list_stock_levels(sku, start, end)
        zero_days = [lvl["snapshot_date"] for lvl in levels if lvl["on_hand_qty"] == 0]
        if zero_days:
            stockouts[sku] = zero_days

    panel = F.daily_units_frame(rows, start=start, end=end)
    return panel, stockouts


def _baseline_predictions(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Seasonal-naive: day-of-week mean of each SKU's last 28 training days."""
    preds = np.zeros(len(valid), dtype=float)
    for sku, grp in valid.groupby("sku"):
        recent = train[train["sku"] == sku].tail(28)
        if recent.empty:
            continue
        wd_mean = recent.groupby(recent["date"].dt.dayofweek)["units"].mean()
        overall = float(recent["units"].mean())
        for pos, day in zip(grp.index.to_numpy(), grp["date"]):
            preds[pos] = float(wd_mean.get(day.dayofweek, overall))
    return preds


def train_global_model(session, *, valid_days: int = 28, model_dir: str = "./models") -> dict:
    panel, stockouts = _load_panel_and_stockouts(session)
    if panel.empty:
        raise ValueError("No sales history to train on. Seed the database first.")

    feats = F.build_features(panel, stockouts=stockouts)
    usable = F.training_matrix(feats).sort_values(["date", "sku"]).reset_index(drop=True)

    split_date = panel["date"].max() - pd.Timedelta(days=valid_days)
    train = usable[usable["date"] <= split_date].reset_index(drop=True)
    valid = usable[usable["date"] > split_date].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError("Not enough history for a time-based train/validation split.")

    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBRegressor

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train[F.CATEGORICAL_COLUMNS].to_numpy(dtype=object))

    def _matrix(df: pd.DataFrame) -> np.ndarray:
        out = df.copy()
        out[F.CATEGORICAL_COLUMNS] = encoder.transform(out[F.CATEGORICAL_COLUMNS].to_numpy(dtype=object))
        return out[F.FEATURE_COLUMNS].fillna(0.0).astype(float).to_numpy()

    X_train, y_train = _matrix(train), train[F.TARGET].to_numpy(dtype=float)
    X_valid, y_valid = _matrix(valid), valid[F.TARGET].to_numpy(dtype=float)

    model = XGBRegressor(**_XGB_PARAMS)
    model.fit(X_train, y_train)

    xgb_pred = np.clip(model.predict(X_valid), 0.0, None)
    base_pred = np.clip(_baseline_predictions(train, valid), 0.0, None)

    xgb_metrics = E.regression_metrics(y_valid, xgb_pred)
    base_metrics = E.regression_metrics(y_valid, base_pred)
    residual_std = float(np.std(y_valid - xgb_pred, ddof=0))
    the_verdict = E.verdict(xgb_metrics, base_metrics)

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "encoder": encoder,
        "feature_columns": F.FEATURE_COLUMNS,
        "categorical_columns": F.CATEGORICAL_COLUMNS,
        "residual_std": round(residual_std, 4),
    }
    import joblib

    bundle_path = model_dir_path / f"{_MODEL_BASENAME}.joblib"
    joblib.dump(bundle, bundle_path)

    meta = {
        "trained_at": dt.datetime.utcnow().isoformat(),
        "feature_columns": F.FEATURE_COLUMNS,
        "categorical_columns": F.CATEGORICAL_COLUMNS,
        "valid_days": valid_days,
        "data_range": {
            "start": str(panel["date"].min().date()),
            "end": str(panel["date"].max().date()),
        },
        "n_train_rows": int(len(train)),
        "n_valid_rows": int(len(valid)),
        "residual_std": round(residual_std, 4),
        "metrics": {"xgboost": xgb_metrics, "seasonal_naive": base_metrics},
        "verdict": the_verdict,
    }
    meta_path = model_dir_path / f"{_MODEL_BASENAME}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("forecast_model_trained", verdict=the_verdict, path=str(bundle_path))

    return {
        "model_path": str(bundle_path),
        "meta_path": str(meta_path),
        "metrics": meta["metrics"],
        "verdict": the_verdict,
        "data_range": meta["data_range"],
    }
