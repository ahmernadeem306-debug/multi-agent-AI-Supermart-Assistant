"""Forecast accuracy metrics with zero-safe MAPE handling."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, MAPE (over non-zero actuals) and zero-safe WAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"mae": None, "rmse": None, "mape": None, "wape": None, "n": 0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None
    )
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom * 100) if denom > 0 else None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2) if mape is not None else None,
        "wape": round(wape, 2) if wape is not None else None,
        "n": int(y_true.size),
    }


def verdict(model_metrics: dict, baseline_metrics: dict) -> str:
    """Plain-language comparison of the model against the seasonal-naive baseline."""
    m, b = model_metrics.get("mae"), baseline_metrics.get("mae")
    if m is None or b is None:
        return "inconclusive (missing metrics)"
    if m < b:
        return f"XGBoost beats the baseline (MAE {m} vs {b}, {round((b - m) / b * 100, 1)}% better)"
    if m > b:
        return f"XGBoost does NOT beat the baseline (MAE {m} vs {b}); the baseline is used unless overridden"
    return "XGBoost matches the baseline (MAE tie)"
