"""Train the global XGBoost demand model and back-test it against the baseline.

    python scripts/train_forecast.py [--valid-days 28]

Uses a time-based holdout (never random), prints an MAE/RMSE/MAPE comparison
table versus the seasonal-naive baseline, and saves the model plus a metadata
JSON to models/. If XGBoost does not beat the baseline it says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db.base import get_session  # noqa: E402
from app.ml.train import train_global_model  # noqa: E402


def _row(name: str, m: dict) -> str:
    return f"  {name:<16} {str(m['mae']):>10} {str(m['rmse']):>10} {str(m['mape']):>8} {str(m['wape']):>8}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BizAgent demand forecast model.")
    parser.add_argument("--valid-days", type=int, default=28, help="Length of the holdout window.")
    args = parser.parse_args()

    settings = get_settings()
    with get_session() as session:
        result = train_global_model(
            session, valid_days=args.valid_days, model_dir=settings.model_dir
        )

    m = result["metrics"]
    print(f"\nData range: {result['data_range']['start']} -> {result['data_range']['end']}")
    print(f"\n  {'model':<16} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'WAPE':>8}")
    print("  " + "-" * 56)
    print(_row("seasonal_naive", m["seasonal_naive"]))
    print(_row("xgboost", m["xgboost"]))
    print(f"\nVerdict: {result['verdict']}")
    print(f"Saved:   {result['model_path']}")
    print(f"         {result['meta_path']}")


if __name__ == "__main__":
    main()
