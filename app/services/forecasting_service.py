"""Forecasting orchestration and the risk views derived from a forecast.

Wraps a :class:`ForecastBackend` (XGBoost with a seasonal-naive fallback) and
turns its daily forecast into stockout risk, reorder recommendations,
perishable expiry risk and a store-wide alert ranking. Contains no LLM code.
"""
from __future__ import annotations

import datetime as dt
import functools
import math

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.exceptions import DataNotFoundError, ForecastError
from app.db.repositories.inventory_repo import InventoryRepository
from app.db.repositories.sales_repo import SalesRepository
from app.db.repositories.supplier_repo import SupplierRepository
from app.logging_config import get_logger
from app.ml.backends import SeasonalNaiveBackend, XGBoostBackend

logger = get_logger(__name__)

_RISK_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1}
_COVER_SCREEN_DAYS = 21
_DEFAULT_LEAD_TIME = 7


def _guard(fn):
    """Map any unexpected data-layer failure to a clean ForecastError."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except (DataNotFoundError, ForecastError):
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as ForecastError
            logger.error("forecast_operation_failed", op=fn.__name__, error=str(exc))
            raise ForecastError(f"{fn.__name__} failed: {exc}") from exc

    return wrapper


class ForecastingService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.inv = InventoryRepository(session)
        self.sales = SalesRepository(session)
        self.suppliers = SupplierRepository(session)
        self._backend_obj = None

    # ------------------------------------------------------------- internals
    def _backend(self):
        if self._backend_obj is None:
            choice = self.settings.forecast_backend.lower()
            if choice == "xgboost":
                try:
                    self._backend_obj = XGBoostBackend(self.session, self.settings.model_dir)
                except ForecastError as exc:
                    logger.warning("xgb_model_unavailable_using_baseline", error=exc.message)
                    self._backend_obj = SeasonalNaiveBackend(self.session)
            else:
                self._backend_obj = SeasonalNaiveBackend(self.session)
        return self._backend_obj

    @property
    def backend_name(self) -> str:
        return self._backend().name

    def _require_product(self, sku: str) -> dict:
        product = self.inv.get_product(sku)
        if product is None:
            raise DataNotFoundError(f"Unknown SKU '{sku}'.")
        return product

    def _clamp_horizon(self, horizon_days: int | None) -> int:
        horizon = horizon_days or self.settings.forecast_horizon_days
        return max(1, min(int(horizon), self.settings.forecast_max_horizon_days))

    def _on_hand(self, sku: str) -> int:
        level = self.inv.get_latest_stock_level(sku)
        return int(level["on_hand_qty"]) if level else 0

    def _lead_time_days(self, product: dict) -> int:
        supplier = self.suppliers.get_supplier(product["supplier_id"])
        return int(supplier["lead_time_days"]) if supplier else _DEFAULT_LEAD_TIME

    def _predict(self, sku: str, horizon_days: int):
        try:
            return self._backend().predict(sku, horizon_days)
        except (DataNotFoundError, ForecastError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ForecastError(f"Forecast failed for {sku}: {exc}") from exc

    def _risk_level(self, days_of_cover: float | None) -> str:
        if days_of_cover is None:
            return "low"
        if days_of_cover < self.settings.stockout_critical_days:
            return "critical"
        if days_of_cover < self.settings.stockout_high_days:
            return "high"
        if days_of_cover < self.settings.stockout_medium_days:
            return "medium"
        return "low"

    # --------------------------------------------------------------- forecast
    @_guard
    def forecast(self, sku: str, horizon_days: int | None = None) -> dict:
        product = self._require_product(sku)
        horizon = self._clamp_horizon(horizon_days)
        backend = self._backend()

        history = backend.history(sku)
        history_days = int((history > 0).any() and len(history))
        insufficient = len(history) < self.settings.forecast_min_history_days
        warnings: list[str] = []
        if insufficient:
            warnings.append(
                f"Only {len(history)} days of history (< {self.settings.forecast_min_history_days}); "
                "forecast is low-confidence."
            )
        if history.sum() == 0:
            warnings.append("This SKU has no recorded sales; forecast defaults to zero demand.")

        series = self._predict(sku, horizon)
        avg = round(sum(f.predicted_units for f in series) / len(series), 3) if series else 0.0
        return {
            "sku": sku,
            "name": product["name"],
            "aisle": product["aisle"],
            "backend": backend.name,
            "horizon_days": horizon,
            "generated_at": dt.datetime.utcnow().isoformat(),
            "history": [
                {"date": d.date().isoformat(), "units": float(v)}
                for d, v in history.tail(60).items()
            ],
            "forecast": [f.model_dump(mode="json") for f in series],
            "avg_daily_forecast": avg,
            "insufficient_history": insufficient,
            "history_days": history_days,
            "warnings": warnings,
        }

    # ---------------------------------------------------------- stockout risk
    @_guard
    def stockout_risk(self, sku: str, horizon_days: int | None = None) -> dict:
        product = self._require_product(sku)
        window = self.settings.forecast_max_horizon_days
        on_hand = self._on_hand(sku)
        series = self._predict(sku, window)

        cumulative = 0.0
        projected_date: str | None = None
        days_of_cover: float | None = None
        for i, f in enumerate(series, start=1):
            cumulative += f.predicted_units
            if cumulative >= on_hand and projected_date is None:
                projected_date = f.date.isoformat()
                days_of_cover = i - 1 if on_hand > 0 else 0
                break
        avg_daily = round(sum(f.predicted_units for f in series) / len(series), 3) if series else 0.0
        if projected_date is None:
            days_of_cover = None

        level = self._risk_level(days_of_cover)
        return {
            "sku": sku,
            "name": product["name"],
            "aisle": product["aisle"],
            "on_hand_qty": on_hand,
            "projected_stockout_date": projected_date,
            "days_of_cover": days_of_cover if days_of_cover is not None else f">{window}",
            "risk_level": level,
            "assumptions": {
                "backend": self.backend_name,
                "projection_window_days": window,
                "avg_daily_forecast": avg_daily,
                "no_replenishment_assumed": True,
            },
        }

    # --------------------------------------------------------- reorder point
    @_guard
    def reorder_point(self, sku: str) -> dict:
        product = self._require_product(sku)
        lead_time = self._lead_time_days(product)
        default_h = self.settings.forecast_horizon_days
        series = self._predict(sku, max(lead_time, default_h))

        lead_demand = sum(f.predicted_units for f in series[:lead_time])
        horizon_demand = sum(f.predicted_units for f in series[:default_h])
        sigma = self._backend().daily_std(sku)
        safety_stock = self.settings.reorder_z_score * sigma * math.sqrt(max(lead_time, 1))

        recommended_rop = int(round(lead_demand + safety_stock))
        current_rop = int(product["reorder_point"])
        on_hand = self._on_hand(sku)
        order_qty = int(round(max(0.0, lead_demand + horizon_demand + safety_stock - on_hand)))

        return {
            "sku": sku,
            "name": product["name"],
            "lead_time_days": lead_time,
            "forecast_lead_time_demand": round(lead_demand, 2),
            "safety_stock": round(safety_stock, 2),
            "daily_demand_std": sigma,
            "recommended_reorder_point": recommended_rop,
            "current_reorder_point": current_rop,
            "delta": recommended_rop - current_rop,
            "recommended_order_qty": order_qty,
            "backend": self.backend_name,
        }

    # ----------------------------------------------------------- expiry risk
    @_guard
    def expiry_risk(self, sku: str) -> list[dict]:
        product = self._require_product(sku)
        batches = [b for b in self.inv.list_batches(sku) if b["qty_remaining"] > 0]
        if not product["is_perishable"] or not batches:
            return []

        window = self.settings.forecast_max_horizon_days
        series = self._predict(sku, window)
        today = dt.date.today()
        unit_cost = float(product["unit_cost"])

        consumed = 0.0
        out: list[dict] = []
        for batch in sorted(batches, key=lambda b: b["expiry_date"]):
            days_left = (batch["expiry_date"] - today).days
            usable_days = max(0, min(days_left, window))
            demand_by_expiry = sum(f.predicted_units for f in series[:usable_days])
            available = max(0.0, demand_by_expiry - consumed)
            sold = min(float(batch["qty_remaining"]), available)
            consumed += sold
            projected_expiring = max(0.0, batch["qty_remaining"] - sold)
            ratio = projected_expiring / batch["qty_remaining"] if batch["qty_remaining"] else 0.0
            action = (
                "markdown" if ratio >= 0.5 else "promote" if ratio >= 0.2 else "transfer" if ratio > 0 else "none"
            )
            out.append(
                {
                    "sku": sku,
                    "batch_no": batch["batch_no"],
                    "expiry_date": batch["expiry_date"].isoformat(),
                    "days_until_expiry": days_left,
                    "qty_remaining": batch["qty_remaining"],
                    "projected_units_expiring": int(round(projected_expiring)),
                    "estimated_write_off_value": round(projected_expiring * unit_cost, 2),
                    "recommended_action": action,
                }
            )
        return out

    # ------------------------------------------------------------- bundles
    @_guard
    def sku_alerts(self, sku: str) -> dict:
        return {
            "sku": sku,
            "stockout_risk": self.stockout_risk(sku),
            "reorder_recommendation": self.reorder_point(sku),
            "expiry_risk": self.expiry_risk(sku),
        }

    def _quick_cover(self, sku: str, on_hand: int) -> float | None:
        end = dt.date.today()
        start = end - dt.timedelta(days=30)
        units = self.sales.total_units_sold(
            sku, dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)
        )
        avg = units / 30
        return round(on_hand / avg, 1) if avg > 0 else None

    @_guard
    def store_alerts(self, limit: int = 25) -> dict:
        today = dt.date.today()
        expiring = {
            b["sku"]
            for b in self.inv.list_expiring_batches(as_of=today, days_ahead=self.settings.stockout_medium_days)
        }
        # Fast screen everything on velocity-based cover, then run the full
        # forecast only for the worst candidates (bounds cost on 60+ SKUs).
        screened: list[tuple[float, dict]] = []
        for product in self.inv.list_products():
            sku = product["sku"]
            on_hand = self._on_hand(sku)
            cover = self._quick_cover(sku, on_hand)
            if cover is None and sku not in expiring:
                continue
            rank_key = cover if cover is not None else 0.0
            screened.append((rank_key, product))
        screened.sort(key=lambda t: t[0])
        candidates = [p for _, p in screened[: max(limit * 2, 20)]]

        alerts: list[dict] = []
        for product in candidates:
            sku = product["sku"]
            risk = self.stockout_risk(sku)
            expiry = self.expiry_risk(sku) if product["is_perishable"] else []
            expiring_units = sum(e["projected_units_expiring"] for e in expiry)
            doc = risk["days_of_cover"]
            numeric_cover = doc if isinstance(doc, (int, float)) else self.settings.forecast_max_horizon_days
            score = _RISK_WEIGHT[risk["risk_level"]] * 10 + (30 - min(numeric_cover, 30)) + expiring_units / 10
            alerts.append(
                {
                    "sku": sku,
                    "name": product["name"],
                    "aisle": product["aisle"],
                    "risk_level": risk["risk_level"],
                    "days_of_cover": risk["days_of_cover"],
                    "projected_stockout_date": risk["projected_stockout_date"],
                    "projected_units_expiring": int(expiring_units),
                    "risk_score": round(score, 2),
                }
            )
        alerts.sort(key=lambda a: a["risk_score"], reverse=True)
        return {"generated_at": dt.datetime.utcnow().isoformat(), "count": len(alerts), "alerts": alerts[:limit]}

    @_guard
    def reorder_recommendations(self, limit: int = 25) -> dict:
        screened: list[tuple[float, str]] = []
        for product in self.inv.list_products():
            sku = product["sku"]
            cover = self._quick_cover(sku, self._on_hand(sku))
            if cover is not None and cover >= _COVER_SCREEN_DAYS:
                continue
            screened.append((cover if cover is not None else 0.0, sku))
        screened.sort(key=lambda t: t[0])

        rows: list[dict] = []
        for _, sku in screened[: max(limit * 2, 20)]:
            rec = self.reorder_point(sku)
            threshold = max(1.0, 0.15 * rec["current_reorder_point"])
            if abs(rec["delta"]) >= threshold:
                rows.append(rec)
        rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        return {"generated_at": dt.datetime.utcnow().isoformat(), "count": len(rows), "recommendations": rows[:limit]}
