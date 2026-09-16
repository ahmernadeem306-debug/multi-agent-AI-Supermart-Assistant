"""Forecasting — SKU selector, history + forecast chart, risk badges, metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402
from components.charts import history_forecast_chart  # noqa: E402

st.set_page_config(page_title="BizAgent · Forecasting", page_icon="📈", layout="wide")
theme.inject_global_css()
common.sidebar("forecast")

theme.page_header("Demand Forecasting",
                   "Per-SKU demand forecast, stockout risk, reorder recommendation and expiry alerts.",
                   icon="trend")

_RISK_BADGE = {"critical": "🔴 critical", "high": "🟠 high", "medium": "🟡 medium", "low": "🟢 low"}

sku_payload = common.api_get("/forecast/skus")
skus = [s["sku"] for s in sku_payload["skus"]] if sku_payload else []
col1, col2 = st.columns([3, 1])
sku = col1.selectbox("SKU", skus) if skus else col1.text_input("SKU", "")
horizon = col2.slider("Horizon (days)", 7, 30, 14, step=1)

if sku:
    data = common.api_get(f"/forecast/{sku}", {"horizon_days": horizon, "include_risk": True})
    if data:
        for w in data.get("warnings", []):
            st.warning(w)
        st.caption(f"Backend: **{data['backend']}** · avg forecast {data['avg_daily_forecast']} units/day")
        st.altair_chart(
            history_forecast_chart(data["history"], data["forecast"]), use_container_width=True
        )

        risk = data.get("stockout_risk") or {}
        reorder = data.get("reorder_recommendation") or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("Stockout risk", _RISK_BADGE.get(risk.get("risk_level"), risk.get("risk_level", "—")),
                  help=f"Projected stockout: {risk.get('projected_stockout_date') or 'beyond horizon'}")
        m2.metric("Days of cover", str(risk.get("days_of_cover", "—")))
        m3.metric("Reorder point", reorder.get("recommended_reorder_point", "—"),
                  delta=reorder.get("delta"), help=f"Current: {reorder.get('current_reorder_point')}")

        if reorder:
            st.write(
                f"**Reorder recommendation:** set reorder point to "
                f"**{reorder.get('recommended_reorder_point')}** "
                f"(currently {reorder.get('current_reorder_point')}, Δ{reorder.get('delta')}); "
                f"recommended order quantity **{reorder.get('recommended_order_qty')}** "
                f"(lead time {reorder.get('lead_time_days')} days, safety stock {reorder.get('safety_stock')})."
            )

        expiry = data.get("expiry_risk") or []
        if expiry:
            st.subheader("Perishable expiry risk")
            st.dataframe(pd.DataFrame(expiry), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Backtest metrics")
meta_path = Path("models/forecast_xgb.meta.json")
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.caption(f"Trained {meta.get('trained_at', '')} · data {meta['data_range']['start']} → {meta['data_range']['end']}")
    st.dataframe(pd.DataFrame(meta["metrics"]).T, use_container_width=True)
    st.write(f"**Verdict:** {meta.get('verdict')}")
else:
    st.info("No trained model yet. Run `python scripts/train_forecast.py`. The seasonal-naive baseline is used until then.")
