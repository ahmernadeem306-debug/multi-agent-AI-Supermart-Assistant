"""Dashboard — store KPIs, aisle metrics, low stock, expiring batches, active alerts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402

st.set_page_config(page_title="BizAgent · Dashboard", page_icon="📊", layout="wide")
theme.inject_global_css()
common.sidebar("dashboard")

theme.page_header("Operations Dashboard",
                   "Store KPIs, aisle metrics, low stock, expiring batches and active risk alerts.",
                   icon="chart")
days = st.sidebar.slider("Metrics window (days)", 7, 180, 30, step=7)

aisles_payload = common.api_get("/metrics/aisles", {"days": days})
if aisles_payload:
    frame = pd.DataFrame(aisles_payload["aisles"]).set_index("aisle")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"{frame['revenue'].sum():,.0f}")
    k2.metric("Units sold", f"{int(frame['units_sold'].sum()):,}")
    k3.metric("Out-of-stock SKUs", int(frame["out_of_stock_count"].sum()))
    k4.metric("Avg margin %", f"{frame['avg_margin_pct'].mean():.1f}")

    st.subheader("Aisle metrics")
    st.dataframe(frame, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Revenue by aisle**")
    c1.bar_chart(frame["revenue"])
    c2.markdown("**Units sold by aisle**")
    c2.bar_chart(frame["units_sold"])
    c3.markdown("**Shrinkage value by aisle**")
    c3.bar_chart(frame["shrinkage_value"])

st.subheader("Active risk alerts")
alerts = common.api_get("/forecast/alerts", {"limit": 15})
if alerts is not None:
    rows = alerts.get("alerts", [])
    if rows:
        st.caption(f"{alerts['count']} SKU(s) flagged for stockout / expiry risk")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No SKUs are currently at risk.")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Low stock")
    low = common.api_get("/metrics/low-stock", {"limit": 100})
    if low is not None:
        st.caption(f"{low['count']} SKU(s) at or below reorder point")
        st.dataframe(pd.DataFrame(low["items"]) if low["items"] else pd.DataFrame(),
                     use_container_width=True, hide_index=True)
with col_b:
    st.subheader("Expiring batches (7 days)")
    exp = common.api_get("/metrics/expiring", {"days_ahead": 7})
    if exp is not None:
        st.caption(f"{exp['count']} batch(es)")
        st.dataframe(pd.DataFrame(exp["items"]) if exp["items"] else pd.DataFrame(),
                     use_container_width=True, hide_index=True)
