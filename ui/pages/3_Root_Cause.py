"""Root Cause Analysis — pick a SKU / anomaly / date range, run, render the report."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402
from components.report_view import render_rca_report  # noqa: E402

st.set_page_config(page_title="BizAgent · Root Cause", page_icon="🔍", layout="wide")
theme.inject_global_css()
common.sidebar("rca")

theme.page_header(
    "Root Cause Analysis",
    "Runs a fixed multi-step LangGraph workflow: sales → inventory → shrinkage → supply → "
    "forecast → policy evidence, then LLM hypothesis ranking. Every cause carries evidence.",
    icon="search",
)

prefill = st.session_state.get("rca_prefill", {})
sku_options = []
sku_payload = common.api_get("/forecast/skus")
if sku_payload:
    sku_options = [s["sku"] for s in sku_payload["skus"]]

col1, col2 = st.columns([2, 1])
with col1:
    default_sku = prefill.get("sku")
    idx = sku_options.index(default_sku) if default_sku in sku_options else 0
    sku = st.selectbox("SKU", sku_options, index=idx) if sku_options else st.text_input("SKU", default_sku or "")
with col2:
    anomaly_types = ["auto", "stockout", "shrinkage", "supply_delay", "margin_drop"]
    a_default = prefill.get("anomaly_type", "auto")
    anomaly_type = st.selectbox("Anomaly type", anomaly_types, index=anomaly_types.index(a_default))

date_col1, date_col2 = st.columns(2)
start_date = date_col1.date_input("Start date (optional)", value=None)
end_date = date_col2.date_input("End date (optional)", value=None)

if st.button("Run root-cause analysis", type="primary", disabled=not sku):
    payload = {"sku": sku, "anomaly_type": anomaly_type}
    if start_date:
        payload["start_date"] = str(start_date)
    if end_date:
        payload["end_date"] = str(end_date)
    with st.spinner("Gathering evidence and ranking causes…"):
        report = common.api_post("/rca", payload, timeout=180.0)
    if report:
        st.session_state["rca_last_report"] = report

if st.session_state.get("rca_last_report"):
    st.divider()
    render_rca_report(st.session_state["rca_last_report"])
