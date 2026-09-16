"""BizAgent Streamlit shell: landing page + shared sidebar."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components import common, theme  # noqa: E402

st.set_page_config(page_title="BizAgent", page_icon="🛒", layout="wide")
theme.inject_global_css()
common.sidebar("home")

theme.hero(
    "BizAgent — AI Supermart Operations Assistant",
    "A multi-agent operations assistant. A Supervisor agent routes each question to "
    "Sales, Inventory, Finance, Forecasting and Policy specialists, which call read-only "
    "MCP tools, an XGBoost demand model and a RAG policy knowledge base, then synthesise "
    "a single grounded answer. A dedicated LangGraph workflow produces evidence-backed "
    "root-cause reports. Every run is written to the agent decision log.",
    tags=["RAG Knowledge Base", "MCP Tool Server", "Multi-Agent Orchestration",
          "Demand Forecasting", "Root-Cause Diagnostics"],
    tag_icons=["book", "cart", "trend", "chart", "search"],
)

st.subheader("Pages")

cols = st.columns(3)
_pages = [
    ("chart", "Dashboard", "Store KPIs, aisle metrics, low stock, expiring batches, active risk alerts."),
    ("chat", "Ask BizAgent", "Ask a question and inspect the full agent trace and citations."),
    ("search", "Root Cause Analysis", "Diagnose a planted or real anomaly with ranked, evidence-backed causes."),
    ("trend", "Forecasting", "Per-SKU demand forecast, stockout risk, reorder recommendation, expiry alerts."),
    ("book", "Knowledge Base", "Ingested policy documents, upload / re-ingest, a retrieval test box."),
    ("receipt", "Agent Logs", "The decision log: every run with its tool trace and timings."),
]
for i, (icon_key, name, desc) in enumerate(_pages):
    with cols[i % 3]:
        theme.render_html(
            f"""
            <div class="bg-header" style="margin-bottom:0.9rem;">
                <div class="bg-icon">{theme.icon_svg(icon_key, 24, 'var(--forest)')}</div>
                <div>
                    <p class="bg-title" style="font-size:1.1rem;">{name}</p>
                    <p class="bg-subtitle">{desc}</p>
                </div>
            </div>
            """
        )
