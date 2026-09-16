"""Agent Logs — searchable/filterable run history with per-run detail."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402

st.set_page_config(page_title="BizAgent · Agent Logs", page_icon="🧾", layout="wide")
theme.inject_global_css()
common.sidebar("logs")

theme.page_header(
    "Agent Decision Log",
    "Every /query and /rca run: route, agents invoked, tool calls with arguments and "
    "timing, retrieved documents, final answer and status.",
    icon="receipt",
)

limit = st.sidebar.slider("Rows", 10, 200, 50, step=10)
page = common.api_get("/logs", {"limit": limit, "offset": 0})
if not page:
    st.stop()

rows = page["runs"]
if not rows:
    st.info("No runs logged yet. Ask a question or run a root-cause analysis.")
    st.stop()

f1, f2, f3 = st.columns([2, 1, 1])
search = f1.text_input("Search query text")
routes = sorted({r["route"] for r in rows if r["route"]})
route_filter = f2.selectbox("Route", ["(all)", *routes])
status_filter = f3.selectbox("Status", ["(all)", "success", "partial", "error", "pending"])

filtered = [
    r
    for r in rows
    if (not search or search.lower() in (r["user_query"] or "").lower())
    and (route_filter == "(all)" or r["route"] == route_filter)
    and (status_filter == "(all)" or r["status"] == status_filter)
]

st.caption(f"{len(filtered)} of {page['total']} runs shown")
st.dataframe(
    pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "ts": r["ts"],
                "route": r["route"],
                "status": r["status"],
                "agents": ", ".join(r.get("agents_invoked") or []),
                "latency_ms": r["latency_ms"],
                "query": (r["user_query"] or "")[:90],
            }
            for r in filtered
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

if filtered:
    selected = st.selectbox("Inspect a run", [r["run_id"] for r in filtered])
    detail = common.api_get(f"/logs/{selected}")
    if detail:
        st.subheader("Run detail")
        st.write(f"**Query:** {detail['user_query']}")
        st.write(f"**Route:** {detail['route']} · **Status:** {detail['status']} · "
                 f"**Confidence:** {detail.get('confidence')} · **Latency:** {detail['latency_ms']} ms")
        if detail.get("error"):
            st.warning(detail["error"])
        st.write("**Final answer / summary:**")
        st.write(detail.get("final_answer") or "—")
        tool_calls = detail.get("tool_calls") or []
        st.write(f"**Tool calls ({len(tool_calls)}):**")
        if tool_calls:
            st.dataframe(pd.DataFrame(tool_calls), use_container_width=True, hide_index=True)
        citations = detail.get("retrieved_docs") or []
        if citations:
            st.write(f"**Retrieved documents ({len(citations)}):**")
            st.dataframe(pd.DataFrame(citations), use_container_width=True, hide_index=True)
