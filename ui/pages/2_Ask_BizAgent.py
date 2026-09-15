"""Ask BizAgent — question box, answer, agent trace and citations."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402

st.set_page_config(page_title="BizAgent · Ask", page_icon="💬", layout="wide")
theme.inject_global_css()
common.sidebar("ask")

theme.page_header("Ask BizAgent",
                   "Supervisor routing → specialist agents → MCP tools + RAG → synthesised answer.",
                   icon="chat")

question = st.text_input("Ask a question about your store's operations")

if st.button("Ask", type="primary") and question:
    with st.spinner("Routing to specialist agents…"):
        data = common.api_post("/query", {"question": question}, timeout=180.0)
    if not data:
        st.stop()

    if data.get("status") != "success":
        st.warning(f"Answer is degraded (status: {data.get('status')}). Some agents could not complete.")

    st.markdown("### Answer")
    st.write(data["answer"])
    meta = f"run_id `{data['run_id']}` · route **{data.get('route')}** · {data['latency_ms']} ms"
    if data.get("confidence") is not None:
        meta += f" · confidence {data['confidence']:.2f}"
    st.caption(meta)

    with st.expander("Agent trace", expanded=True):
        st.write(f"**Route:** {data.get('route')}")
        st.write(f"**Agents invoked:** {', '.join(data.get('agents_invoked') or []) or '—'}")
        if data.get("plan_reasoning"):
            st.write(f"**Plan reasoning:** {data['plan_reasoning']}")
        tool_calls = data.get("tool_calls") or []
        if tool_calls:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "agent": tc.get("agent"),
                            "tool": tc.get("tool"),
                            "arguments": str(tc.get("arguments")),
                            "rows": tc.get("row_count"),
                            "ms": tc.get("duration_ms"),
                            "error": tc.get("error_code") or "",
                        }
                        for tc in tool_calls
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No MCP tools were called for this question.")

    citations = data.get("citations") or []
    if citations:
        st.markdown("### Citations")
        for c in citations:
            st.markdown(
                f"**{c.get('doc_title')}** — {c.get('section') or 'n/a'} "
                f"(chunk {c.get('chunk_index')}, score {c.get('score')})"
            )
            st.caption(c.get("snippet", ""))
