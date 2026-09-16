"""Render a RootCauseReport in Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

_URGENCY_COLOR = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


def render_rca_report(report: dict) -> None:
    status = report.get("status")
    if status == "no_anomaly":
        st.info("No significant anomaly was detected for this target.")
    elif status == "partial":
        st.warning("Partial report — some evidence was unavailable or the LLM was not reachable.")

    st.subheader("Summary")
    st.write(report.get("summary", "—"))
    st.caption(
        f"status: {status} · llm calls: {report.get('llm_calls', 0)} · "
        f"generated {report.get('generated_at', '')}"
    )

    evidence = {e["id"]: e for e in report.get("evidence", [])}

    st.subheader("Ranked causes")
    causes = report.get("ranked_causes", [])
    if not causes:
        st.caption("No causes were established.")
    for i, c in enumerate(causes, start=1):
        label = "Primary cause" if i == 1 and not c.get("contributing_factor") else f"Contributing factor {i}"
        st.markdown(f"**{label} — {c['category']}**  ·  confidence {c['confidence']:.0%}")
        st.progress(min(max(c["confidence"], 0.0), 1.0))
        st.write(c["cause"])
        with st.expander(f"Evidence ({len(c.get('evidence_ids', []))})"):
            for eid in c.get("evidence_ids", []):
                ev = evidence.get(eid)
                if ev:
                    st.markdown(f"- **{ev['title']}** ({ev['source']}): {ev['detail']}")
                else:
                    st.markdown(f"- `{eid}` (unresolved)")

    timeline = report.get("timeline", [])
    if timeline:
        st.subheader("Timeline")
        st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)

    findings = report.get("policy_findings", [])
    if findings:
        st.subheader("Policy findings")
        for f in findings:
            cit = f.get("citation", {})
            st.markdown(f"**{cit.get('doc_title', 'Policy')}** — {f.get('relevance', '')}")
            st.caption(f"“{f.get('clause', '')}” · {cit.get('source_path', '')} "
                       f"(chunk {cit.get('chunk_index')}, score {cit.get('score')})")

    actions = report.get("recommended_actions", [])
    if actions:
        st.subheader("Recommended actions")
        for a in actions:
            st.markdown(
                f"{_URGENCY_COLOR.get(a.get('urgency'), '•')} **{a.get('action')}**  \n"
                f"Owner: {a.get('owner_role')} · Urgency: {a.get('urgency')} · "
                f"Expected impact: {a.get('expected_impact')}"
            )

    gaps = report.get("data_gaps", [])
    if gaps:
        st.subheader("Data gaps")
        for g in gaps:
            st.markdown(f"- {g}")

    with st.expander("Full tool trace"):
        st.dataframe(pd.DataFrame(report.get("tool_trace", [])), use_container_width=True)
