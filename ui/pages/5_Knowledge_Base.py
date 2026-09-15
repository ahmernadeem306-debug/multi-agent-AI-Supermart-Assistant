"""Knowledge Base — document list, upload, re-ingest, chunk counts, retrieval test."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, theme  # noqa: E402

st.set_page_config(page_title="BizAgent · Knowledge Base", page_icon="📚", layout="wide")
theme.inject_global_css()
common.sidebar("kb")

theme.page_header(
    "Knowledge Base",
    "Policy SOPs, the returns policy, the employee handbook and supplier contracts, "
    "ingested into ChromaDB with cited retrieval.",
    icon="book",
)

docs = common.api_get("/documents")
if docs:
    st.subheader(f"Ingested documents ({docs['count']}) — {docs['total_chunks']} chunks")
    if docs["documents"]:
        frame = pd.DataFrame(docs["documents"])[
            ["title", "doc_type", "chunk_count", "ingested_at", "source_path"]
        ]
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.info("No documents ingested yet. Run `python scripts/ingest_docs.py`.")

col_up, col_re = st.columns(2)
with col_up:
    st.subheader("Upload a document")
    uploaded = st.file_uploader("PDF, Markdown, text or Word (max 10 MB)", type=["pdf", "md", "txt", "docx"])
    if uploaded and st.button("Upload and ingest"):
        try:
            r = httpx.post(
                f"{common.API_BASE_URL}/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=180.0,
            )
            if r.status_code >= 400:
                st.error(r.json().get("detail", "Upload failed."))
            else:
                res = r.json()
                st.success(f"Ingested {res['filename']} ({(res.get('ingested') or {}).get('chunk_count', 0)} chunks).")
                common.api_get.clear()
                st.rerun()
        except httpx.HTTPError as exc:
            st.error(f"Upload failed: {exc}")

with col_re:
    st.subheader("Re-ingest corpus")
    force = st.checkbox("Force re-embed all files")
    if st.button("Re-ingest"):
        res = common.api_post("/documents/reingest", {"force": force}, timeout=300.0)
        if res:
            st.success(f"{len(res['results'])} file(s) processed, {res['total_chunks']} chunks total.")
            common.api_get.clear()
            st.rerun()

st.subheader("Retrieval test")
query = st.text_input("Ask a policy question to see which chunks are retrieved")
if st.button("Search", type="primary") and query:
    result = common.api_post("/documents/search", {"query": query})
    if result:
        if not result.get("found"):
            st.warning(result.get("message", "No supporting policy document found above the threshold."))
        for c in result.get("citations", []):
            st.markdown(
                f"**{c['doc_title']}** — {c.get('section') or 'n/a'} "
                f"(chunk {c['chunk_index']}, score {c['score']})"
            )
            st.caption(c["snippet"])
