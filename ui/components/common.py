"""Shared helpers: API access with caching, the sidebar, and error rendering."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# Make ``from components import ...`` work when Streamlit runs a page module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import theme  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120.0

PLANTED_DEMOS = {
    "Late-delivery stockout (SKU-1035)": {"sku": "SKU-1035", "anomaly_type": "stockout"},
    "Theft shrinkage spike (SKU-1002)": {"sku": "SKU-1002", "anomaly_type": "shrinkage"},
    "Expiry write-off + stockout (SKU-1001)": {"sku": "SKU-1001", "anomaly_type": "stockout"},
    "Demand shift, stale reorder point (SKU-1003)": {"sku": "SKU-1003", "anomaly_type": "auto"},
}


def _show_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
        except Exception:  # noqa: BLE001
            body = {}
        st.error(f"[{body.get('error_code', body.get('detail', 'error'))}] "
                 f"{body.get('message', body.get('detail', exc))}")
    elif isinstance(exc, httpx.RequestError):
        st.error("Could not reach the BizAgent API. Is it running? (`make run-api`)")
    else:
        st.error(f"Unexpected error: {exc}")


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE_URL}{path}", params=params or {}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return None


def api_post(path: str, payload: dict, *, timeout: float = _TIMEOUT) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        if r.status_code >= 400:
            _show_error(httpx.HTTPStatusError("error", request=r.request, response=r))
            return None
        return r.json()
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return None


def health() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def sidebar(active: str = "") -> None:
    """Render the shared sidebar: health, config indicators, demo shortcut."""
    with st.sidebar:
        theme.render_html(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem;">'
            f'{theme.icon_svg("cart", 26, "var(--wheat)")}'
            f'<span style="font-family:\'Fraunces\',serif;font-weight:600;font-size:1.4rem;">BizAgent</span>'
            f'</div>'
        )
        h = health()
        if h and h.get("status") == "ok":
            st.success("API: connected")
        elif h:
            st.warning("API: degraded")
        else:
            st.error("API: unreachable")

        if h:
            st.caption(f"Tool provider: **{h.get('tool_provider', '?')}**")
            backend = h.get("forecast_backend", "?")
            trained = h.get("forecast_model_trained")
            note = "" if backend != "xgboost" else (" (model trained)" if trained else " (no model — using baseline)")
            st.caption(f"Forecast backend: **{backend}**{note}")

        st.divider()
        st.caption("Run demo scenario")
        choice = st.selectbox("Planted anomaly", list(PLANTED_DEMOS), key="demo_choice", label_visibility="collapsed")
        if st.button("Load into Root Cause →", use_container_width=True):
            st.session_state["rca_prefill"] = PLANTED_DEMOS[choice]
            try:
                st.switch_page("pages/3_Root_Cause.py")
            except Exception:  # noqa: BLE001 - older Streamlit
                st.info("Open the Root Cause Analysis page; the scenario is pre-filled.")

        st.divider()
        st.caption(
            "Pages: Dashboard · Ask BizAgent · Root Cause · Forecasting · Knowledge Base · "
            "Agent Logs · Manage Data"
        )
