"""Parity: MCPToolProvider (real stdio subprocess) and DirectToolProvider
return identical results for identical arguments.

The MCP half spawns a subprocess, so the whole module is marked ``live`` and
is deselected by the default ``-m "not live"``. Run it explicitly with:

    pytest -m live tests/test_tool_provider_parity.py
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.mcp_client.adapter import DirectToolProvider
from app.mcp_client.client import MCPToolProvider

pytestmark = [pytest.mark.live, pytest.mark.usefixtures("seeded_db")]

TODAY = dt.date.today()

PARITY_CALLS: list[tuple[str, dict]] = [
    ("get_product_info", {"sku": "SKU-1001"}),
    ("get_stock_level", {"sku": "SKU-1001", "include_batches": True}),
    ("list_low_stock", {"limit": 25}),
    (
        "get_sales_history",
        {
            "sku": "SKU-1001",
            "start_date": str(TODAY - dt.timedelta(days=21)),
            "end_date": str(TODAY),
            "granularity": "weekly",
        },
    ),
    ("get_top_sellers", {"aisle": "Produce", "days": 30, "limit": 5}),
    ("get_shrinkage_report", {"aisle": "Produce", "days": 90}),
    ("get_supplier_status", {"supplier_id": 1}),
    ("detect_anomalies", {"kind": "stockout", "days": 90}),
    ("get_demand_forecast", {"sku": "SKU-1001", "horizon_days": 7}),
]


@pytest.fixture(scope="module")
def providers():
    direct = DirectToolProvider()
    mcp = MCPToolProvider()
    try:
        yield direct, mcp
    finally:
        mcp.close()


def test_tool_lists_match(providers):
    direct, mcp = providers
    assert direct.list_tools() == mcp.list_tools()
    assert len(direct.list_tools()) == 13


@pytest.mark.parametrize("name,arguments", PARITY_CALLS, ids=[c[0] for c in PARITY_CALLS])
def test_result_parity(providers, name, arguments):
    direct, mcp = providers
    direct_result = direct.call_tool(name, arguments)
    mcp_result = mcp.call_tool(name, arguments)

    assert direct_result.is_error is False
    assert mcp_result.is_error is False
    assert direct_result.summary == mcp_result.summary
    assert direct_result.data == mcp_result.data


def test_error_parity_unknown_sku(providers):
    direct, mcp = providers
    direct_result = direct.call_tool("get_stock_level", {"sku": "SKU-NOPE"})
    mcp_result = mcp.call_tool("get_stock_level", {"sku": "SKU-NOPE"})
    assert direct_result.is_error and mcp_result.is_error
