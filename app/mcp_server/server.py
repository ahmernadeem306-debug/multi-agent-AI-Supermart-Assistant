"""The BizAgent MCP tool server.

A real Model Context Protocol server (FastMCP, stdio transport) exposing 13
read-only retrieval tools over the seeded POS / stockroom / supplier data and
the demand-forecast model. Launched as a subprocess by
:class:`app.mcp_client.client.MCPToolProvider` and directly by
``scripts/run_mcp_server.py``.
"""
from __future__ import annotations

from collections.abc import Callable

from app.mcp_server import (
    schemas,
    tools_finance,
    tools_forecast,
    tools_inventory,
    tools_sales,
    tools_supplier,
)

SERVER_NAME = "bizagent-tools"

# name -> plain importable tool function. One implementation, two transports:
# the FastMCP server registers these, and DirectToolProvider calls them in-process.
TOOL_REGISTRY: dict[str, Callable] = {
    "get_product_info": tools_inventory.get_product_info,
    "get_stock_level": tools_inventory.get_stock_level,
    "list_low_stock": tools_inventory.list_low_stock,
    "get_sales_history": tools_sales.get_sales_history,
    "get_top_sellers": tools_sales.get_top_sellers,
    "get_shrinkage_report": tools_finance.get_shrinkage_report,
    "get_expiring_batches": tools_inventory.get_expiring_batches,
    "get_supplier_status": tools_supplier.get_supplier_status,
    "get_open_purchase_orders": tools_supplier.get_open_purchase_orders,
    "get_margin_report": tools_finance.get_margin_report,
    "get_aisle_metrics": tools_finance.get_aisle_metrics,
    "detect_anomalies": tools_inventory.detect_anomalies,
    "get_demand_forecast": tools_forecast.get_demand_forecast,
}


# name -> Pydantic input model, for the machine-readable tool catalogue the
# Day 3 agent prompts embed.
TOOL_INPUT_MODELS = {
    "get_product_info": schemas.GetProductInfoInput,
    "get_stock_level": schemas.GetStockLevelInput,
    "list_low_stock": schemas.ListLowStockInput,
    "get_sales_history": schemas.GetSalesHistoryInput,
    "get_top_sellers": schemas.GetTopSellersInput,
    "get_shrinkage_report": schemas.GetShrinkageReportInput,
    "get_expiring_batches": schemas.GetExpiringBatchesInput,
    "get_supplier_status": schemas.GetSupplierStatusInput,
    "get_open_purchase_orders": schemas.GetOpenPurchaseOrdersInput,
    "get_margin_report": schemas.GetMarginReportInput,
    "get_aisle_metrics": schemas.GetAisleMetricsInput,
    "detect_anomalies": schemas.DetectAnomaliesInput,
    "get_demand_forecast": schemas.GetDemandForecastInput,
}


def build_server():
    """Construct and return the FastMCP server with all tools registered."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME)
    for name, fn in TOOL_REGISTRY.items():
        mcp.add_tool(fn, name=name)
    return mcp
