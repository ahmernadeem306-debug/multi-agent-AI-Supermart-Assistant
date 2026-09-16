"""Demo artefact: exercise every MCP tool over a real stdio session.

    python scripts/mcp_smoke_test.py

Connects as an MCP client, lists the tools the server advertises, then calls
each one with realistic arguments against the seeded database and prints the
one-line summary it returns.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_client.client import MCPToolProvider  # noqa: E402

_TODAY = dt.date.today()

CALLS: list[tuple[str, dict]] = [
    ("get_product_info", {"name_query": "Milk"}),
    ("get_stock_level", {"sku": "SKU-1035", "include_batches": True}),
    ("list_low_stock", {"limit": 10}),
    (
        "get_sales_history",
        {
            "sku": "SKU-1035",
            "start_date": str(_TODAY - dt.timedelta(days=21)),
            "end_date": str(_TODAY),
            "granularity": "weekly",
        },
    ),
    ("get_top_sellers", {"days": 30, "limit": 5}),
    ("get_shrinkage_report", {"aisle": "Produce", "days": 90}),
    ("get_expiring_batches", {"days_ahead": 10}),
    ("get_supplier_status", {"supplier_id": 1}),
    ("get_open_purchase_orders", {"late_only": True}),
    ("get_margin_report", {"aisle": "Dairy", "days": 30}),
    ("get_aisle_metrics", {"aisle": "Beverages", "days": 30}),
    ("detect_anomalies", {"kind": "demand_shift", "days": 90}),
    ("get_demand_forecast", {"sku": "SKU-1035", "horizon_days": 7, "include_risk": True}),
]


def main() -> None:
    provider = MCPToolProvider()
    try:
        tools = provider.list_tools()
        print(f"Server advertises {len(tools)} tools:")
        for name in tools:
            print(f"  - {name}")
        print()

        missing = sorted(set(tools) - {name for name, _ in CALLS})
        if missing:
            print(f"WARNING: no smoke call defined for: {missing}\n")

        failures = 0
        for name, args in CALLS:
            result = provider.call_tool(name, args)
            flag = "ERROR" if result.is_error else "ok"
            print(f"[{flag}] {name}({args}) -> {result.summary}")
            if result.is_error:
                failures += 1
        print()
        print("All tool calls succeeded." if not failures else f"{failures} tool call(s) failed.")
        sys.exit(1 if failures else 0)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
