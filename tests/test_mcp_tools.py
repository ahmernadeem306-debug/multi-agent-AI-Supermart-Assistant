"""Direct tests of every MCP tool function (no subprocess).

Asserts output-schema validity, result-limit enforcement and error behaviour
for unknown identifiers, against the seeded dataset.
"""
from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.core.exceptions import DataNotFoundError
from app.mcp_server import schemas, tools_finance, tools_inventory, tools_sales, tools_supplier
from app.mcp_server.server import TOOL_REGISTRY

pytestmark = pytest.mark.usefixtures("seeded_db")

TODAY = dt.date.today()


def test_registry_has_thirteen_tools():
    assert len(TOOL_REGISTRY) == 13
    assert "get_demand_forecast" in TOOL_REGISTRY


def test_get_product_info_by_name_and_by_sku():
    by_name = tools_inventory.get_product_info(name_query="Milk")
    assert isinstance(by_name, schemas.ProductMatchesOutput)
    assert by_name.matches and all("sku" in m for m in by_name.matches)

    by_sku = tools_inventory.get_product_info(sku="SKU-1001")
    assert by_sku.matches[0]["sku"] == "SKU-1001"

    missing = tools_inventory.get_product_info(sku="SKU-NONE")
    assert missing.matches == []


def test_get_product_info_requires_an_argument():
    with pytest.raises(ValidationError):
        tools_inventory.get_product_info()


def test_get_stock_level_unknown_sku_raises():
    with pytest.raises(DataNotFoundError):
        tools_inventory.get_stock_level(sku="SKU-DOES-NOT-EXIST")


def test_get_stock_level_with_batches():
    out = tools_inventory.get_stock_level(sku="SKU-1001", include_batches=True)
    assert isinstance(out, schemas.StockLevelOutput)
    assert out.stock["sku"] == "SKU-1001"
    assert isinstance(out.batches, list)


def test_list_low_stock_limit_enforced():
    assert tools_inventory.list_low_stock(limit=1).count <= 1
    with pytest.raises(ValidationError):
        tools_inventory.list_low_stock(limit=9999)
    with pytest.raises(ValidationError):
        tools_inventory.list_low_stock(limit=0)


def test_get_sales_history_validates_date_order():
    out = tools_sales.get_sales_history(
        sku="SKU-1001",
        start_date=str(TODAY - dt.timedelta(days=7)),
        end_date=str(TODAY),
    )
    assert isinstance(out, schemas.SalesHistoryOutput)
    assert out.total_units >= 0
    with pytest.raises(ValidationError):
        tools_sales.get_sales_history(
            sku="SKU-1001", start_date=str(TODAY), end_date=str(TODAY - dt.timedelta(days=1))
        )


def test_get_top_sellers_and_shrinkage_report():
    top = tools_sales.get_top_sellers(days=30, limit=5)
    assert top.count <= 5
    report = tools_finance.get_shrinkage_report(aisle="Produce", days=90)
    assert isinstance(report, schemas.ShrinkageReportOutput)
    assert report.total_cost >= 0


def test_get_expiring_batches_and_margin_report():
    batches = tools_inventory.get_expiring_batches(days_ahead=10)
    assert batches.count == len(batches.items)
    margin = tools_finance.get_margin_report(aisle="Dairy", days=30)
    assert "gross_margin_pct" in margin.report


def test_get_supplier_status_by_id_and_unknown():
    out = tools_supplier.get_supplier_status(supplier_id=1)
    assert isinstance(out, schemas.SupplierStatusOutput)
    assert "on_time_delivery_rate" in out.supplier
    with pytest.raises(DataNotFoundError):
        tools_supplier.get_supplier_status(supplier_id=9999)


def test_get_supplier_status_requires_an_argument():
    with pytest.raises(ValidationError):
        tools_supplier.get_supplier_status()


def test_get_open_purchase_orders_late_filter():
    late = tools_supplier.get_open_purchase_orders(late_only=True)
    assert all(po.get("delay_days", 1) > 0 for po in late.items)


def test_get_aisle_metrics_unknown_aisle_raises():
    with pytest.raises(DataNotFoundError):
        tools_finance.get_aisle_metrics(aisle="Nonexistent Aisle")


def test_detect_anomalies_all_kinds():
    for kind in ("shrinkage", "late_delivery", "demand_shift", "stockout"):
        out = tools_inventory.detect_anomalies(kind=kind, days=90)
        assert isinstance(out, schemas.AnomaliesOutput)
        assert out.count == len(out.items)
    with pytest.raises(ValidationError):
        tools_inventory.detect_anomalies(kind="bogus")


def test_every_registered_tool_returns_a_summary():
    calls = {
        "get_product_info": {"sku": "SKU-1001"},
        "get_stock_level": {"sku": "SKU-1001"},
        "list_low_stock": {"limit": 5},
        "get_sales_history": {
            "sku": "SKU-1001",
            "start_date": str(TODAY - dt.timedelta(days=7)),
            "end_date": str(TODAY),
        },
        "get_top_sellers": {"days": 30, "limit": 3},
        "get_shrinkage_report": {"days": 90},
        "get_expiring_batches": {"days_ahead": 7},
        "get_supplier_status": {"supplier_id": 1},
        "get_open_purchase_orders": {"late_only": True},
        "get_margin_report": {"aisle": "Produce", "days": 30},
        "get_aisle_metrics": {"aisle": "Produce", "days": 30},
        "detect_anomalies": {"kind": "stockout", "days": 90},
        "get_demand_forecast": {"sku": "SKU-1001", "horizon_days": 7},
    }
    assert set(calls) == set(TOOL_REGISTRY)
    for name, kwargs in calls.items():
        out = TOOL_REGISTRY[name](**kwargs)
        assert out.summary and isinstance(out.summary, str)
