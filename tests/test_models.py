from __future__ import annotations

from sqlalchemy import inspect

from app.db.base import get_engine

EXPECTED_TABLES = {
    "suppliers",
    "products",
    "stock_levels",
    "stock_batches",
    "sales_transactions",
    "purchase_orders",
    "shrinkage_events",
    "documents",
    "agent_runs",
}


def test_all_nine_tables_created():
    inspector = inspect(get_engine())
    table_names = set(inspector.get_table_names())
    assert EXPECTED_TABLES <= table_names


def test_stock_levels_has_sku_snapshot_date_index():
    inspector = inspect(get_engine())
    index_columns = {
        col for idx in inspector.get_indexes("stock_levels") for col in idx["column_names"]
    }
    assert {"sku", "snapshot_date"} <= index_columns


def test_sales_transactions_has_sku_ts_index():
    inspector = inspect(get_engine())
    index_columns = {
        col for idx in inspector.get_indexes("sales_transactions") for col in idx["column_names"]
    }
    assert {"sku", "ts"} <= index_columns


def test_purchase_orders_has_sku_status_index():
    inspector = inspect(get_engine())
    index_columns = {
        col for idx in inspector.get_indexes("purchase_orders") for col in idx["column_names"]
    }
    assert {"sku", "status"} <= index_columns


def test_shrinkage_events_has_sku_event_date_index():
    inspector = inspect(get_engine())
    index_columns = {
        col for idx in inspector.get_indexes("shrinkage_events") for col in idx["column_names"]
    }
    assert {"sku", "event_date"} <= index_columns


def test_products_primary_key_is_sku():
    inspector = inspect(get_engine())
    pk = inspector.get_pk_constraint("products")
    assert pk["constrained_columns"] == ["sku"]
