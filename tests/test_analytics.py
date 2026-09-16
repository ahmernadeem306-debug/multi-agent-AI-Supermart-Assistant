"""Analytics tests: each detector flags its planted anomaly and spares a
known-normal control SKU. Runs against the full seeded dataset.
"""
from __future__ import annotations

import pytest

from app.services.analytics_service import AnalyticsService

# A stable seeded SKU with no planted anomaly: no stockouts, no shrinkage,
# no late POs, no demand shift (verified against the seeded dataset).
CONTROL_SKU = "SKU-1010"


@pytest.fixture()
def analytics(read_session):
    return AnalyticsService(read_session)


def test_detect_stockout_flags_planted_and_spares_control(analytics, planted_anomalies):
    flagged = {row["sku"] for row in analytics.detect_stockout_events(days=90)}
    assert planted_anomalies["A"]["sku"] in flagged  # late-delivery stockout
    assert planted_anomalies["C"]["sku"] in flagged  # expiry write-off stockout
    assert CONTROL_SKU not in flagged


def test_detect_shrinkage_spike_flags_planted_and_spares_control(analytics, planted_anomalies):
    rows = analytics.detect_shrinkage_spikes(days=90)
    flagged = {row["sku"] for row in rows}
    assert planted_anomalies["B"]["sku"] in flagged
    assert CONTROL_SKU not in flagged
    spike = next(r for r in rows if r["sku"] == planted_anomalies["B"]["sku"])
    assert spike["dominant_reason"] == "theft"


def test_detect_demand_shift_flags_planted_and_spares_control(analytics, planted_anomalies):
    rows = analytics.detect_demand_shifts()
    flagged = {row["sku"] for row in rows}
    assert planted_anomalies["D"]["sku"] in flagged
    assert CONTROL_SKU not in flagged
    shift = next(r for r in rows if r["sku"] == planted_anomalies["D"]["sku"])
    assert shift["ratio"] >= 1.5


def test_detect_late_deliveries_flags_planted_and_spares_control(analytics, planted_anomalies):
    rows = analytics.detect_late_deliveries()
    flagged = {row["sku"] for row in rows}
    assert planted_anomalies["A"]["sku"] in flagged
    assert CONTROL_SKU not in flagged
    assert all(row["delay_days"] > 0 for row in rows)


def test_detect_dispatch_rejects_unknown_kind(analytics):
    with pytest.raises(ValueError):
        analytics.detect("not-a-kind")


def test_aisle_metrics_shape(analytics):
    metrics = analytics.aisle_metrics("Produce", days=30)
    for key in (
        "sku_count",
        "revenue",
        "units_sold",
        "out_of_stock_count",
        "low_stock_count",
        "shrinkage_value",
        "avg_margin_pct",
        "expiring_batch_count",
    ):
        assert key in metrics
    assert metrics["sku_count"] > 0

    everything = analytics.all_aisle_metrics(days=30)
    assert len(everything) == 8
