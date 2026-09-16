"""Run the root-cause workflow over all four planted anomalies.

    python scripts/run_demo_scenarios.py

This is both the demo rehearsal and a regression harness: it prints each
RootCauseReport and exits non-zero if any planted anomaly does not resolve to
its expected primary cause category. Works with or without a Groq key (it
falls back to the deterministic ranker).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm.groq_client import GroqProvider  # noqa: E402
from app.mcp_client.adapter import DirectToolProvider  # noqa: E402
from app.agents.rca_schemas import RcaRequest  # noqa: E402
from app.agents.rca_workflow import RcaWorkflow  # noqa: E402

_EXPECTED = {
    "stockout_late_delivery": {"supply_delay"},
    "shrinkage_theft_spike": {"shrinkage_theft"},
    "expiry_writeoff_stockout": {"expiry_writeoff"},
    "demand_shift_stale_reorder_point": {"reorder_point", "demand_shift"},
}


def _retriever():
    try:
        from app.rag.retriever import build_retriever

        return build_retriever()
    except Exception as exc:  # noqa: BLE001
        print(f"  (retriever unavailable: {exc})")

        class _Null:
            def retrieve(self, *a, **k):
                raise RuntimeError("knowledge base unavailable")

        return _Null()


def main() -> None:
    settings = get_settings()
    llm = GroqProvider(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        timeout_seconds=settings.groq_timeout_seconds,
    )
    workflow = RcaWorkflow(llm, DirectToolProvider(), _retriever())
    anomalies = json.loads(Path("data/anomalies.json").read_text(encoding="utf-8"))["anomalies"]

    failures = 0
    for a in anomalies:
        started = time.monotonic()
        report = workflow.run(RcaRequest(sku=a["sku"], anomaly_type="auto"))
        elapsed_ms = int((time.monotonic() - started) * 1000)
        primary = report.ranked_causes[0].category if report.ranked_causes else None
        expected = _EXPECTED.get(a["type"], set())
        ok = primary in expected
        failures += 0 if ok else 1

        print("\n" + "=" * 78)
        print(f"[{a['id']}] {a['sku']} — {a['type']}   ({'PASS' if ok else 'FAIL'})")
        print("=" * 78)
        print(f"status={report.status}  llm_calls={report.llm_calls}  latency={elapsed_ms} ms")
        print(f"summary: {report.summary}")
        print("ranked causes:")
        for c in report.ranked_causes:
            print(f"  - [{c.category}] conf={c.confidence:.2f} contributing={c.contributing_factor}")
            print(f"      {c.cause}")
            print(f"      evidence: {', '.join(c.evidence_ids)}")
        print(f"policy findings: {len(report.policy_findings)}  timeline events: {len(report.timeline)}  "
              f"recommended actions: {len(report.recommended_actions)}  data gaps: {len(report.data_gaps)}")
        if not ok:
            print(f"  EXPECTED primary in {expected}, GOT {primary}")

    print("\n" + "=" * 78)
    print(f"{len(anomalies) - failures}/{len(anomalies)} scenarios passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
