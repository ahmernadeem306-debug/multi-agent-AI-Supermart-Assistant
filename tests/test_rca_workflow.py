"""RCA workflow with mocked LLM and real (direct) tools over the seeded data."""
from __future__ import annotations

import json

import pytest

from app.core.exceptions import ToolExecutionError
from app.mcp_client.adapter import DirectToolProvider
from app.rag.retriever import RetrievalResult
from app.agents.rca_schemas import RcaRequest
from app.agents.rca_workflow import RcaWorkflow, _validate_causes
from app.agents.rca_schemas import EvidenceItem, RankedCause
from tests.fakes import RaisingLLM, ScriptedLLM, StubRetriever

pytestmark = pytest.mark.usefixtures("seeded_db")

_EXPECTED = {
    "A": ("SKU-1035", "supply_delay"),
    "B": ("SKU-1002", "shrinkage_theft"),
    "C": ("SKU-1001", "expiry_writeoff"),
    "D": ("SKU-1003", {"reorder_point", "demand_shift"}),
}


@pytest.fixture()
def retriever():
    return StubRetriever(
        RetrievalResult(
            query="policy",
            found=True,
            chunks=["A late delivery of 4-7 days incurs a 12% credit and a written explanation."],
            citations=[
                __import__("app.rag.retriever", fromlist=["Citation"]).Citation(
                    doc_title="Supplier Agreement — FreshFarm Produce Co.",
                    source_path="data/knowledge/supplier_contract_freshfarm.md",
                    chunk_index=3,
                    section="Delivery performance and late-delivery penalties",
                    score=0.55,
                    snippet="A delivery received 1-3 days late incurs a credit of 5%",
                )
            ],
        )
    )


def test_all_evidence_steps_run(retriever, planted_anomalies):
    wf = RcaWorkflow(RaisingLLM(), DirectToolProvider(), retriever)
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    joined = " ".join(report.checks_performed).lower()
    for token in ("sales", "stock", "shrinkage", "purchase order", "forecast", "sop"):
        assert token in joined
    # the tool trace shows every evidence family was exercised
    tools_used = {t.get("tool") for t in report.tool_trace}
    assert {"get_sales_history", "get_stock_level", "get_shrinkage_report",
            "get_open_purchase_orders", "get_demand_forecast"} <= tools_used


@pytest.mark.parametrize("case_id", ["A", "B", "C", "D"])
def test_each_planted_anomaly_gets_correct_primary_cause(retriever, planted_anomalies, case_id):
    sku, expected = _EXPECTED[case_id]
    wf = RcaWorkflow(RaisingLLM(), DirectToolProvider(), retriever)
    report = wf.run(RcaRequest(sku=sku, anomaly_type="auto"))
    assert report.ranked_causes, "expected at least one ranked cause"
    primary = report.ranked_causes[0].category
    if isinstance(expected, set):
        assert primary in expected
    else:
        assert primary == expected
    # HARD RULE: every cause carries resolvable evidence
    evidence_ids = {e.id for e in report.evidence}
    for cause in report.ranked_causes:
        assert cause.evidence_ids
        assert set(cause.evidence_ids) <= evidence_ids


def test_no_anomaly_control_sku(retriever):
    wf = RcaWorkflow(RaisingLLM(), DirectToolProvider(), retriever)
    report = wf.run(RcaRequest(sku="SKU-1010", anomaly_type="auto"))
    assert report.status == "no_anomaly"
    assert report.ranked_causes == []
    assert report.checks_performed  # it still reports what it checked


def test_cause_with_unresolvable_evidence_is_dropped():
    pool = {"real.evidence": EvidenceItem(id="real.evidence", source="tool:x", title="t", detail="d").model_dump()}
    causes = [
        RankedCause(cause="valid", category="other", confidence=0.5, evidence_ids=["real.evidence"]),
        RankedCause(cause="hallucinated", category="other", confidence=0.9, evidence_ids=["made.up.id"]),
        RankedCause(cause="partly", category="other", confidence=0.6, evidence_ids=["real.evidence", "ghost"]),
    ]
    kept = _validate_causes(causes, pool)
    assert [c.cause for c in kept] == ["valid", "partly"]
    assert kept[1].evidence_ids == ["real.evidence"]  # ghost id stripped


def test_llm_hallucinated_cause_is_dropped_end_to_end(retriever):
    def responder(prompt: str, system: str | None) -> str:
        return json.dumps(
            {
                "summary": "Test summary.",
                "ranked_causes": [
                    {"cause": "Invented cause", "category": "other", "confidence": 0.99,
                     "evidence_ids": ["totally.made.up"], "contributing_factor": False}
                ],
                "recommended_actions": [],
                "data_gaps": [],
            }
        )

    wf = RcaWorkflow(ScriptedLLM(responder), DirectToolProvider(), retriever)
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    assert all("Invented cause" not in c.cause for c in report.ranked_causes)
    assert report.ranked_causes[0].category == "supply_delay"  # deterministic primary survives


def test_failing_evidence_step_becomes_a_data_gap_not_an_exception(retriever):
    class PartlyBrokenTools(DirectToolProvider):
        def call_tool(self, name, arguments):
            if name == "get_demand_forecast":
                raise ToolExecutionError("forecast backend exploded")
            return super().call_tool(name, arguments)

    wf = RcaWorkflow(RaisingLLM(), PartlyBrokenTools(), retriever)
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    assert any("forecast" in g.lower() for g in report.data_gaps)
    assert report.ranked_causes  # RCA still produced a report
    assert report.status == "partial"


def test_retriever_failure_degrades_policy_evidence(planted_anomalies):
    wf = RcaWorkflow(
        RaisingLLM(),
        DirectToolProvider(),
        StubRetriever(exc=ToolExecutionError("chroma directory deleted")),
    )
    report = wf.run(RcaRequest(sku="SKU-1035", anomaly_type="auto"))
    assert any("policy" in g.lower() for g in report.data_gaps)
    assert report.policy_findings == []
    assert report.ranked_causes  # non-policy evidence still drives the report
