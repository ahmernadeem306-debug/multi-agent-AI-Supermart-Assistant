"""Specialist agents with a mocked LLM and a stub ToolProvider."""
from __future__ import annotations

import json

from app.agents.finance_agent import FinanceAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.inventory_agent import InventoryAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.sales_agent import SalesAgent
from app.rag.retriever import RetrievalResult
from tests.fakes import RaisingLLM, ScriptedLLM, StubRetriever, StubToolProvider


def _responder(tool_calls: list[dict], *, answer_used: bool = True):
    def _r(prompt: str, system: str | None) -> str:
        if '"tool_calls":' in prompt and "Return an empty list" in prompt:
            return json.dumps({"tool_calls": tool_calls, "reasoning": "stub"})
        if '"answer":' in prompt:
            return json.dumps({"answer": "Grounded answer.", "confidence": 0.7, "used_tool_data": answer_used})
        return "stub"
    return _r


def test_inventory_agent_selects_permitted_tool_and_passes_arguments():
    tools = StubToolProvider({"get_stock_level": {"summary": "SKU-1001: 12 on hand", "stock": {"on_hand_qty": 12}}})
    llm = ScriptedLLM(_responder([{"tool": "get_stock_level", "arguments": {"sku": "SKU-1001"}}]))
    out = InventoryAgent(llm, tools, max_tool_calls=3).run({"question": "stock for SKU-1001?"})

    assert tools.calls == [("get_stock_level", {"sku": "SKU-1001"})]
    output = out["agent_outputs"]["inventory"]
    assert output["error"] is None
    assert output["tool_calls"][0]["tool"] == "get_stock_level"
    assert output["tool_calls"][0]["summary"].startswith("SKU-1001")


def test_agent_drops_disallowed_tools_from_the_plan():
    tools = StubToolProvider({"get_margin_report": {"summary": "margin"}})
    # sales agent is asked to call a finance tool -> must be filtered out
    llm = ScriptedLLM(_responder([{"tool": "get_margin_report", "arguments": {"aisle": "Dairy"}}]))
    SalesAgent(llm, tools, max_tool_calls=3).run({"question": "margin?"})
    assert tools.calls == []  # get_margin_report is not in the sales allow-list


def test_agent_caps_tool_calls_at_max():
    tools = StubToolProvider({"get_sales_history": {"summary": "s"}, "get_top_sellers": {"summary": "t"},
                              "get_product_info": {"summary": "p"}, "detect_anomalies": {"summary": "a"}})
    plan = [
        {"tool": "get_sales_history", "arguments": {"sku": "SKU-1001", "start_date": "2026-01-01", "end_date": "2026-02-01"}},
        {"tool": "get_top_sellers", "arguments": {}},
        {"tool": "get_product_info", "arguments": {"sku": "SKU-1001"}},
        {"tool": "detect_anomalies", "arguments": {"kind": "stockout"}},
    ]
    SalesAgent(ScriptedLLM(_responder(plan)), tools, max_tool_calls=2).run({"question": "q"})
    assert len(tools.calls) == 2


def test_agent_handles_empty_tool_results_without_fabricating():
    tools = StubToolProvider({})  # every call returns an empty payload
    llm = ScriptedLLM(_responder([{"tool": "list_low_stock", "arguments": {"limit": 5}}], answer_used=False))
    out = FinanceAgent(llm, tools, max_tool_calls=3).run({"question": "anything low?"})
    output = out["agent_outputs"]["finance"]
    assert output["error"] is None
    assert output["used_tool_data"] is False


def test_agent_failure_is_isolated_not_raised():
    tools = StubToolProvider({"get_stock_level": {"summary": "ok"}})
    out = InventoryAgent(RaisingLLM(), tools, max_tool_calls=3).run({"question": "q"})
    output = out["agent_outputs"]["inventory"]
    assert output["error"] is not None
    assert out["errors"] and out["errors"][0].startswith("inventory:")


def test_forecast_agent_can_call_the_demand_forecast_tool():
    assert "get_demand_forecast" in ForecastAgent.allowed_tools


def test_policy_agent_returns_citations_from_retrieved_chunks():
    llm = ScriptedLLM(lambda p, s: json.dumps(
        {"answer": "Frozen goods: 7 days.", "confidence": 0.8, "used_tool_data": True}
    ))
    out = PolicyAgent(llm, StubRetriever()).run({"question": "return window for frozen?"})
    output = out["agent_outputs"]["policy"]
    assert output["citations"]
    assert output["citations"][0]["doc_title"] == "Returns and Refunds Policy"
    assert out["citations"] == output["citations"]


def test_policy_agent_says_so_when_nothing_retrieved():
    empty = RetrievalResult(query="q", found=False)
    out = PolicyAgent(ScriptedLLM(lambda p, s: "{}"), StubRetriever(result=empty)).run({"question": "q"})
    output = out["agent_outputs"]["policy"]
    assert "no supporting policy" in output["answer"].lower()
    assert output["citations"] == []


def test_policy_agent_degrades_when_retriever_raises():
    from app.core.exceptions import RetrievalError

    out = PolicyAgent(
        ScriptedLLM(lambda p, s: "{}"), StubRetriever(exc=RetrievalError("chroma empty"))
    ).run({"question": "q"})
    output = out["agent_outputs"]["policy"]
    assert output["error"] is not None
