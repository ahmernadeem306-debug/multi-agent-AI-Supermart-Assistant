"""Supervisor routing: labelled fixture set + keyword fallback behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.supervisor import RoutingPlan, Supervisor
from tests.fakes import RaisingLLM, ScriptedLLM

_CASES = json.loads((Path(__file__).parent / "fixtures" / "routing_cases.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=[c["query"][:40] for c in _CASES])
def test_keyword_router_matches_labels(case):
    plan = Supervisor(RaisingLLM()).keyword_route(case["query"])
    assert plan.route == case["expected_route"]


def test_keyword_router_accuracy_on_fixture_set():
    sup = Supervisor(RaisingLLM())
    correct = sum(
        1 for c in _CASES if sup.keyword_route(c["query"]).route == c["expected_route"]
    )
    assert correct / len(_CASES) >= 0.9


def test_route_falls_back_to_keyword_when_llm_raises():
    plan = Supervisor(RaisingLLM()).route("Which SKUs are low on stock in Dairy?")
    assert plan.route == "inventory"
    assert "keyword" in plan.reasoning


def test_route_uses_llm_result_when_available():
    responder = lambda p, s: json.dumps(  # noqa: E731
        {"route": "finance", "agents": ["finance"], "reasoning": "llm", "needs_policy_check": True}
    )
    plan = Supervisor(ScriptedLLM(responder)).route("How did margin move?")
    assert plan.route == "finance"
    assert "policy" in plan.agents  # needs_policy_check adds the policy agent


def test_smalltalk_has_no_specialists():
    plan = Supervisor(RaisingLLM()).route("hello there")
    assert plan.route == "smalltalk"
    assert plan.agents == []


def test_rca_route_gets_default_agents_when_none_matched():
    plan = Supervisor(RaisingLLM()).keyword_route("Why did this happen on the floor?")
    assert plan.route == "rca"
    assert set(plan.agents) == {"sales", "inventory", "finance"}
