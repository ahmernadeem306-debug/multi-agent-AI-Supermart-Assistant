"""Supervisor agent: classify the query and plan the specialist fan-out.

Uses the LLM for routing; on any LLM failure it falls back to a keyword
router so the system degrades instead of dying.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.llm.parsing import generate_structured
from app.llm.prompts import supervisor as prompts
from app.llm.provider import LLMProvider
from app.logging_config import get_logger

logger = get_logger(__name__)

Route = Literal["sales", "inventory", "finance", "forecast", "policy", "rca", "smalltalk"]
SPECIALISTS = ("sales", "inventory", "finance", "forecast", "policy")

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "policy": ("policy", "sop", "procedure", "return", "refund", "handbook", "contract",
               "penalty", "clause", "rule", "compliance", "allowed", "tolerance"),
    "finance": ("margin", "revenue", "cogs", "profit", "shrinkage cost", "erosion", "cost of goods"),
    "forecast": ("forecast", "predict", "will stock out", "run out", "reorder point",
                 "days of cover", "next week", "demand next"),
    "sales": ("sales", "sold", "selling", "velocity", "top seller", "best seller",
              "revenue trend", "units sold", "demand shift"),
    "inventory": ("stock", "on hand", "low stock", "inventory", "batch", "expiring",
                  "expiry", "discrepancy", "shelf", "backroom", "out of stock"),
}
_RCA_MARKERS = ("why did", "why is", "root cause", "what caused", "explain why", "reason for")
_SMALLTALK = ("hello", "hi ", "hey", "thanks", "thank you", "how are you", "good morning")


class RoutingPlan(BaseModel):
    route: Route = "inventory"
    agents: list[str] = Field(default_factory=list)
    reasoning: str = ""
    needs_policy_check: bool = False


class Supervisor:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def route(self, question: str) -> RoutingPlan:
        try:
            plan = generate_structured(
                self.llm, prompts.build_user_prompt(question), prompts.SYSTEM,
                RoutingPlan, temperature=0.0,
            )
            return self._normalise(plan, question)
        except Exception as exc:  # noqa: BLE001 - degrade to keyword routing
            logger.warning("supervisor_llm_failed_using_keyword_router", error=str(exc))
            return self.keyword_route(question)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalise(plan: RoutingPlan, question: str) -> RoutingPlan:
        agents = [a for a in plan.agents if a in SPECIALISTS]
        if plan.route in SPECIALISTS and plan.route not in agents:
            agents.append(plan.route)
        if plan.route == "rca" and not agents:
            agents = ["sales", "inventory", "finance"]
        if plan.needs_policy_check and "policy" not in agents:
            agents.append("policy")
        if plan.route == "smalltalk":
            agents = []
        if plan.route not in ("smalltalk",) and not agents:
            agents = ["inventory"]
        plan.agents = agents
        return plan

    def keyword_route(self, question: str) -> RoutingPlan:
        text = f" {question.lower()} "
        if any(m in text for m in _SMALLTALK) and len(text.split()) <= 6:
            return RoutingPlan(route="smalltalk", agents=[], reasoning="keyword: greeting")

        scores = {
            route: sum(1 for kw in kws if kw in text) for route, kws in _KEYWORDS.items()
        }
        hit = [r for r, s in scores.items() if s > 0]
        is_rca = any(m in text for m in _RCA_MARKERS)

        if is_rca:
            agents = hit or ["sales", "inventory", "finance"]
            return RoutingPlan(
                route="rca", agents=agents, reasoning="keyword: root-cause phrasing",
                needs_policy_check=True,
            )
        if not hit:
            return RoutingPlan(route="inventory", agents=["inventory"], reasoning="keyword: default")
        primary = max(scores, key=scores.get)
        agents = sorted(hit, key=lambda r: scores[r], reverse=True)
        return RoutingPlan(
            route=primary, agents=agents, reasoning=f"keyword: matched {agents}",
            needs_policy_check="policy" in agents,
        )
