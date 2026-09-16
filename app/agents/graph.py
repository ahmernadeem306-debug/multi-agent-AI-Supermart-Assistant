"""LangGraph assembly: supervisor -> specialist fan-out -> synthesis -> END.

The graph is compiled once at API startup, not per request. Specialist nodes
write into reducer-merged state fields so parallel branches never clobber
each other.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.mcp_client.adapter import ToolProvider
from app.agents.finance_agent import FinanceAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.inventory_agent import InventoryAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.sales_agent import SalesAgent
from app.agents.state import AgentState
from app.agents.supervisor import SPECIALISTS, Supervisor
from app.agents.synthesis import synthesize

logger = get_logger(__name__)


def _pick_specialists(state: AgentState) -> list[str]:
    agents = [a for a in state.get("plan", {}).get("agents", []) if a in SPECIALISTS]
    return agents or ["synthesis"]


def build_graph(
    llm: LLMProvider,
    tools: ToolProvider,
    retriever,
    settings: Settings | None = None,
):
    """Compile and return the orchestration graph."""
    settings = settings or get_settings()
    supervisor = Supervisor(llm)
    max_calls = settings.agent_max_tool_calls
    specialists = {
        "sales": SalesAgent(llm, tools, max_tool_calls=max_calls),
        "inventory": InventoryAgent(llm, tools, max_tool_calls=max_calls),
        "finance": FinanceAgent(llm, tools, max_tool_calls=max_calls),
        "forecast": ForecastAgent(llm, tools, max_tool_calls=max_calls),
        "policy": PolicyAgent(llm, retriever),
    }

    def supervisor_node(state: AgentState) -> dict:
        plan = supervisor.route(state["question"])
        logger.info("route_selected", route=plan.route, agents=plan.agents)
        return {
            "route": plan.route,
            "plan": plan.model_dump(),
            "needs_policy_check": plan.needs_policy_check,
        }

    def synthesis_node(state: AgentState) -> dict:
        return synthesize(state, llm, temperature=settings.llm_temperature_synthesis)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    for name, agent in specialists.items():
        graph.add_node(name, agent.run)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _pick_specialists,
        {**{name: name for name in specialists}, "synthesis": "synthesis"},
    )
    for name in specialists:
        graph.add_edge(name, "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_graph(compiled, run_id: str, question: str, settings: Settings | None = None) -> dict:
    """Invoke the compiled graph and return the final state as a plain dict."""
    settings = settings or get_settings()
    initial: AgentState = {
        "run_id": run_id,
        "question": question,
        "agent_outputs": {},
        "tool_calls": [],
        "citations": [],
        "errors": [],
        "status": "success",
    }
    try:
        final = compiled.invoke(
            initial, config={"recursion_limit": settings.graph_recursion_limit}
        )
        return dict(final)
    except Exception as exc:  # noqa: BLE001 - recursion / step-limit or unexpected
        logger.error("graph_run_failed", error=str(exc))
        return {
            **initial,
            "final_answer": f"The assistant stopped before finishing: {exc}",
            "confidence": 0.0,
            "status": "partial",
        }
