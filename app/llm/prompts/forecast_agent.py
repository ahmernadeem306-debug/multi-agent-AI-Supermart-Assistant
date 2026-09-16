"""Demand Forecasting specialist agent prompts.

Day 4: the trained forecasting model is wired in via the ``get_demand_forecast``
MCP tool; the agent prefers it and falls back to velocity/days-of-cover
reasoning if the tool returns nothing.
"""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION, format_tools

ALLOWED_TOOLS = [
    "get_product_info",
    "get_demand_forecast",
    "get_sales_history",
    "get_stock_level",
    "get_top_sellers",
]

SYSTEM = (
    "You are the Demand Forecasting specialist agent for BizAgent. You answer "
    "questions about future demand, stockout risk, reorder points and "
    "perishable expiry risk. Prefer the get_demand_forecast tool, which returns "
    "a per-day forecast plus projected stockout date, a reorder-point "
    "recommendation and expiry risk. Use get_sales_history / get_stock_level "
    "only to add context or when a forecast is unavailable. State which backend "
    "produced the forecast (xgboost or seasonal_naive baseline).\n\n"
    f"Permitted tools:\n{format_tools(ALLOWED_TOOLS)}\n\n" + ANTI_HALLUCINATION
)


def build_user_prompt(question: str) -> str:
    return (
        "Decide which permitted tools to call, and with what arguments, to "
        "answer the demand / stockout / reorder / expiry question. Call at most "
        "3 tools. For get_demand_forecast pass the SKU and a horizon in days "
        "(1-30; 7 for 'next week', 14 default). Resolve a product name to a SKU "
        "with get_product_info first if needed.\n\n"
        f"QUESTION: {question}\n\n"
        'Return JSON: {"tool_calls": [{"tool": "<name>", "arguments": {..}}], '
        '"reasoning": "<one sentence>"}. Return an empty list if no tool is needed.'
    )


def build_answer_prompt(question: str, tool_transcript: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"TOOL RESULTS:\n{tool_transcript}\n\n"
        "Answer using only these results. Report the projected stockout date "
        "and risk level, the reorder recommendation versus the current setting, "
        "and any expiry risk. Name the forecast backend used. "
        'Return JSON: {"answer": "<text>", "confidence": <0-1>, '
        '"used_tool_data": <boolean>}.'
    )
