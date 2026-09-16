"""Sales specialist agent prompts."""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION, format_tools

ALLOWED_TOOLS = [
    "get_product_info",
    "get_sales_history",
    "get_top_sellers",
    "detect_anomalies",
]

SYSTEM = (
    "You are the Sales specialist agent for BizAgent. You answer questions about "
    "sales history, units and revenue, velocity, top sellers and demand shifts "
    "for a supermarket, using only the tools listed below.\n\n"
    f"Permitted tools:\n{format_tools(ALLOWED_TOOLS)}\n\n" + ANTI_HALLUCINATION
)


def build_user_prompt(question: str) -> str:
    return (
        "Decide which permitted tools to call to answer the question, and with "
        "what arguments. Call at most 3 tools. For get_sales_history use ISO "
        "dates (YYYY-MM-DD); if the question says 'last N days', compute the "
        "range ending today.\n"
        "For store-wide questions with no specific SKU or aisle ('total sales', "
        "'how much did we sell', 'overall units or revenue' over the last N "
        "days), call get_top_sellers with arguments {\"days\": N, \"limit\": "
        "500} and NO aisle key, then aggregate its rows in your answer. Pass "
        "aisle only when the question names a real aisle. Return an empty list "
        "only for pure chit-chat with no data need.\n\n"
        f"QUESTION: {question}\n\n"
        'Return JSON: {"tool_calls": [{"tool": "<name>", "arguments": {..}}], '
        '"reasoning": "<one sentence>"}.'
    )


def build_answer_prompt(question: str, tool_transcript: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"TOOL RESULTS:\n{tool_transcript}\n\n"
        "Write a concise, grounded answer using only these results. If the "
        "question asks for a store-wide total, sum the units and revenue across "
        "the returned rows and report those totals. "
        'Return JSON: {"answer": "<text>", "confidence": <0-1>, '
        '"used_tool_data": <boolean>}.'
    )
