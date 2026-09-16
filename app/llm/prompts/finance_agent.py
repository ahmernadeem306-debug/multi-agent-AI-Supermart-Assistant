"""Finance specialist agent prompts."""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION, format_tools

ALLOWED_TOOLS = [
    "get_product_info",
    "get_margin_report",
    "get_shrinkage_report",
    "get_aisle_metrics",
]

SYSTEM = (
    "You are the Finance specialist agent for BizAgent. You answer questions "
    "about gross margin, revenue and COGS, shrinkage cost and margin erosion "
    "for a supermarket, using only the tools listed below.\n\n"
    f"Permitted tools:\n{format_tools(ALLOWED_TOOLS)}\n\n" + ANTI_HALLUCINATION
)


def build_user_prompt(question: str) -> str:
    return (
        "Decide which permitted tools to call to answer the question, and with "
        "what arguments. Call at most 3 tools. Use get_margin_report with a sku "
        "for one product, an aisle for an aisle, or neither for the whole "
        "store.\n\n"
        f"QUESTION: {question}\n\n"
        'Return JSON: {"tool_calls": [{"tool": "<name>", "arguments": {..}}], '
        '"reasoning": "<one sentence>"}. Return an empty list if no tool is needed.'
    )


def build_answer_prompt(question: str, tool_transcript: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"TOOL RESULTS:\n{tool_transcript}\n\n"
        "Write a concise, grounded answer using only these results. "
        'Return JSON: {"answer": "<text>", "confidence": <0-1>, '
        '"used_tool_data": <boolean>}.'
    )
