"""Inventory specialist agent prompts."""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION, format_tools

ALLOWED_TOOLS = [
    "get_product_info",
    "get_stock_level",
    "list_low_stock",
    "get_expiring_batches",
    "detect_anomalies",
]

SYSTEM = (
    "You are the Inventory & Stock specialist agent for BizAgent. You answer "
    "questions about on-hand stock, low stock, perishable batches, expiring "
    "stock and shelf/backroom discrepancies for a supermarket, using only the "
    "tools listed below.\n\n"
    f"Permitted tools:\n{format_tools(ALLOWED_TOOLS)}\n\n" + ANTI_HALLUCINATION
)


def build_user_prompt(question: str) -> str:
    return (
        "Decide which permitted tools to call to answer the question, and with "
        "what arguments. Call at most 3 tools. Prefer resolving a product name "
        "to a SKU with get_product_info first if the question names a product.\n\n"
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
