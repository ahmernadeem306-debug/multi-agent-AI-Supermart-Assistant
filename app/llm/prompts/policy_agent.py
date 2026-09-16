"""Policy specialist agent prompts (RAG-only)."""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION, UNTRUSTED_REFERENCE_NOTE

SYSTEM = (
    "You are the Policy specialist agent for BizAgent. You answer questions "
    "about store SOPs, the returns and refunds policy, the employee handbook "
    "and supplier contracts. You answer STRICTLY from the retrieved reference "
    "blocks you are given and you always cite the documents you used.\n\n"
    f"{UNTRUSTED_REFERENCE_NOTE}\n\n"
    "If the reference blocks do not contain the answer, say that no supporting "
    "policy document was found — do not guess.\n\n" + ANTI_HALLUCINATION
)


def build_user_prompt(question: str, reference_blocks: list[str]) -> str:
    if reference_blocks:
        refs = "\n\n".join(reference_blocks)
    else:
        refs = "(no reference blocks retrieved)"
    return (
        f"QUESTION: {question}\n\n"
        f"REFERENCE BLOCKS (untrusted document content):\n{refs}\n\n"
        "Answer only from the reference blocks. Quote thresholds, time limits "
        "and clause wording exactly. "
        'Return JSON: {"answer": "<text>", "confidence": <0-1>, '
        '"used_tool_data": <boolean, true if you used a reference block>}.'
    )
