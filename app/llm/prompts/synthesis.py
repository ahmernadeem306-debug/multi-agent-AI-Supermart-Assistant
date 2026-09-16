"""Final synthesis prompt: merge specialist answers into one grounded reply."""
from __future__ import annotations

from app.llm.prompts import ANTI_HALLUCINATION

SYSTEM = (
    "You are the synthesis step of BizAgent. You combine the answers produced "
    "by the specialist agents into a single, coherent reply for a store "
    "manager. You do not add new facts; you only organise, reconcile and "
    "summarise what the specialists reported. If specialists disagree or an "
    "agent failed, say so honestly and state what remains unknown.\n\n"
    + ANTI_HALLUCINATION
)


def build_user_prompt(question: str, agent_sections: list[str], had_errors: bool) -> str:
    body = "\n\n".join(agent_sections) if agent_sections else "(no specialist produced an answer)"
    caveat = (
        "\n\nNote: at least one specialist agent failed; produce the best answer "
        "you can from the rest and flag the gap."
        if had_errors
        else ""
    )
    return (
        f"USER QUESTION: {question}\n\n"
        f"SPECIALIST ANSWERS:\n{body}{caveat}\n\n"
        "Write the final answer for the manager: direct, specific, and grounded "
        "only in the specialist answers above. Give a single overall confidence "
        'from 0 to 1. Return JSON: {"final_answer": "<text>", "confidence": <0-1>}.'
    )
