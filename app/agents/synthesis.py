"""Synthesis node: merge the specialist answers into one grounded reply."""
from __future__ import annotations

from pydantic import BaseModel

from app.core.exceptions import LLMError, LLMParseError
from app.llm.parsing import generate_structured
from app.llm.prompts import synthesis as prompts
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.agents.state import AgentState

logger = get_logger(__name__)


class _Synth(BaseModel):
    final_answer: str
    confidence: float = 0.5


_SMALLTALK_SYSTEM = (
    "You are BizAgent, a friendly supermarket operations assistant. Reply briefly "
    "to small talk and offer to help with sales, inventory, finance, forecasting "
    "or policy questions."
)


def synthesize(state: AgentState, llm: LLMProvider, *, temperature: float = 0.35) -> dict:
    outputs: dict[str, dict] = state.get("agent_outputs", {})
    route = state.get("route", "")

    if route == "smalltalk" and not outputs:
        try:
            reply = llm.generate(state["question"], _SMALLTALK_SYSTEM, temperature=0.4)
        except (LLMError, LLMParseError):
            reply = "Hello. I can help with sales, inventory, finance, forecasting and policy questions."
        return {"final_answer": reply.strip(), "confidence": 0.9, "status": "success"}

    if not outputs:
        return {
            "final_answer": "No specialist agent produced an answer for this question.",
            "confidence": 0.0,
            "status": "error",
        }

    sections: list[str] = []
    had_errors = False
    confidences: list[float] = []
    for name, out in outputs.items():
        sections.append(f"[{name.upper()} AGENT]\n{out.get('answer', '').strip()}")
        if out.get("error"):
            had_errors = True
        else:
            confidences.append(float(out.get("confidence", 0.0)))

    try:
        synth = generate_structured(
            llm,
            prompts.build_user_prompt(state["question"], sections, had_errors),
            prompts.SYSTEM,
            _Synth,
            temperature=temperature,
        )
        return {
            "final_answer": synth.final_answer.strip(),
            "confidence": round(synth.confidence, 3),
            "status": "partial" if had_errors else "success",
        }
    except (LLMError, LLMParseError) as exc:
        logger.warning("synthesis_llm_failed_concatenating", error=str(exc))
        merged = "\n\n".join(sections)
        fallback_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        return {
            "final_answer": (
                "I could not synthesise a single answer, so here is each agent's "
                f"finding:\n\n{merged}"
            ),
            "confidence": fallback_conf,
            "status": "partial",
        }
