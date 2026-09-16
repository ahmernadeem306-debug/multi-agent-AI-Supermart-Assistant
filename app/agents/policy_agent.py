"""Policy specialist agent — RAG only, always cites its sources."""
from __future__ import annotations

from app.core.exceptions import BizAgentError
from app.llm.parsing import generate_structured
from app.llm.prompts import policy_agent as prompts
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.agents.state import AgentAnswer, AgentOutput, AgentState

logger = get_logger(__name__)


class PolicyAgent:
    """Answers strictly from retrieved policy chunks; never calls MCP tools."""

    name = "policy"

    def __init__(self, llm: LLMProvider, retriever, *, doc_type: str | None = None) -> None:
        self.llm = llm
        self.retriever = retriever
        self.doc_type = doc_type

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        citations: list[dict] = []
        try:
            result = self.retriever.retrieve(question, doc_type=self.doc_type)
            if not result.found:
                output = AgentOutput(
                    agent=self.name,
                    answer="No supporting policy document was found for this question.",
                    confidence=0.2,
                )
                return _emit(output)

            citations = [c.model_dump(mode="json") for c in result.citations]
            answer: AgentAnswer = generate_structured(
                self.llm,
                prompts.build_user_prompt(question, result.context_blocks),
                prompts.SYSTEM,
                AgentAnswer,
                temperature=0.1,
            )
            output = AgentOutput(
                agent=self.name,
                answer=answer.answer,
                confidence=answer.confidence,
                used_tool_data=answer.used_tool_data,
                citations=citations,
            )
        except BizAgentError as exc:
            logger.error("policy_agent_failed", error=exc.message)
            output = AgentOutput(
                agent=self.name,
                answer=f"The policy agent could not complete: {exc.message}",
                error=exc.message,
                citations=citations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("policy_agent_failed", error=str(exc))
            output = AgentOutput(
                agent=self.name,
                answer=f"The policy agent could not complete: {exc}",
                error=str(exc),
                citations=citations,
            )
        return _emit(output)


def _emit(output: AgentOutput) -> dict:
    return {
        "agent_outputs": {output.agent: output.model_dump(mode="json")},
        "citations": output.citations,
        "errors": [f"{output.agent}: {output.error}"] if output.error else [],
    }
