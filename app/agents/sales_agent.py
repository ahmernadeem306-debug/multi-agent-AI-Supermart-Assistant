"""Sales specialist agent."""
from __future__ import annotations

from app.llm.prompts import sales_agent as prompts
from app.agents.state import SpecialistAgent


class SalesAgent(SpecialistAgent):
    name = "sales"
    system = prompts.SYSTEM
    allowed_tools = prompts.ALLOWED_TOOLS
    plan_prompt = staticmethod(prompts.build_user_prompt)
    answer_prompt = staticmethod(prompts.build_answer_prompt)
