"""Demand Forecasting specialist agent.

Day 3: velocity- and cover-based estimates only. The tool-calling shape is
identical to the other specialists, so Day 4 can append ``get_demand_forecast``
to ``allowed_tools`` without changing this class.
"""
from __future__ import annotations

from app.llm.prompts import forecast_agent as prompts
from app.agents.state import SpecialistAgent


class ForecastAgent(SpecialistAgent):
    name = "forecast"
    system = prompts.SYSTEM
    allowed_tools = prompts.ALLOWED_TOOLS
    plan_prompt = staticmethod(prompts.build_user_prompt)
    answer_prompt = staticmethod(prompts.build_answer_prompt)
