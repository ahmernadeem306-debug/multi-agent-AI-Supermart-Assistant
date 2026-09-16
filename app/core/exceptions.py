"""Exception hierarchy shared across the application.

FastAPI route handlers catch BizAgentError subclasses and convert them into
structured JSON responses; raw provider exceptions must never reach the UI.
"""
from __future__ import annotations


class BizAgentError(Exception):
    """Base class for all application-specific errors."""

    error_code: str = "bizagent_error"
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(BizAgentError):
    """Raised when required configuration is missing or invalid."""

    error_code = "config_error"
    status_code = 500


class ToolExecutionError(BizAgentError):
    """Raised when an MCP/direct tool call fails."""

    error_code = "tool_execution_error"
    status_code = 500


class LLMError(BizAgentError):
    """Base class for LLM provider errors."""

    error_code = "llm_error"
    status_code = 503


class LLMParseError(LLMError):
    """Raised when structured output cannot be parsed after one repair retry."""

    error_code = "llm_parse_error"
    status_code = 502


class RateLimitError(LLMError):
    """Raised when the LLM provider reports a rate limit (HTTP 429)."""

    error_code = "rate_limit_error"
    status_code = 429


class RetrievalError(BizAgentError):
    """Raised when RAG retrieval fails."""

    error_code = "retrieval_error"
    status_code = 500


class ForecastError(BizAgentError):
    """Raised when a forecasting operation fails."""

    error_code = "forecast_error"
    status_code = 500


class DataNotFoundError(BizAgentError):
    """Raised when a requested entity (SKU, aisle, supplier, ...) does not exist."""

    error_code = "data_not_found"
    status_code = 404
