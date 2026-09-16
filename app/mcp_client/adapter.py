"""ToolProvider abstraction.

Agents (from Day 3) depend on the :class:`ToolProvider` protocol, never on a
concrete transport. Two implementations exist:

* :class:`DirectToolProvider` - calls the tool functions in-process.
* :class:`app.mcp_client.client.MCPToolProvider` - spawns the real MCP server
  over stdio and calls tools through an MCP session.

Both return identical :class:`ToolResult` objects for identical arguments
(proven by ``tests/test_tool_provider_parity.py``). The active provider is
chosen by ``TOOL_PROVIDER`` in configuration (default ``mcp``).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.core.exceptions import BizAgentError, ToolExecutionError
from app.logging_config import get_logger
from app.mcp_server.server import TOOL_INPUT_MODELS, TOOL_REGISTRY

logger = get_logger(__name__)

TOOL_NAMES: list[str] = sorted(TOOL_REGISTRY)


class ToolResult(BaseModel):
    """Uniform result of a tool call, whatever the transport."""

    name: str
    data: dict = {}
    summary: str = ""
    is_error: bool = False
    error_code: str | None = None


@runtime_checkable
class ToolProvider(Protocol):
    def list_tools(self) -> list[str]:
        """Return the names of the available tools."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool by name with a dict of arguments."""

    def close(self) -> None:
        """Release any resources (subprocess, session)."""


def _error_result(name: str, code: str, message: str) -> ToolResult:
    return ToolResult(name=name, data={"error": message}, summary=message, is_error=True, error_code=code)


class DirectToolProvider:
    """Calls the tool functions directly, with no subprocess or MCP session."""

    def list_tools(self) -> list[str]:
        return list(TOOL_NAMES)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return _error_result(name, "unknown_tool", f"Unknown tool '{name}'.")
        try:
            output = fn(**(arguments or {}))
        except (ValidationError, ValueError) as exc:
            logger.warning("tool_validation_error", tool=name, error=str(exc))
            return _error_result(name, "validation_error", str(exc))
        except BizAgentError as exc:
            logger.warning("tool_error", tool=name, error_code=exc.error_code, error=exc.message)
            return _error_result(name, exc.error_code, exc.message)
        data = output.model_dump(mode="json")
        return ToolResult(name=name, data=data, summary=data.get("summary", ""))

    def close(self) -> None:  # nothing to release
        return None


def get_tool_provider(settings: Settings | None = None) -> ToolProvider:
    """Return the configured ToolProvider (``mcp`` or ``direct``).

    If ``TOOL_PROVIDER=mcp`` but the stdio server fails to start and
    ``TOOL_PROVIDER_FALLBACK=true``, fall back to :class:`DirectToolProvider`
    with a loud log line so the demo stays alive.
    """
    settings = settings or get_settings()
    choice = settings.tool_provider.lower()
    if choice == "direct":
        return DirectToolProvider()
    if choice == "mcp":
        from app.mcp_client.client import MCPToolProvider

        try:
            return MCPToolProvider()
        except Exception as exc:  # noqa: BLE001 - fallback is deliberate
            if settings.tool_provider_fallback:
                logger.error("mcp_provider_failed_falling_back_to_direct", error=str(exc))
                return DirectToolProvider()
            raise ToolExecutionError(f"Failed to start the MCP tool provider: {exc}") from exc
    raise ToolExecutionError(f"Unknown TOOL_PROVIDER '{settings.tool_provider}'.")


def tool_catalogue() -> list[dict]:
    """Machine-readable catalogue (name, description, JSON input schema).

    Embedded into the Day 3 agent prompts so the LLM knows what each tool
    does and how to shape its arguments.
    """
    catalogue = []
    for name, fn in TOOL_REGISTRY.items():
        model = TOOL_INPUT_MODELS[name]
        catalogue.append(
            {
                "name": name,
                "description": (fn.__doc__ or "").strip(),
                "input_schema": model.model_json_schema(),
            }
        )
    return catalogue
