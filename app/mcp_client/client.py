"""MCPToolProvider: talks to the real MCP server over stdio.

The MCP client stack is async; agents and tests are sync. This provider owns
a dedicated event loop on a background thread, opens one long-lived stdio
session to ``scripts/run_mcp_server.py``, and marshals each sync call onto
that loop. A failed handshake raises :class:`ToolExecutionError`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.exceptions import ToolExecutionError
from app.logging_config import get_logger
from app.mcp_client.adapter import ToolResult

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SERVER_SCRIPT = _REPO_ROOT / "scripts" / "run_mcp_server.py"


class MCPToolProvider:
    def __init__(
        self,
        server_script: str | Path | None = None,
        python_executable: str | None = None,
        start_timeout: float = 30.0,
        call_timeout: float = 60.0,
    ) -> None:
        self._call_timeout = call_timeout
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop")
        self._thread.start()

        params = StdioServerParameters(
            command=python_executable or sys.executable,
            args=[str(server_script or _DEFAULT_SERVER_SCRIPT)],
            cwd=str(_REPO_ROOT),
            env=os.environ.copy(),
        )
        try:
            self._submit(self._connect(params)).result(timeout=start_timeout)
        except Exception as exc:  # noqa: BLE001 - surfaced as ToolExecutionError
            self.close()
            raise ToolExecutionError(f"MCP stdio handshake failed: {exc}") from exc
        logger.info("mcp_provider_connected", server_script=str(server_script or _DEFAULT_SERVER_SCRIPT))

    # ------------------------------------------------------------------ plumbing
    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect(self, params: StdioServerParameters) -> None:
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    # -------------------------------------------------------------------- public
    def list_tools(self) -> list[str]:
        if self._session is None:
            raise ToolExecutionError("MCP session is not connected.")
        result = self._submit(self._session.list_tools()).result(timeout=self._call_timeout)
        return sorted(tool.name for tool in result.tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._session is None:
            raise ToolExecutionError("MCP session is not connected.")
        try:
            result = self._submit(
                self._session.call_tool(name, arguments or {})
            ).result(timeout=self._call_timeout)
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp_call_failed", tool=name, error=str(exc))
            raise ToolExecutionError(f"MCP call to '{name}' failed: {exc}") from exc
        return _to_tool_result(name, result)

    def close(self) -> None:
        if self._exit_stack is not None:
            try:
                self._submit(self._exit_stack.aclose()).result(timeout=10)
            except Exception:  # noqa: BLE001 - best effort on teardown
                pass
            self._exit_stack = None
            self._session = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "MCPToolProvider":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _to_tool_result(name: str, result: Any) -> ToolResult:
    data: dict = dict(result.structuredContent) if result.structuredContent else {}
    if not data and getattr(result, "content", None):
        text = getattr(result.content[0], "text", "") or ""
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, dict) else {"result": parsed}
        except (json.JSONDecodeError, TypeError):
            data = {"raw": text}
    if set(data.keys()) == {"result"} and isinstance(data["result"], dict):
        data = data["result"]
    is_error = bool(getattr(result, "isError", False))
    summary = data.get("summary") or (data.get("raw", "") if is_error else "")
    return ToolResult(
        name=name,
        data=data,
        summary=summary,
        is_error=is_error,
        error_code="tool_error" if is_error else None,
    )
