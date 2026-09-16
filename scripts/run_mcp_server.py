"""Launch the BizAgent MCP tool server over stdio.

    python scripts/run_mcp_server.py

Runs until the client closes stdin. Used directly for manual inspection and
spawned as a subprocess by app.mcp_client.client.MCPToolProvider.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import configure_logging  # noqa: E402
from app.mcp_server.server import build_server  # noqa: E402


def _warm_up() -> None:
    """Initialise heavy libraries on the main thread before the async server
    starts — importing xgboost inside an anyio worker thread can hang on
    Windows when stdio is piped."""
    try:
        import xgboost  # noqa: F401

        from app.config import get_settings
        from app.ml.backends import load_bundle

        load_bundle(get_settings().model_dir)
    except Exception:  # noqa: BLE001 - no model yet is fine (baseline is used)
        pass


def main() -> None:
    # Force all logging to stderr; stdout is the JSON-RPC transport.
    configure_logging("WARNING")
    _warm_up()
    build_server().run("stdio")



if __name__ == "__main__":
    main()
