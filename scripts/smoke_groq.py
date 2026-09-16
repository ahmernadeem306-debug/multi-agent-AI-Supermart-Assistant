"""Standalone smoke test for the Groq API key / connectivity.

Usage:
    python scripts/smoke_groq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.exceptions import ConfigError  # noqa: E402
from app.llm.groq_client import GroqProvider  # noqa: E402


def main() -> None:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc.message}")
        print("Set GROQ_API_KEY in your .env file (see .env.example) and try again.")
        sys.exit(1)

    provider = GroqProvider(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        timeout_seconds=settings.groq_timeout_seconds,
    )

    print(f"Calling Groq model '{settings.groq_model}'...")
    response = provider.generate("Say 'BizAgent is connected.' and nothing else.")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
