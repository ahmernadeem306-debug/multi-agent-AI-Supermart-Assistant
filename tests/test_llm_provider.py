"""Unit tests for GroqProvider. The groq SDK is fully mocked —
no live API calls are made here."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import LLMError, RateLimitError
from app.llm.groq_client import GroqProvider


def _make_provider() -> GroqProvider:
    with patch("app.llm.groq_client.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()
        provider = GroqProvider(api_key="fake-key", model="llama-3.1-8b-instant")
    return provider


def _fake_completion(text: str) -> MagicMock:
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=text))]
    return fake_response


def test_generate_returns_text():
    provider = _make_provider()
    provider._client.chat.completions.create.return_value = _fake_completion("hello from groq")

    result = provider.generate("hi")

    assert result == "hello from groq"


def test_transient_error_is_retried_then_succeeds():
    provider = _make_provider()
    provider._client.chat.completions.create.side_effect = [
        TimeoutError("connection timed out"),
        _fake_completion("recovered"),
    ]

    result = provider.generate("hi")

    assert result == "recovered"
    assert provider._client.chat.completions.create.call_count == 2


def test_rate_limit_maps_to_rate_limit_error():
    provider = _make_provider()
    provider._client.chat.completions.create.side_effect = Exception("429 Too Many Requests: quota exceeded")

    with pytest.raises(RateLimitError):
        provider.generate("hi")


def test_non_transient_error_raises_llm_error():
    provider = _make_provider()
    provider._client.chat.completions.create.side_effect = ValueError("invalid request: malformed content")

    with pytest.raises(LLMError):
        provider.generate("hi")


def test_generate_passes_model_name():
    provider = _make_provider()
    provider._client.chat.completions.create.return_value = _fake_completion("ok")

    provider.generate("hi")

    _, kwargs = provider._client.chat.completions.create.call_args
    assert kwargs["model"] == "llama-3.1-8b-instant"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_embed_raises_llm_error():
    provider = _make_provider()

    with pytest.raises(LLMError):
        provider.embed(["text"])
