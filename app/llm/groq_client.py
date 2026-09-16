"""Groq implementation of LLMProvider, using the official `groq` SDK.

Nothing outside app/llm/ may import `groq` — agents and routes depend on the
LLMProvider protocol instead.
"""
from __future__ import annotations

import json

from groq import Groq
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.exceptions import BizAgentError, LLMError, LLMParseError, RateLimitError
from app.logging_config import get_logger

logger = get_logger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """True for network errors, 5xx and 429s that tenacity should retry on.

    Already-classified BizAgentError instances (raised by _call itself once
    it has interpreted the raw SDK exception) are never retried again.
    """
    if isinstance(exc, BizAgentError):
        return False
    message = str(exc).lower()
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(token in message for token in ("timeout", "unavailable", "503", "500", "429"))


def _is_rate_limit(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "quota" in message


class GroqProvider:
    """Groq implementation of the LLMProvider protocol."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        embedding_model: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self._client = Groq(api_key=api_key, timeout=timeout_seconds)
        self._model_name = model
        self._embedding_model = embedding_model
        self._timeout_seconds = timeout_seconds

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )
    def _call(self, prompt: str, system: str | None, temperature: float | None = None) -> str:
        try:
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            kwargs: dict = {"model": self._model_name, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitError(f"Groq rate limit exceeded: {exc}") from exc
            if _is_transient(exc):
                raise
            raise LLMError(f"Groq generation failed: {exc}") from exc

    def generate(
        self, prompt: str, system: str | None = None, *, temperature: float | None = None
    ) -> str:
        return self._call(prompt, system, temperature)

    def generate_structured(self, prompt: str, system: str | None, schema: type[BaseModel]) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema())
        full_prompt = (
            f"{prompt}\n\nRespond with ONLY valid JSON matching this schema, no markdown fences:\n{schema_json}"
        )
        raw = self.generate(full_prompt, system)
        parsed = self._try_parse(raw, schema)
        if parsed is not None:
            return parsed

        repair_prompt = (
            f"Your previous response was not valid JSON for this schema:\n{schema_json}\n\n"
            f"Previous response:\n{raw}\n\nReturn ONLY corrected, valid JSON matching the schema."
        )
        repaired_raw = self.generate(repair_prompt, system)
        parsed = self._try_parse(repaired_raw, schema)
        if parsed is not None:
            return parsed

        raise LLMParseError("Groq did not return valid structured output after one repair attempt.")

    @staticmethod
    def _try_parse(raw: str, schema: type[BaseModel]) -> BaseModel | None:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError(
            "Groq does not provide an embeddings API. Set EMBEDDING_BACKEND=local "
            "to use the offline sentence-transformers embedding backend instead."
        )
