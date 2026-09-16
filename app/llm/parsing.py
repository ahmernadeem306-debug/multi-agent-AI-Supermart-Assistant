"""Structured-output parsing with exactly one repair retry.

Agents call :func:`generate_structured` with an :class:`LLMProvider`; they
never talk to the SDK. The flow is: request JSON -> strip code fences ->
``json.loads`` -> validate against a Pydantic model. On failure, one repair
call that feeds the validation error back to the model. On a second failure,
raise :class:`LLMParseError` so the caller can degrade honestly.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import LLMParseError
from app.llm.provider import LLMProvider
from app.logging_config import get_logger

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json(text: str) -> str:
    """Return the most likely JSON payload from a raw model response.

    Strips Markdown code fences and, if the result is still prose-wrapped,
    slices from the first ``{`` / ``[`` to its matching closing bracket.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        pass
    start = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return cleaned
    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == opener:
            depth += 1
        elif cleaned[i] == closer:
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return cleaned[start:]


def _try_validate(raw: str, schema: type[SchemaT]) -> SchemaT | None:
    try:
        return schema.model_validate(json.loads(extract_json(raw)))
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None


def _schema_hint(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema())


def generate_structured(
    llm: LLMProvider,
    prompt: str,
    system: str | None,
    schema: type[SchemaT],
    *,
    temperature: float | None = 0.0,
) -> SchemaT:
    """Generate JSON, validate it against ``schema``, repair once, else raise."""
    instructed = (
        f"{prompt}\n\nRespond with ONLY valid JSON matching this schema "
        f"(no markdown fences, no commentary):\n{_schema_hint(schema)}"
    )
    raw = llm.generate(instructed, system, temperature=temperature)
    parsed = _try_validate(raw, schema)
    if parsed is not None:
        return parsed

    logger.warning("structured_parse_failed_repairing", schema=schema.__name__)
    repair = (
        "Your previous response was not valid JSON for the schema below.\n"
        f"Schema:\n{_schema_hint(schema)}\n\nPrevious response:\n{raw}\n\n"
        "Return ONLY corrected, valid JSON matching the schema."
    )
    parsed = _try_validate(llm.generate(repair, system, temperature=temperature), schema)
    if parsed is not None:
        return parsed

    raise LLMParseError(
        f"Model did not return valid JSON for {schema.__name__} after one repair attempt."
    )
