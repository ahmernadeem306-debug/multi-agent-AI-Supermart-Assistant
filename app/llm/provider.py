"""LLMProvider protocol.

Agents and API routes depend on this protocol, never on a concrete Groq
class, so that swapping providers means writing one new implementation and
changing configuration only.
"""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProvider(Protocol):
    def generate(
        self, prompt: str, system: str | None = None, *, temperature: float | None = None
    ) -> str:
        """Generate free-text output for a prompt with an optional system instruction.

        ``temperature`` (0-1) is a hint: low for routing / argument synthesis,
        slightly higher for final narrative synthesis. Implementations may
        ignore it.
        """
        ...

    def generate_structured(
        self, prompt: str, system: str | None, schema: type[SchemaT]
    ) -> SchemaT:
        """Generate output validated against a Pydantic schema.

        Implementations should request JSON, parse it, validate it against
        `schema`, and attempt exactly one repair retry on failure before
        raising LLMParseError.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return an embedding vector for each input text."""
        ...
