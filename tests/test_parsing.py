"""Structured-output parsing: fences, prose wrap, repair path, hard failure."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core.exceptions import LLMParseError
from app.llm.parsing import extract_json, generate_structured
from tests.fakes import ScriptedLLM


class _Model(BaseModel):
    name: str
    value: int


def _llm(*responses: str) -> ScriptedLLM:
    seq = iter(responses)
    return ScriptedLLM(lambda p, s: next(seq))


def test_extract_json_plain_fenced_and_prose():
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here it is:\n{"a": 1}\nHope that helps.') == '{"a": 1}'


def test_valid_json_first_try():
    result = generate_structured(_llm('{"name": "x", "value": 3}'), "p", None, _Model)
    assert result == _Model(name="x", value=3)


def test_fenced_json_first_try():
    result = generate_structured(_llm('```json\n{"name": "y", "value": 5}\n```'), "p", None, _Model)
    assert result.value == 5


def test_prose_wrapped_json_first_try():
    raw = 'Here is your object: {"name": "z", "value": 7} — done.'
    assert generate_structured(_llm(raw), "p", None, _Model).name == "z"


def test_malformed_then_repaired():
    llm = _llm("not json at all", '{"name": "ok", "value": 1}')
    result = generate_structured(llm, "p", None, _Model)
    assert result.name == "ok"
    assert len(llm.calls) == 2  # original + one repair


def test_irrecoverable_raises_llmparseerror():
    llm = _llm("garbage", "still garbage")
    with pytest.raises(LLMParseError):
        generate_structured(llm, "p", None, _Model)
    assert len(llm.calls) == 2  # exactly one repair attempt, then give up
