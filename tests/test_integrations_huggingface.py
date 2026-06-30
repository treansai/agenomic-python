"""Tests for the Hugging Face instrumentation integration."""

from __future__ import annotations

from typing import Any

import pytest

from agenomic.integrations.huggingface import (
    instrument_huggingface,
    trace_huggingface_call,
)
from agenomic.providers.huggingface import HuggingFaceClient, HuggingFaceConfig
from agenomic.trace.context import set_current_recorder
from agenomic.trace.recorder import TraceRecorder
from agenomic.types.trace import CallStatus


class _StubClient(HuggingFaceClient):
    """HuggingFaceClient whose inference methods are stubbed (no network)."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(HuggingFaceConfig(_token="hf_secret"))
        self._fail = fail

    def generate_text(self, model: str, prompt: str, parameters: Any = None) -> Any:
        if self._fail:
            raise RuntimeError("inference exploded hf_secret leak")
        return {"generated_text": "hi"}

    def embeddings(self, model: str, inputs: Any) -> Any:
        return [[0.1, 0.2]]


def test_instrument_records_generate_text() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        client = instrument_huggingface(_StubClient())
        out = client.generate_text("gpt2", "hello", {"temperature": 0.5})
    finally:
        set_current_recorder(None)
    assert out == {"generated_text": "hi"}
    assert len(rec.model_calls) == 1
    call = rec.model_calls[0]
    assert call.provider == "huggingface"
    assert call.model == "gpt2"
    assert call.temperature == 0.5
    assert call.prompt_hash
    assert call.output_hash
    assert call.status is CallStatus.SUCCESS


def test_instrument_records_error_without_token() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        client = instrument_huggingface(_StubClient(fail=True))
        with pytest.raises(RuntimeError):
            client.generate_text("gpt2", "hello")
    finally:
        set_current_recorder(None)
    assert len(rec.model_calls) == 1
    call = rec.model_calls[0]
    assert call.status is CallStatus.ERROR
    assert call.provider == "huggingface"
    # The recorded ModelCall never carries the token.
    assert "hf_secret" not in call.model_dump_json()


def test_instrument_records_embeddings() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        client = instrument_huggingface(_StubClient())
        client.embeddings("all-MiniLM-L6-v2", "hi")
    finally:
        set_current_recorder(None)
    assert rec.model_calls[0].model == "all-MiniLM-L6-v2"


def test_trace_huggingface_call_wraps_arbitrary_fn() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    calls: list[str] = []

    def fake_text_generation(*, text: str) -> str:
        calls.append(text)
        return "generated"

    try:
        out = trace_huggingface_call(
            fake_text_generation,
            model="gpt2",
            prompt="hello",
            parameters={"temperature": 0.2},
            text="hello",  # forwarded to fake_text_generation
        )
    finally:
        set_current_recorder(None)
    assert out == "generated"
    assert calls == ["hello"]
    assert rec.model_calls[0].provider == "huggingface"
    assert rec.model_calls[0].model == "gpt2"


def test_no_recorder_is_noop() -> None:
    set_current_recorder(None)
    client = instrument_huggingface(_StubClient())
    assert client.generate_text("gpt2", "hello") == {"generated_text": "hi"}
