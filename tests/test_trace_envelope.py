"""Tests for envelope_builder."""

from __future__ import annotations

from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule
from agenomic.trace.envelope_builder import build_envelope
from agenomic.trace.recorder import TraceRecorder
from agenomic.types.trace import ModelCall


def test_build_with_redaction() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    rec.record_model_call(ModelCall(provider="openai", model="gpt"))
    eng = RedactionEngine([RedactionRule(path="kwargs.k", mode=RedactionMode.MASK)])
    env = build_envelope(
        rec,
        raw_input={"kwargs": {"k": "secret"}},
        raw_output={"r": 1},
        error=None,
        release=None,
        capture_input=True,
        capture_output=True,
        redaction=eng,
    )
    assert env.input.payload_inline["kwargs"]["k"] == "***"
    assert env.final_output.payload_inline == {"r": 1}
    assert len(env.model_calls) == 1


def test_build_without_capture() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    env = build_envelope(
        rec,
        raw_input={"kwargs": {"k": "x"}},
        raw_output={"r": 1},
        error=None,
        release=None,
        capture_input=False,
        capture_output=False,
        redaction=None,
    )
    assert env.input.payload_inline == {"redacted": True}
    assert env.final_output.payload_inline == {"redacted": True}


def test_build_records_error() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    env = build_envelope(
        rec,
        raw_input={},
        raw_output=None,
        error="ValueError: bad",
        release="v1",
        capture_input=True,
        capture_output=True,
        redaction=None,
    )
    assert env.error == "ValueError: bad"
    assert env.release == "v1"
