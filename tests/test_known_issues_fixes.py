"""Regression tests for the cheap items of the Python SDK known-issues list."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agenomic.cli.__main__ import main
from agenomic.exporters.base import Exporter
from agenomic.integrations._hashing import hash_request
from agenomic.integrations.openai import instrument_openai
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule
from agenomic.trace.context import set_current_recorder
from agenomic.trace.decorator import trace_agent_run
from agenomic.trace.recorder import TraceRecorder
from agenomic.types.envelope import TraceEnvelope


class Opaque:
    def __repr__(self) -> str:
        return "Opaque()"


class NoCopy:
    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        raise TypeError("cannot copy")


def test_hash_request_tolerates_unencodable_values() -> None:
    plain = hash_request({"model": "m"})
    assert hash_request({"model": "m", "timeout": Opaque()}) != plain
    assert hash_request({"model": "m", "timeout": Opaque()}) == hash_request(
        {"model": "m", "timeout": Opaque()}
    )
    assert hash_request({"a": [Opaque(), (1, 2)], "b": {3: Opaque()}})


def test_hash_request_keeps_hashes_of_encodable_payloads() -> None:
    from agenomic.crypto.canonical import canonical_cbor
    from agenomic.crypto.hashing import blake3_hex

    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert hash_request(payload) == blake3_hex(canonical_cbor(payload))


def test_instrumented_openai_sends_the_request_with_an_opaque_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    create = MagicMock(
        return_value=SimpleNamespace(id="r", system_fingerprint="fp", choices=[], usage=None)
    )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    recorder = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(recorder)
    try:
        client = instrument_openai(fake)
        client.chat.completions.create(model="m", messages=[], timeout=Opaque())
    finally:
        set_current_recorder(None)
    create.assert_called_once()
    assert len(recorder.model_calls) == 1


class _Collect(Exporter):
    def __init__(self) -> None:
        self.envelopes: list[TraceEnvelope] = []

    def export(self, envelope: TraceEnvelope) -> None:
        self.envelopes.append(envelope)


def test_uncopyable_argument_does_not_break_a_successful_run_with_redaction() -> None:
    exporter = _Collect()
    engine = RedactionEngine([RedactionRule(path="x", mode=RedactionMode.MASK)])

    @trace_agent_run("agent://a/b", exporter=exporter, redaction=engine)
    def handle(lock: Any) -> str:
        return "done"

    assert handle(NoCopy()) == "done"
    assert exporter.envelopes[0].input.payload_inline == {"redacted": True}
    assert exporter.envelopes[0].final_output.payload_inline == "done"


def test_uncopyable_thread_lock_argument() -> None:
    exporter = _Collect()
    engine = RedactionEngine([RedactionRule(path="x", mode=RedactionMode.MASK)])

    @trace_agent_run("agent://a/b", exporter=exporter, redaction=engine)
    def handle(lock: Any) -> int:
        return 1

    assert handle(threading.Lock()) == 1
    assert len(exporter.envelopes) == 1


def test_benchmark_serve_accepts_agenomic_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AGENOMIC_BASE_URL", raising=False)
    monkeypatch.setenv("AGENOMIC_ENDPOINT", "https://cloud.example")
    rc = main(["benchmark", "serve", "--agent", "a", "--bridge", "nope"])
    assert rc != 0
    assert "--bridge must be" in capsys.readouterr().err


def test_benchmark_serve_still_requires_a_base_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AGENOMIC_BASE_URL", raising=False)
    monkeypatch.delenv("AGENOMIC_ENDPOINT", raising=False)
    with pytest.raises(SystemExit):
        main(["benchmark", "serve", "--agent", "a", "--bridge", "nope"])
    assert "--base-url" in capsys.readouterr().err
