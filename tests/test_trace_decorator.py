"""Tests for the @trace_agent_run decorator."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agenomic.exporters.base import Exporter
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule
from agenomic.trace.context import current_recorder
from agenomic.trace.decorator import trace_agent_run
from agenomic.types.envelope import TraceEnvelope
from agenomic.types.trace import ModelCall


class CollectExporter(Exporter):
    def __init__(self) -> None:
        self.envelopes: list[TraceEnvelope] = []

    def export(self, envelope: TraceEnvelope) -> None:
        self.envelopes.append(envelope)


def test_sync_captures_input_output() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)
    def handle(query: str) -> dict[str, Any]:
        return {"answer": query.upper()}

    handle("hello")
    assert len(exp.envelopes) == 1
    env = exp.envelopes[0]
    assert env.agent_id == "agent://acme/demo"
    assert env.input.payload_inline["args"] == ["hello"]
    assert env.final_output.payload_inline == {"answer": "HELLO"}
    assert env.error is None
    assert env.duration_ms is not None


def test_async_captures_input_output() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)
    async def handle(query: str) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"answer": query.lower()}

    asyncio.run(handle("HEY"))
    assert len(exp.envelopes) == 1
    assert exp.envelopes[0].final_output.payload_inline == {"answer": "hey"}


def test_async_exception_captured() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)
    async def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        asyncio.run(boom())
    assert len(exp.envelopes) == 1
    assert "RuntimeError: nope" in (exp.envelopes[0].error or "")


def test_async_with_async_exporter() -> None:
    class AsyncExporter:
        def __init__(self) -> None:
            self.envelopes: list[Any] = []

        async def export(self, envelope: Any) -> None:
            await asyncio.sleep(0)
            self.envelopes.append(envelope)

        def close(self) -> None: ...

    exp = AsyncExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)  # type: ignore[arg-type]
    async def handle() -> int:
        return 42

    asyncio.run(handle())
    assert len(exp.envelopes) == 1


def test_exception_recorded_and_reraised() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    assert len(exp.envelopes) == 1
    assert "ValueError: nope" in (exp.envelopes[0].error or "")


def test_recorder_visible_inside() -> None:
    seen: list[Any] = []

    @trace_agent_run("agent://acme/demo")
    def handle() -> None:
        rec = current_recorder()
        seen.append(rec)
        if rec is not None:
            rec.record_model_call(ModelCall(provider="openai", model="x"))
            rec.add_label("env", "test")

    handle()
    assert seen[0] is not None


def test_recorder_visible_inside_with_export() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://acme/demo", exporter=exp)
    def handle() -> None:
        rec = current_recorder()
        assert rec is not None
        rec.record_model_call(ModelCall(provider="openai", model="x"))
        rec.add_label("env", "test")

    handle()
    env = exp.envelopes[0]
    assert len(env.model_calls) == 1
    assert env.labels == {"env": "test"}


def test_redaction_applied_to_input() -> None:
    exp = CollectExporter()
    redaction = RedactionEngine([RedactionRule(path="kwargs.password", mode=RedactionMode.MASK)])

    @trace_agent_run("agent://a/b", exporter=exp, redaction=redaction)
    def login(*, user: str, password: str) -> dict[str, str]:
        return {"user": user}

    login(user="alice", password="hunter2")
    env = exp.envelopes[0]
    assert env.input.payload_inline["kwargs"]["password"] == "***"


def test_contextvar_resets_after_call() -> None:
    @trace_agent_run("agent://a/b")
    def handle() -> None:
        assert current_recorder() is not None

    assert current_recorder() is None
    handle()
    assert current_recorder() is None


def test_release_set_in_envelope() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://a/b", exporter=exp, release="v1.2.3")
    def handle() -> int:
        return 1

    handle()
    assert exp.envelopes[0].release == "v1.2.3"


def test_capture_input_false_redacts() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://a/b", exporter=exp, capture_input=False)
    def handle(secret: str) -> int:
        return len(secret)

    handle("hunter2")
    env = exp.envelopes[0]
    assert env.input.payload_inline == {"redacted": True}
    assert env.final_output.payload_inline == 7


def test_capture_output_false_redacts() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://a/b", exporter=exp, capture_output=False)
    def handle() -> dict[str, str]:
        return {"sensitive": "data"}

    handle()
    env = exp.envelopes[0]
    assert env.final_output.payload_inline == {"redacted": True}


def test_consecutive_calls_no_leakage() -> None:
    exp = CollectExporter()

    @trace_agent_run("agent://a/b", exporter=exp)
    def handle() -> None:
        rec = current_recorder()
        assert rec is not None

    handle()
    handle()
    handle()
    assert len(exp.envelopes) == 3
    ids = {e.run_id for e in exp.envelopes}
    assert len(ids) == 3
