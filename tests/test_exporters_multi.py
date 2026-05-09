"""Tests for MultiExporter."""

from __future__ import annotations

from typing import Any

from agenomic.exporters.base import Exporter
from agenomic.exporters.multi import MultiExporter
from agenomic.types.envelope import TraceEnvelope
from agenomic.types.trace import TraceInput, TraceOutput


class Boom(Exporter):
    def export(self, envelope: TraceEnvelope) -> None:
        raise RuntimeError("nope")


class Mem(Exporter):
    def __init__(self) -> None:
        self.items: list[Any] = []

    def export(self, envelope: TraceEnvelope) -> None:
        self.items.append(envelope)


def test_one_failure_does_not_block_others() -> None:
    a, b = Mem(), Mem()
    multi = MultiExporter(a, Boom(), b)
    env = TraceEnvelope(
        trace_id="t",
        run_id="r",
        agent_id="agent://a/b",
        input=TraceInput(payload_inline={}),
        final_output=TraceOutput(payload_inline={}),
    )
    multi.export(env)
    assert len(a.items) == 1
    assert len(b.items) == 1
