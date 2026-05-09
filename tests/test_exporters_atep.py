"""Tests for the ATEP local exporter."""

from __future__ import annotations

from pathlib import Path

from agenomic.atep.event import StreamId
from agenomic.atep.store import AtepStore
from agenomic.crypto.signing import SigningKey
from agenomic.exporters.atep_local import AtepLocalExporter
from agenomic.types.envelope import TraceEnvelope
from agenomic.types.trace import TraceInput, TraceOutput


def _env(i: int) -> TraceEnvelope:
    return TraceEnvelope(
        trace_id=f"t{i}",
        run_id=f"r{i}",
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": i}),
        final_output=TraceOutput(payload_inline={"a": i}),
    )


def test_envelopes_become_signed_events(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    exp = AtepLocalExporter(store, sk)
    for i in range(5):
        exp.export(_env(i))
    events = list(store.read_stream(StreamId.INTERACTION))
    assert len(events) == 5
    report = store.verify_all(sk.verifying_key())
    assert report.ok, report.failures
    assert report.events_checked == 5


def test_parent_chain_is_linear(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    exp = AtepLocalExporter(store, sk)
    for i in range(3):
        exp.export(_env(i))
    events = list(store.read_stream(StreamId.INTERACTION))
    assert events[0].header.parents == []
    assert events[1].header.parents == [events[0].causal_hash]
    assert events[2].header.parents == [events[1].causal_hash]


def test_resumes_chain_across_processes(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store1 = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    AtepLocalExporter(store1, sk).export(_env(0))

    store2 = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    exp2 = AtepLocalExporter(store2, sk)
    exp2.export(_env(1))

    events = list(store2.read_stream(StreamId.INTERACTION))
    assert len(events) == 2
    assert events[1].header.parents == [events[0].causal_hash]
    assert events[1].header.stream_seq == 1
