"""Tests for ATEP store."""

from __future__ import annotations

from pathlib import Path

import pytest
import ulid

from agenomic.atep.clock import Hlc
from agenomic.atep.event import AtepEvent, EventHeader, StreamId
from agenomic.atep.store import AtepStore
from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import AtepError


def _event(sk: SigningKey, stream: StreamId, seq: int) -> AtepEvent:
    hdr = EventHeader(
        event_id=ulid.new().bytes,
        agent_id="agent://acme/demo",
        stream=stream,
        stream_seq=seq,
        clock=Hlc(seq + 1, 0, 0),
        event_type="t",
        payload_schema_uri="atep://schemas/v1/t",
    )
    return AtepEvent.seal(hdr, {"i": seq}, sk)


def test_init_and_verify_all(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    for i in range(5):
        store.append_event(_event(sk, StreamId.CAPABILITY, i))
    for i in range(3):
        store.append_event(_event(sk, StreamId.KNOWLEDGE, i))
    report = store.verify_all(sk.verifying_key())
    assert report.ok, report.failures
    assert report.segments_checked == 8
    assert report.events_checked == 8


def test_root_hash_deterministic(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    for i in range(3):
        store.append_event(_event(sk, StreamId.CAPABILITY, i))
    h1 = store.compute_root_hash()
    # Reopen the store from disk
    store2 = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    h2 = store2.compute_root_hash()
    assert h1 == h2
    assert len(h1) == 32


def test_wrong_agent_id_rejects(tmp_path: Path) -> None:
    AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    with pytest.raises(AtepError):
        AtepStore.open_or_init(tmp_path, "agent://other/x")


def test_event_agent_id_must_match(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    hdr = EventHeader(
        event_id=ulid.new().bytes,
        agent_id="agent://other/x",
        stream=StreamId.IDENTITY,
        stream_seq=0,
        clock=Hlc(1, 0, 0),
        event_type="t",
        payload_schema_uri="atep://t",
    )
    ev = AtepEvent.seal(hdr, {}, sk)
    with pytest.raises(AtepError):
        store.append_event(ev)


def test_read_stream_only_returns_matching(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    store = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    store.append_event(_event(sk, StreamId.CAPABILITY, 0))
    store.append_event(_event(sk, StreamId.KNOWLEDGE, 0))
    cap = list(store.read_stream(StreamId.CAPABILITY))
    assert len(cap) == 1
    assert cap[0].header.stream == StreamId.CAPABILITY


def test_counters_resume_from_disk(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    s1 = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    p1 = s1.append_event(_event(sk, StreamId.IDENTITY, 0))
    assert "identity-00000001.atep" in p1.name
    s2 = AtepStore.open_or_init(tmp_path, "agent://acme/demo")
    p2 = s2.append_event(_event(sk, StreamId.IDENTITY, 1))
    assert "identity-00000002.atep" in p2.name
