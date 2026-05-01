"""Tests for ATEP segment writer/reader."""
from __future__ import annotations

from pathlib import Path

import pytest
import ulid

from agentlock.atep.clock import Hlc
from agentlock.atep.event import AtepEvent, EventHeader, StreamId
from agentlock.atep.segment import SegmentReader, SegmentWriter
from agentlock.crypto.signing import SigningKey
from agentlock.exceptions import AtepError


def _make_event(sk: SigningKey, seq: int) -> AtepEvent:
    hdr = EventHeader(
        event_id=ulid.new().bytes,
        agent_id="agent://acme/demo",
        stream=StreamId.IDENTITY,
        stream_seq=seq,
        clock=Hlc(seq + 1, 0, 0),
        event_type="t",
        payload_schema_uri="atep://schemas/v1/t",
    )
    return AtepEvent.seal(hdr, {"i": seq}, sk)


def test_round_trip_100(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    path = tmp_path / "seg.atep"
    events = [_make_event(sk, i) for i in range(100)]
    with SegmentWriter(path) as w:
        for ev in events:
            w.append(ev)
    reader = SegmentReader(path)
    assert reader.event_count == 100
    assert reader.verify_merkle_root()
    read_events = list(reader.iter_events())
    assert len(read_events) == 100
    for original, read in zip(events, read_events, strict=True):
        assert read.causal_hash == original.causal_hash
        assert read.verify(sk.verifying_key())


def test_empty_segment(tmp_path: Path) -> None:
    path = tmp_path / "empty.atep"
    with SegmentWriter(path):
        pass
    reader = SegmentReader(path)
    assert reader.event_count == 0
    assert reader.verify_merkle_root()


def test_tamper_byte_fails_crc_or_merkle(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    path = tmp_path / "seg.atep"
    with SegmentWriter(path) as w:
        for i in range(5):
            w.append(_make_event(sk, i))
    data = bytearray(path.read_bytes())
    # Flip a byte inside an event frame (after the 76-byte header)
    data[100] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(AtepError):
        SegmentReader(path)


def test_invalid_magic_head(tmp_path: Path) -> None:
    path = tmp_path / "bad.atep"
    path.write_bytes(b"XXXX" + b"\x00" * 80)
    with pytest.raises(AtepError):
        SegmentReader(path)


def test_segment_too_short(tmp_path: Path) -> None:
    path = tmp_path / "tiny.atep"
    path.write_bytes(b"ATEP")
    with pytest.raises(AtepError):
        SegmentReader(path)


def test_first_last_hlc_recorded(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    path = tmp_path / "hlc.atep"
    with SegmentWriter(path) as w:
        for i in range(3):
            w.append(_make_event(sk, i))
    reader = SegmentReader(path)
    assert reader.first_hlc == Hlc(1, 0, 0)
    assert reader.last_hlc == Hlc(3, 0, 0)
