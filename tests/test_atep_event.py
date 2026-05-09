"""Tests for ATEP signed events."""

from __future__ import annotations

import ulid

from agenomic.atep.clock import Hlc
from agenomic.atep.event import AtepEvent, EventHeader, StreamId
from agenomic.crypto.signing import SigningKey


def _hdr(seq: int = 0, parents: list[bytes] | None = None) -> EventHeader:
    return EventHeader(
        event_id=ulid.new().bytes,
        agent_id="agent://acme/demo",
        stream=StreamId.IDENTITY,
        stream_seq=seq,
        clock=Hlc(seq + 1, 0, 0),
        parents=parents or [],
        event_type="t",
        payload_schema_uri="atep://schemas/v1/t",
    )


def test_seal_verify_roundtrip() -> None:
    sk = SigningKey.generate()
    ev = AtepEvent.seal(_hdr(), {"x": 1}, sk)
    assert len(ev.causal_hash) == 32
    assert ev.verify(sk.verifying_key())


def test_tampered_payload_fails_verify() -> None:
    sk = SigningKey.generate()
    ev = AtepEvent.seal(_hdr(), {"x": 1}, sk)
    ev.payload["x"] = 2
    assert not ev.verify(sk.verifying_key())


def test_same_inputs_same_causal_hash() -> None:
    hdr = _hdr()
    h1 = AtepEvent.compute_causal_hash(hdr, {"a": 1})
    h2 = AtepEvent.compute_causal_hash(hdr, {"a": 1})
    assert h1 == h2


def test_parent_order_does_not_affect_causal_hash() -> None:
    p1 = b"\x01" * 32
    p2 = b"\x02" * 32
    hdr_a = _hdr(parents=[p1, p2])
    hdr_b = _hdr(parents=[p2, p1])
    # Same event_id required for byte equality — set them equal
    hdr_b = hdr_a.model_copy(update={"parents": [p2, p1]})
    h_a = AtepEvent.compute_causal_hash(hdr_a, {"k": 1})
    h_b = AtepEvent.compute_causal_hash(hdr_b, {"k": 1})
    assert h_a == h_b


def test_wrong_signer_fails() -> None:
    sk1 = SigningKey.generate()
    sk2 = SigningKey.generate()
    ev = AtepEvent.seal(_hdr(), {"x": 1}, sk1)
    assert not ev.verify(sk2.verifying_key())
