"""Tests for canonical CBOR encoding."""

from __future__ import annotations

from agenomic.crypto.canonical import canonical_cbor, canonical_cbor_decode


def test_idempotent() -> None:
    payload = {"a": 1, "b": [1, 2, 3], "c": "x"}
    assert canonical_cbor(payload) == canonical_cbor(payload)


def test_key_ordering_independent() -> None:
    a = {"a": 2, "b": 1}
    b = {"b": 1, "a": 2}
    assert canonical_cbor(a) == canonical_cbor(b)


def test_nested_key_ordering() -> None:
    a = {"outer": {"z": 1, "a": 2}, "first": 0}
    b = {"first": 0, "outer": {"a": 2, "z": 1}}
    assert canonical_cbor(a) == canonical_cbor(b)


def test_round_trip() -> None:
    payload = {"k": [1, "two", b"\x00\x01"]}
    assert canonical_cbor_decode(canonical_cbor(payload)) == payload
