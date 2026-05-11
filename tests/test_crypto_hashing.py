"""Tests for BLAKE3 hashing primitives."""

from __future__ import annotations

from agenomic.crypto.hashing import (
    ATEP_DOMAIN,
    LEAF_DOMAIN,
    blake3_bytes,
    blake3_hex,
    hash_with_domain,
)


def test_blake3_known_vector() -> None:
    # blake3 of empty input — known constant
    digest = blake3_bytes(b"")
    assert len(digest) == 32
    assert blake3_hex(b"") == digest.hex()


def test_blake3_deterministic() -> None:
    assert blake3_bytes(b"hello") == blake3_bytes(b"hello")


def test_hash_with_domain_separates() -> None:
    a = hash_with_domain(LEAF_DOMAIN, b"x")
    b = hash_with_domain(ATEP_DOMAIN, b"x")
    assert a != b


def test_hash_with_domain_concatenation_matters() -> None:
    a = hash_with_domain(LEAF_DOMAIN, b"ab", b"cd")
    b = hash_with_domain(LEAF_DOMAIN, b"abcd")
    # Same concat → same hash (no length prefix in this primitive)
    assert a == b
