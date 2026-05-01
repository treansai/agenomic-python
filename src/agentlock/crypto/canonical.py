"""Canonical CBOR encoding (RFC 8949 §4.2).

Used for all hashing and signing of ATEP events and attestations.
Reproducible bit-for-bit on every platform.
"""
from __future__ import annotations

from typing import Any

import cbor2


def canonical_cbor(value: Any) -> bytes:
    """Encode `value` in canonical CBOR (RFC 8949 §4.2).

    Example:
        >>> canonical_cbor({"b": 1, "a": 2}) == canonical_cbor({"a": 2, "b": 1})
        True
    """
    return cbor2.dumps(value, canonical=True, datetime_as_timestamp=True)


def canonical_cbor_decode(data: bytes) -> Any:
    """Decode CBOR `data`.

    Example:
        >>> canonical_cbor_decode(canonical_cbor({"a": 1}))
        {'a': 1}
    """
    return cbor2.loads(data)
