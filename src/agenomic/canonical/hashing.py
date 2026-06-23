"""Canonical v0.3 trace hashing — a byte-exact port of the spec's
``scripts/trace-crypto.js`` so SDK-produced traces verify against the
``agenomic/v0.3`` schema's custom chain + Merkle checks.

The rules (spec ``schemas/v0.3/trace-event.schema.json``):

- ``event_hash = "blake3:" + BLAKE3(canonical_json(event_sans_event_hash) + prev_event_hash)``
- ``run_merkle_root = "blake3-merkle-v1:" + BLAKE3-Merkle(event_hash_0..n)``
  with RFC 0002 leaf/node domain separation and odd-node duplication.

``canonical_json`` here mirrors the spec verifier exactly: objects are emitted
with **sorted keys**, arrays in order, and scalars via the JSON grammar. It is
*not* the Python ``json`` module's default output (key order, spacing), so it is
implemented directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenomic.crypto.hashing import blake3_bytes, blake3_hex

#: The genesis ``prev_event_hash`` for the first event of a run.
GENESIS_PREV_EVENT_HASH = "blake3:" + "0" * 64

_LEAF_DOMAIN = b"AGENTLOCK-LEAF-v1\x00"
_NODE_DOMAIN = b"AGENTLOCK-NODE-v1\x00"
_EMPTY_DOMAIN = b"AGENOMIC-EMPTY-v1\x00"
_HASH_PREFIX = re.compile(r"^(blake3|sha256|blake3-merkle-v1):")


def canonical_json(value: Any) -> str:
    """Deterministic JSON string with **sorted object keys**, byte-identical to
    the spec verifier's ``canonicalJson``.

    Example:
        >>> canonical_json({"b": 1, "a": [True, None]})
        '{"a":[true,null],"b":1}'
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "null"  # JSON.stringify(NaN|Infinity) === "null"
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        items = (
            json.dumps(str(k), ensure_ascii=False) + ":" + canonical_json(value[k])
            for k in sorted(value.keys())
        )
        return "{" + ",".join(items) + "}"
    raise TypeError(f"canonical_json: unsupported type {type(value).__name__}")


def content_hash(payload: Any) -> str:
    """Content-addressed ``blake3:<hex>`` digest of a (redacted) payload.

    Example:
        >>> content_hash({"k": "v"}).startswith("blake3:")
        True
    """
    return "blake3:" + blake3_hex(canonical_json(payload).encode("utf-8"))


def event_hash(event_without_hash: dict[str, Any]) -> str:
    """``"blake3:" + BLAKE3(canonical_json(event) + prev_event_hash)``.

    ``event_without_hash`` must already carry ``prev_event_hash`` and must
    **not** carry ``event_hash``.
    """
    prev = event_without_hash["prev_event_hash"]
    body = canonical_json(event_without_hash) + prev
    return "blake3:" + blake3_hex(body.encode("utf-8"))


def _hash_to_bytes(h: str) -> bytes:
    return bytes.fromhex(_HASH_PREFIX.sub("", h))


def merkle_root(event_hashes: list[str]) -> str:
    """``blake3-merkle-v1:<hex>`` over the run's ``event_hash`` leaves
    (RFC 0002 domains, odd-node duplication). Empty runs hash a fixed domain.
    """
    if not event_hashes:
        return "blake3-merkle-v1:" + blake3_hex(_EMPTY_DOMAIN)
    nodes = [
        blake3_bytes(
            _LEAF_DOMAIN + str(i).rjust(16, "0").encode("ascii") + b"\x00" + _hash_to_bytes(h)
        )
        for i, h in enumerate(event_hashes)
    ]
    while len(nodes) > 1:
        nxt: list[bytes] = []
        for j in range(0, len(nodes), 2):
            left = nodes[j]
            right = nodes[j + 1] if j + 1 < len(nodes) else nodes[j]
            nxt.append(blake3_bytes(_NODE_DOMAIN + left + right))
        nodes = nxt
    return "blake3-merkle-v1:" + nodes[0].hex()


__all__ = [
    "GENESIS_PREV_EVENT_HASH",
    "canonical_json",
    "content_hash",
    "event_hash",
    "merkle_root",
]
