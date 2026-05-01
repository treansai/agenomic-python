"""BLAKE3 hashing primitives with domain separation.

Domain separators MUST match `agentlock-cli` and `agentlock-cloud`
byte-for-byte. Do not change without bumping schema versions.
"""

from __future__ import annotations

import blake3

LEAF_DOMAIN = b"AGENTLOCK-LEAF-v1\x00"
NODE_DOMAIN = b"AGENTLOCK-NODE-v1\x00"
ATEP_DOMAIN = b"ATEP-v1\x00"
ATTESTATION_DOMAIN = b"AGENTLOCK-ATTESTATION-v1\x00"


def blake3_hex(data: bytes) -> str:
    """Compute hex BLAKE3 digest of `data`.

    Example:
        >>> blake3_hex(b"hello")[:8]
        'ea8f163d'
    """
    return blake3.blake3(data).hexdigest()


def blake3_bytes(data: bytes) -> bytes:
    """Compute 32-byte BLAKE3 digest of `data`.

    Example:
        >>> len(blake3_bytes(b"hello"))
        32
    """
    return blake3.blake3(data).digest()


def hash_with_domain(domain: bytes, *parts: bytes) -> bytes:
    """``BLAKE3(domain || part1 || part2 || ...)``. Returns 32 bytes.

    Example:
        >>> len(hash_with_domain(ATEP_DOMAIN, b"x"))
        32
    """
    h = blake3.blake3()
    h.update(domain)
    for p in parts:
        h.update(p)
    return h.digest()
