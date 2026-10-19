"""Request hashing shared by the model-provider wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cbor2

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex

_PLAIN = (str, int, float, bool, bytes, type(None))


def _plain(value: Any) -> Any:
    if isinstance(value, _PLAIN):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return repr(value)


def hash_request(payload: dict[str, Any]) -> str:
    """Hash a provider request, replacing values CBOR cannot encode by their repr.

    A transport object such as an ``httpx.Timeout`` in the keyword arguments
    must not stop the request from being sent.

    Example:
        >>> class Opaque: ...
        >>> len(hash_request({"model": "m", "timeout": Opaque()}))
        64
    """
    try:
        return blake3_hex(canonical_cbor(payload))
    except cbor2.CBOREncodeError:
        return blake3_hex(canonical_cbor(_plain(payload)))
