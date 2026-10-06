"""Arguments hashing that matches the Agenomic gateway byte for byte.

Protect binds a permit to ``arguments_hash`` computed by
``agenomic_tool_execution::canonical::hash_value`` in agenomic-cloud (rule
``agenomic.canon/v1``): object keys sorted bytewise, arrays kept in order, no
whitespace, then ``serde_json::to_string`` (serde_json 1.0.150 without the
``arbitrary_precision`` and ``preserve_order`` features), hashed with BLAKE3
and rendered ``blake3:<hex>``.

This differs from :func:`agenomic.canonical.canonical_json`, which follows
``JSON.stringify`` for floats (``1.0`` renders ``1`` there and ``1.0`` here),
so the SDK helper is not reused for this hash.

Known limit: serde_json parses floats with a fast path that is not always
correctly rounded (the ``float_roundtrip`` feature is off), so a float written
with 17 significant digits can land one ulp away from Python's value on the
server. The gateway stays authoritative; the adapter only compares its own
hashes with each other (authorized arguments against executed arguments).

Example:
    >>> canonical_json({"b": 1.0, "a": [True, None, "x"]})
    '{"a":[true,null,"x"],"b":1.0}'
    >>> arguments_hash({}).startswith("blake3:")
    True
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal

from agenomic.crypto.hashing import blake3_hex

CANONICALIZATION_RULE = "agenomic.canon/v1"

_I64_MIN = -(2**63)
_U64_MAX = 2**64 - 1


class CanonicalError(ValueError):
    """The value has no canonical JSON form the gateway would accept (NaN, lone surrogate, ...)."""


def _format_float(value: float) -> str:
    # serde_json 1.0.150 output: shortest round trip digits, fixed notation when
    # the decimal point position k is in [-4, 16], exponent form otherwise with an
    # explicit sign ("1e+21", "1.23e-6"). Fixed values always carry a fraction.
    if math.isnan(value) or math.isinf(value):
        raise CanonicalError("NaN and infinite numbers have no JSON form")
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    sign = "-" if value < 0 else ""
    digits_tuple, exponent = Decimal(repr(abs(value))).as_tuple()[1:]
    digits = "".join(str(d) for d in digits_tuple)
    assert isinstance(exponent, int)
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped.lstrip("0") or "0"
    n = len(digits)
    k = n + exponent
    if -4 <= k <= 16:
        if k <= 0:
            body = "0." + "0" * (-k) + digits
        elif k < n:
            body = digits[:k] + "." + digits[k:]
        else:
            body = digits + "0" * (k - n) + ".0"
        return sign + body
    mantissa = digits[0] + ("." + digits[1:] if n > 1 else "")
    exp = k - 1
    return f"{sign}{mantissa}e{'+' if exp >= 0 else '-'}{abs(exp)}"


def _format_string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise CanonicalError("string is not valid UTF-8 (lone surrogate)") from e
    # Python's escaping with ensure_ascii=False is serde_json's: the short forms
    # for \b \f \n \r \t, \u00XX (lower case hex) for the other C0 controls,
    # '"' and '\\' escaped, '/' and non-ASCII left as is.
    return json.dumps(value, ensure_ascii=False)


def _write(value: object, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        # Integers outside i64/u64 are parsed as f64 by serde_json.
        if _I64_MIN <= value <= _U64_MAX:
            out.append(str(value))
        else:
            try:
                as_float = float(value)
            except OverflowError as e:
                # serde_json refuses a number outside the f64 range.
                raise CanonicalError("integer is outside the f64 range") from e
            out.append(_format_float(as_float))
    elif isinstance(value, float):
        out.append(_format_float(value))
    elif isinstance(value, str):
        out.append(_format_string(value))
    elif isinstance(value, Mapping):
        encoded: list[tuple[bytes, str]] = []
        for key in value:
            if not isinstance(key, str):
                raise CanonicalError("object keys must be strings")
            try:
                encoded.append((key.encode("utf-8"), key))
            except UnicodeEncodeError as e:
                raise CanonicalError("object key is not valid UTF-8 (lone surrogate)") from e
        out.append("{")
        for i, (_, key) in enumerate(sorted(encoded)):
            if i:
                out.append(",")
            out.append(_format_string(key))
            out.append(":")
            _write(value[key], out)
        out.append("}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _write(item, out)
        out.append("]")
    else:
        raise CanonicalError(f"unsupported type {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Canonical JSON text of ``value`` as the gateway renders it before hashing.

    Raises :class:`CanonicalError` for values the gateway could not receive.

    Example:
        >>> canonical_json({"z": 1, "a": {"y": 1e21, "x": 0.00001}})
        '{"a":{"x":0.00001,"y":1e+21},"z":1}'
    """
    out: list[str] = []
    _write(value, out)
    return "".join(out)


def arguments_hash(value: object) -> str:
    """``blake3:<hex>`` over :func:`canonical_json`, the gateway's ``arguments_hash``.

    Example:
        >>> arguments_hash({"n": 1}) == arguments_hash({"n": 1})
        True
        >>> arguments_hash({"n": 1}) != arguments_hash({"n": "1"})
        True
    """
    return "blake3:" + blake3_hex(canonical_json(value).encode("utf-8"))


def schema_hash(schema: object) -> str:
    """Hash of a tool schema, used as the catalog ``schema_hash``.

    Example:
        >>> schema_hash({"name": "t"}).startswith("blake3:")
        True
    """
    return arguments_hash(schema)
