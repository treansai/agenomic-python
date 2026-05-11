"""Tests for redaction engine."""

from __future__ import annotations

import copy

import pytest

from agenomic.exceptions import RedactionError
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule


def test_remove_leaf() -> None:
    e = RedactionEngine([RedactionRule(path="a.b", mode=RedactionMode.REMOVE)])
    assert e.apply({"a": {"b": 1, "c": 2}}) == {"a": {"c": 2}}


def test_mask_leaf() -> None:
    e = RedactionEngine([RedactionRule(path="a.b", mode=RedactionMode.MASK)])
    assert e.apply({"a": {"b": "secret"}}) == {"a": {"b": "***"}}


def test_hash_deterministic() -> None:
    e = RedactionEngine([RedactionRule(path="a", mode=RedactionMode.HASH)])
    a = e.apply({"a": "hello"})
    b = e.apply({"a": "hello"})
    assert a == b
    assert a["a"].startswith("hash:")
    assert len(a["a"]) == len("hash:") + 16


def test_truncate_string() -> None:
    e = RedactionEngine([RedactionRule(path="a", mode=RedactionMode.TRUNCATE, truncate_length=3)])
    assert e.apply({"a": "abcdefg"}) == {"a": "abc"}


def test_truncate_requires_length() -> None:
    with pytest.raises(ValueError):
        RedactionRule(path="a", mode=RedactionMode.TRUNCATE)


def test_truncate_non_string_passes_through() -> None:
    e = RedactionEngine([RedactionRule(path="a", mode=RedactionMode.TRUNCATE, truncate_length=2)])
    assert e.apply({"a": 12345}) == {"a": 12345}


def test_wildcard_segment_dict() -> None:
    e = RedactionEngine([RedactionRule(path="users.*.email", mode=RedactionMode.MASK)])
    out = e.apply({"users": {"u1": {"email": "a@b.c"}, "u2": {"email": "x@y.z"}}})
    assert out == {"users": {"u1": {"email": "***"}, "u2": {"email": "***"}}}


def test_wildcard_segment_list() -> None:
    e = RedactionEngine([RedactionRule(path="users.*.email", mode=RedactionMode.MASK)])
    out = e.apply({"users": [{"email": "a@b.c"}, {"email": "x@y.z"}]})
    assert out == {"users": [{"email": "***"}, {"email": "***"}]}


def test_recursive_wildcard() -> None:
    e = RedactionEngine([RedactionRule(path="**.email", mode=RedactionMode.MASK)])
    out = e.apply({"a": {"email": "x"}, "deep": {"more": {"email": "y"}}, "z": {"name": "n"}})
    assert out == {
        "a": {"email": "***"},
        "deep": {"more": {"email": "***"}},
        "z": {"name": "n"},
    }


def test_unknown_path_silent() -> None:
    e = RedactionEngine([RedactionRule(path="missing.path", mode=RedactionMode.REMOVE)])
    data = {"a": 1}
    assert e.apply(data) == {"a": 1}


def test_input_not_mutated() -> None:
    e = RedactionEngine([RedactionRule(path="a", mode=RedactionMode.MASK)])
    original = {"a": "secret"}
    snapshot = copy.deepcopy(original)
    e.apply(original)
    assert original == snapshot


def test_hash_unhashable_raises() -> None:
    e = RedactionEngine([RedactionRule(path="a", mode=RedactionMode.HASH)])

    # bytes is not JSON-serializable by default — but we use default=str so falls back.
    # Use a non-serializable object to force the failure.
    class NotJson:
        def __repr__(self) -> str:
            raise TypeError("nope")

    with pytest.raises(RedactionError):
        e.apply({"a": NotJson()})


def test_indexed_list_path() -> None:
    e = RedactionEngine([RedactionRule(path="items.0", mode=RedactionMode.MASK)])
    assert e.apply({"items": ["a", "b"]}) == {"items": ["***", "b"]}


def test_remove_via_wildcard_list() -> None:
    e = RedactionEngine([RedactionRule(path="items.*.secret", mode=RedactionMode.REMOVE)])
    out = e.apply({"items": [{"secret": 1, "ok": 2}, {"secret": 3, "ok": 4}]})
    assert out == {"items": [{"ok": 2}, {"ok": 4}]}
