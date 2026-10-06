from __future__ import annotations

import enum
from collections import OrderedDict
from types import MappingProxyType

import pytest

from agenomic.canonical.hashing import canonical_json
from agenomic.prompts import (
    AjsError,
    PromptIntegrityError,
    PromptVersionRecord,
    canonical_json_v1,
    content_digest,
    ensure_ajs,
    manifest_digest,
    prompt_digest,
    verify_closure,
    verify_version,
)

KEYS = {"￿": 1, "\U0001f600": 2, "": 3, "a": 4}


class Color(enum.IntEnum):
    RED = 1


class Label(str):
    pass


def test_utf16_key_order() -> None:
    assert canonical_json_v1(KEYS) == '{"a":4,"\U0001f600":2,"":3,"￿":1}'


def test_shared_canonical_json_unchanged() -> None:
    assert canonical_json(KEYS) == '{"a":4,"":3,"￿":1,"\U0001f600":2}'
    assert canonical_json({"x": 1e-07}) == '{"x":1e-07}'
    assert canonical_json(KEYS) != canonical_json_v1(KEYS)


def test_float_refused() -> None:
    with pytest.raises(AjsError) as raised:
        canonical_json_v1({"a": [1, 0.5]})
    assert (raised.value.reason, raised.value.value_path) == ("float_not_allowed", "/a/1")
    for value in (float("nan"), float("inf")):
        with pytest.raises(AjsError):
            ensure_ajs(value)
    with pytest.raises(AjsError) as big:
        ensure_ajs(2.0**60)
    assert big.value.reason == "integer_out_of_range"
    with pytest.raises(AjsError) as huge:
        ensure_ajs(2**63)
    assert huge.value.reason == "integer_out_of_range"


def test_normalization() -> None:
    value = ensure_ajs(
        OrderedDict(
            [("b", (Color.RED, -0.0, 1.0)), ("a", Label("x")), ("m", MappingProxyType({"k": True}))]
        )
    )
    assert value == {"a": "x", "b": [1, 0, 1], "m": {"k": True}}
    assert type(value["a"]) is str
    assert type(value["b"][0]) is int
    assert canonical_json_v1({"s": '\n\u001f"\\ \u007f/'}) == '{"s":"\\n\\u001f\\"\\\\ \u007f/"}'


@pytest.mark.parametrize(
    ("value", "reason", "path"),
    [
        ({1: "a"}, "invalid_field_type", "/1"),
        ({"a/b": {"~": "\0"}}, "nul_character", "/a~1b/~0"),
        ({"\ud800": 1}, "invalid_unicode", "/\ud800"),
        (b"bytes", "invalid_field_type", ""),
    ],
)
def test_ajs_reasons(value: object, reason: str, path: str) -> None:
    with pytest.raises(AjsError) as raised:
        ensure_ajs(value)
    assert (raised.value.reason, raised.value.value_path) == (reason, path)


def test_depth_limit() -> None:
    nested: object = 1
    for _ in range(64):
        nested = [nested]
    ensure_ajs(nested)
    with pytest.raises(AjsError) as raised:
        ensure_ajs({"x": nested})
    assert raised.value.reason == "json_too_deep"


def _content(body: str) -> dict[str, object]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


def test_verify_version_and_closure() -> None:
    content = _content("Hello")
    record = PromptVersionRecord(
        prompt_id="prm_x", version=1, content_digest=content_digest(content), content=content
    )
    verify_version(record)
    with pytest.raises(PromptIntegrityError) as raised:
        verify_version(record.model_copy(update={"content": _content("Hello!")}))
    assert raised.value.details["ref"] == "prm_x:1"
    with pytest.raises(PromptIntegrityError) as ajs:
        verify_version(record.model_copy(update={"content": {"x": 0.5}}))
    assert ajs.value.details["reason"] == "float_not_allowed"
    manifest = {
        "schema": "agenomic.prompt_manifest/v1",
        "agent_id": "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c",
        "slots": {
            "a.b": {"prompt_id": "prm_x", "version": 1, "content_digest": record.content_digest}
        },
        "children": {},
    }
    digest = manifest_digest(manifest)
    assert digest == prompt_digest(manifest)
    verify_closure(manifest, [record], digest)
    with pytest.raises(PromptIntegrityError) as wrong:
        verify_closure(manifest, [record], "sha256:" + "0" * 64)
    assert wrong.value.code == "manifest_digest_mismatch"
    with pytest.raises(PromptIntegrityError) as incomplete:
        verify_closure(manifest, [], digest)
    assert incomplete.value.details["missing"] == ["prm_x:1"]
