from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from agenomic.prompts.errors import AjsError, integrity_error

if TYPE_CHECKING:
    from agenomic.prompts.models import PromptVersionRecord

MAX_SAFE_INTEGER = 9007199254740991
MAX_JSON_DEPTH = 64
CONTENT_SCHEMA = "agenomic.prompt_content/v1"
MANIFEST_SCHEMA = "agenomic.prompt_manifest/v1"
ARTIFACT_SET_SCHEMA = "agenomic.prompt_artifact_set/v1"


def utf16_key(value: str) -> bytes:
    return value.encode("utf-16-be", "surrogatepass")


def sorted_keys(mapping: Mapping[str, Any]) -> list[str]:
    return sorted(mapping, key=utf16_key)


def pointer(base: str, key: object) -> str:
    return base + "/" + str(key).replace("~", "~0").replace("/", "~1")


def check_string(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AjsError("invalid_unicode", path) from error
    if "\0" in value:
        raise AjsError("nul_character", path)


def _number(value: float, path: str) -> int:
    if math.isnan(value) or math.isinf(value) or not value.is_integer():
        raise AjsError("float_not_allowed", path)
    if abs(value) > MAX_SAFE_INTEGER:
        raise AjsError("integer_out_of_range", path)
    return int(value)


def _ensure(value: Any, path: str, depth: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        number = operator.index(value)
        if abs(number) > MAX_SAFE_INTEGER:
            raise AjsError("integer_out_of_range", path)
        return int(number)
    if isinstance(value, float):
        return _number(value, path)
    if isinstance(value, str):
        check_string(value, path)
        return "".join((value,))
    if isinstance(value, (list, tuple)):
        if depth + 1 > MAX_JSON_DEPTH:
            raise AjsError("json_too_deep", path)
        return [_ensure(item, pointer(path, index), depth + 1) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if depth + 1 > MAX_JSON_DEPTH:
            raise AjsError("json_too_deep", path)
        for key in value:
            if not isinstance(key, str):
                raise AjsError("invalid_field_type", pointer(path, key))
        normalized: dict[str, Any] = {}
        for key in sorted_keys(value):
            check_string(key, pointer(path, key))
            normalized["".join((key,))] = _ensure(value[key], pointer(path, key), depth + 1)
        return normalized
    raise AjsError("invalid_field_type", path)


def ensure_ajs(value: Any, *, path: str = "") -> Any:
    return _ensure(value, path, 0)


def _ncf(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_ncf(item) for item in value) + "]"
    return (
        "{"
        + ",".join(
            json.dumps(key, ensure_ascii=False) + ":" + _ncf(value[key])
            for key in sorted_keys(value)
        )
        + "}"
    )


def canonical_json_v1(value: Any) -> str:
    return _ncf(ensure_ajs(value))


def normalized_json(value: Any) -> str:
    return _ncf(value)


def sha256_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_digest(document: Mapping[str, Any]) -> str:
    return sha256_digest(canonical_json_v1(document))


def content_digest(content: Mapping[str, Any]) -> str:
    return prompt_digest(content)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return prompt_digest(manifest)


def artifact_set(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SET_SCHEMA,
        "prompt_manifest_digest": document.get("prompt_manifest_digest"),
        "manifest": document.get("manifest"),
        "children": document.get("children"),
        "prompts": document.get("prompts"),
    }


def artifact_set_digest(document: Mapping[str, Any]) -> str:
    return prompt_digest(artifact_set(document))


def verify_version(record: PromptVersionRecord) -> None:
    ref = f"{record.prompt_id}:{record.version}"
    try:
        actual = content_digest(record.content)
    except AjsError as error:
        raise integrity_error(
            "prompt_digest_mismatch",
            f"content of {ref} is outside the JSON subset",
            ref=ref,
            expected=record.content_digest,
            reason=error.reason,
        ) from error
    if actual != record.content_digest:
        raise integrity_error(
            "prompt_digest_mismatch",
            f"content digest of {ref} does not match",
            ref=ref,
            expected=record.content_digest,
            actual=actual,
        )


def pin_ref(pin: Mapping[str, Any]) -> str:
    return f"{pin.get('prompt_id')}:{pin.get('version')}"


def closure_gaps(
    manifest: Mapping[str, Any], prompts: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], set[str]]:
    wanted: set[str] = set()
    missing: set[str] = set()
    pending: list[Mapping[str, Any]] = []
    slots = manifest.get("slots")
    if isinstance(slots, Mapping):
        pending.extend(pin for pin in slots.values() if isinstance(pin, Mapping))
    while pending:
        pin = pending.pop()
        ref = pin_ref(pin)
        if ref in wanted:
            continue
        wanted.add(ref)
        entry = prompts.get(ref)
        if not isinstance(entry, Mapping) or entry.get("content_digest") != pin.get(
            "content_digest"
        ):
            missing.add(ref)
            continue
        content = entry.get("content")
        fragments = content.get("fragments") if isinstance(content, Mapping) else None
        if isinstance(fragments, Mapping):
            pending.extend(p for p in fragments.values() if isinstance(p, Mapping))
    return wanted, missing


def verify_closure(
    manifest: Mapping[str, Any],
    versions: Iterable[PromptVersionRecord],
    expected: str,
) -> None:
    actual = manifest_digest(manifest)
    if actual != expected:
        raise integrity_error(
            "manifest_digest_mismatch",
            "manifest digest does not match",
            expected=expected,
            actual=actual,
        )
    prompts: dict[str, dict[str, Any]] = {}
    for record in versions:
        verify_version(record)
        prompts[f"{record.prompt_id}:{record.version}"] = {
            "content_digest": record.content_digest,
            "content": record.content,
        }
    _, missing = closure_gaps(manifest, prompts)
    if missing:
        raise integrity_error(
            "bundle_incomplete",
            "the prompt closure is incomplete",
            missing=sorted(missing),
            extra=[],
        )
