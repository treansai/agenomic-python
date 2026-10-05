from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from agenomic.prompts.bundle import PromptBundle
from agenomic.prompts.digest import canonical_json_v1, prompt_digest
from agenomic.prompts.errors import (
    AjsError,
    PromptIntegrityError,
    PromptRefError,
    PromptRenderError,
    PromptTemplateError,
)
from agenomic.prompts.models import ManagedPromptVersion, PromptVersionRecord
from agenomic.prompts.refs import PromptUri, PromptVersionRef, parse_execution_ref, parse_prompt_ref
from agenomic.prompts.render import (
    RenderedMessage,
    render_content,
    tokenize,
    validate_content,
)
from agenomic.prompts.secrets import is_secret_shaped_key, scan, scrub, scrub_json

VECTORS = Path(__file__).parent / "fixtures" / "spec_vectors"
CONSUMER = "python"
SUITES = ("render", "template", "digest", "ref", "secrets")
LOCAL_FILES = ("MANIFEST.json", "SPEC_VECTORS.lock")
VECTOR_WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _vector_files() -> list[Path]:
    return sorted(
        path
        for path in VECTORS.rglob("*.json")
        if path.parent != VECTORS and path.name not in LOCAL_FILES
    )


def _python_vectors() -> list[dict[str, Any]]:
    vectors = [_load(path) for path in _vector_files()]
    return [vector for vector in vectors if CONSUMER in vector["consumers"]]


def test_lock_pins_the_manifest() -> None:
    lock = _load(VECTORS / "SPEC_VECTORS.lock")
    assert set(lock) == {"spec_commit", "manifest_sha256"}
    assert re.fullmatch(r"[0-9a-f]{40}", lock["spec_commit"])
    assert lock["manifest_sha256"] == _sha256(VECTORS / "MANIFEST.json")


def test_vendored_files_equal_the_manifest() -> None:
    manifest = _load(VECTORS / "MANIFEST.json")
    assert manifest["renderer_version"] == "1"
    assert manifest["secret_patterns"] == "agenomic-secrets/1"
    vendored = {
        path.relative_to(VECTORS).as_posix(): _sha256(path)
        for path in VECTORS.rglob("*")
        if path.is_file() and path.relative_to(VECTORS).as_posix() not in LOCAL_FILES
    }
    assert vendored == manifest["files"]


def test_every_suite_is_known() -> None:
    directories = {path.name for path in VECTORS.iterdir() if path.is_dir()}
    assert directories <= set(SUITES)
    for path in _vector_files():
        vector = _load(path)
        assert vector["suite"] == path.parent.name
        assert path.name.startswith(vector["id"] + "-")


def _subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items()
        )
    return bool(expected == actual) and type(expected) is type(actual)


def _assert_matches(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    assert actual["ok"] is expected["ok"], actual
    if not expected["ok"]:
        for key, value in expected["error"].items():
            if key == "item" or key == "details":
                assert _subset(value, actual["error"].get(key)), actual["error"]
            else:
                assert actual["error"].get(key) == value, actual["error"]
        return
    for key, value in expected.items():
        if key in ("ok", "langchain_parity"):
            continue
        if key == "warnings":
            assert len(actual["warnings"]) == len(value), actual["warnings"]
            for want, got in zip(value, actual["warnings"], strict=True):
                assert _subset(want, got), actual["warnings"]
            continue
        assert actual[key] == value, (key, actual[key])


def _fragment_source(fragments: dict[str, Any]) -> Any:
    def lookup(prompt_id: str, version: int, digest: str) -> Optional[dict[str, Any]]:
        entry = fragments.get(f"{prompt_id}:{version}")
        return entry if isinstance(entry, dict) else None

    return lookup


def _message(item: Any) -> Any:
    return item.to_dict() if isinstance(item, RenderedMessage) else item


def _run_render(data: dict[str, Any]) -> dict[str, Any]:
    options = data["options"]
    try:
        result = render_content(
            data["content"],
            data["variables"],
            fragments=_fragment_source(data["fragments"]),
            strict=options["strict"],
            history=options["history"],
            allow_duplicate_system=options["allow_duplicate_system"],
            secret_policy=options["secret_policy"],
            expect_kind={"text": "text", "messages": "chat"}.get(data["method"]),
        )
    except PromptRenderError as error:
        return {"ok": False, "error": {"code": error.code, "item": error.details["errors"][0]}}
    return {
        "ok": True,
        "kind": result.kind,
        "text": result.text,
        "messages": None if result.messages is None else [_message(m) for m in result.messages],
        "expanded_template": result.expanded_template,
        "rendered_document": result.rendered_document,
        "rendered_hash": result.rendered_hash,
        "warnings": [warning.to_dict() for warning in result.warnings],
    }


def _version_from_vector(data: dict[str, Any]) -> ManagedPromptVersion:
    fragments = {
        key: PromptVersionRecord(
            prompt_id=key.split(":")[0],
            version=int(key.split(":")[1]),
            prompt_kind=entry.get("prompt_kind"),
            content_digest=prompt_digest(entry["content"]),
            content=entry["content"],
        )
        for key, entry in data["fragments"].items()
    }
    record = PromptVersionRecord(
        prompt_id="prm_vector",
        version=1,
        prompt_kind=data["content"]["kind"],
        content_digest=prompt_digest(data["content"]),
        content=data["content"],
    )
    return ManagedPromptVersion.from_record(
        record, workspace_id=VECTOR_WORKSPACE, lookup=lambda pid, v: fragments.get(f"{pid}:{v}")
    )


def _check_langchain_parity(data: dict[str, Any], expected: dict[str, Any]) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, convert_to_messages

    from agenomic.integrations.langchain_prompts import to_langchain

    template = to_langchain(_version_from_vector(data))
    if expected["kind"] == "text":
        assert template.format(**data["variables"]) == expected["text"]
        return
    got = template.invoke(data["variables"]).to_messages()
    want = convert_to_messages(expected["messages"])
    assert len(got) == len(want)
    roles = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
    for produced, wanted in zip(got, want, strict=True):
        assert type(produced) is type(wanted), roles.get(type(produced))
        assert produced.content == wanted.content


def _run_template(data: dict[str, Any]) -> dict[str, Any]:
    if "template" in data:
        try:
            tokens = tokenize(data["template"])
        except PromptTemplateError as error:
            return {"ok": False, "error": {"code": error.code, "item": error.details["errors"][0]}}
        from agenomic.prompts.render import source

        return {"ok": True, "tokens": [t.to_dict() for t in tokens], "source": source(tokens)}
    report = validate_content(data["content"], fragments=_fragment_source(data["fragments"]))
    try:
        report.raise_for_errors()
    except PromptTemplateError as error:
        return {"ok": False, "error": {"code": error.code, "item": error.details["errors"][0]}}
    return {
        "ok": True,
        "validation": {
            "variables": report.variables,
            "warnings": [warning.to_dict() for warning in report.warnings],
            "content_digest": report.content_digest,
        },
    }


def _run_digest(data: dict[str, Any]) -> dict[str, Any]:
    if data["operation"] == "bundle_load":
        try:
            bundle = PromptBundle.load(
                data["bundle"],
                expected_workspace_id=data["expected_workspace_id"],
                expected_agent_id=data["expected_agent_id"],
                expected_bundle_digest=data["expected_bundle_digest"],
                now=datetime(2026, 10, 5, tzinfo=timezone.utc),
            )
        except PromptIntegrityError as error:
            return {"ok": False, "error": {"code": error.code, "details": error.details}}
        return {
            "ok": True,
            "prompt_bundle_digest": bundle.prompt_bundle_digest,
            "prompt_manifest_digest": bundle.prompt_manifest_digest,
            "prompt_refs": bundle.prompt_refs,
            "managed_slots": bundle.slots(),
        }
    document = data["document"]
    try:
        canonical = canonical_json_v1(document)
        digest = prompt_digest(document)
    except AjsError as error:
        return {
            "ok": False,
            "error": {"item": {"code": error.reason, "value_path": error.value_path}},
        }
    projection = data.get("projection")
    if projection is not None:
        origin = projection["from"]
        if projection["rule"] == "content":
            assert origin["content"] == document
            assert origin["content_digest"] == digest
        else:
            assert {k: v for k, v in origin.items() if k != "plan_digest"} == document
            assert origin["plan_digest"] == digest
    return {
        "ok": True,
        "document_type": document["schema"],
        "canonical": canonical,
        "digest": digest,
    }


def _run_ref(data: dict[str, Any]) -> dict[str, Any]:
    workspace_id = data["workspace_id"]
    try:
        if data["context"] == "management":
            ref: Any = parse_prompt_ref(data["ref"], workspace_id=workspace_id, allow_bare_id=True)
        else:
            ref = parse_execution_ref(data["ref"], workspace_id=workspace_id)
    except PromptRefError as error:
        failure: dict[str, Any] = {"code": error.code}
        if error.reason is not None:
            failure["reason"] = error.reason
        return {"ok": False, "error": failure}
    if isinstance(ref, str):
        return _ref_result("prompt_id", ref, None, None, None, ref, None)
    if isinstance(ref, PromptUri):
        version_ref = (
            str(ref.to_version_ref(workspace_id)) if workspace_id == ref.workspace_id else None
        )
        return _ref_result(
            "uri", ref.prompt_id, ref.version, None, ref.workspace_id, str(ref), version_ref
        )
    if isinstance(ref, PromptVersionRef):
        return _ref_result("version", ref.prompt_id, ref.version, None, None, str(ref), str(ref))
    return _ref_result("alias", ref.prompt_id, None, ref.alias, None, str(ref), None)


def _ref_result(
    form: str,
    prompt_id: str,
    version: Optional[int],
    alias: Optional[str],
    workspace_id: Optional[str],
    canonical: str,
    version_ref: Optional[str],
) -> dict[str, Any]:
    return {
        "ok": True,
        "form": form,
        "prompt_id": prompt_id,
        "version": version,
        "alias": alias,
        "workspace_id": workspace_id,
        "canonical": canonical,
        "version_ref": version_ref,
    }


def _run_secrets(data: dict[str, Any]) -> dict[str, Any]:
    if data["operation"] == "scan":
        return {
            "ok": True,
            "findings": [finding.to_dict() for finding in scan(data["text"])],
            "scrubbed": scrub(data["text"]),
        }
    if data["operation"] == "secret_shaped":
        return {"ok": True, "secret_shaped": [is_secret_shaped_key(key) for key in data["keys"]]}
    return {"ok": True, "scrubbed": scrub_json(data["value"])}


RUNNERS = {
    "render": _run_render,
    "template": _run_template,
    "digest": _run_digest,
    "ref": _run_ref,
    "secrets": _run_secrets,
}


@pytest.mark.parametrize("vector", _python_vectors(), ids=lambda vector: vector["id"])
def test_vector(vector: dict[str, Any]) -> None:
    expected = vector["expected"]
    actual = RUNNERS[vector["suite"]](vector["input"])
    _assert_matches(expected, actual)
    if vector["suite"] == "render" and expected["ok"] and expected["langchain_parity"]:
        _check_langchain_parity(vector["input"], expected)


def test_every_python_vector_ran() -> None:
    vectors = _python_vectors()
    counts = {suite: sum(1 for v in vectors if v["suite"] == suite) for suite in SUITES}
    assert counts == {"render": 66, "template": 69, "digest": 28, "ref": 54, "secrets": 14}
    assert (
        sum(1 for v in vectors if v["suite"] == "render" and v["expected"].get("langchain_parity"))
        > 0
    )
