from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from prompt_fakes import AGENT, WORKSPACE, release_with_child, seeded_engine

from agenomic import Client
from agenomic.cli import __main__ as cli
from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts.digest import prompt_digest
from agenomic.prompts.discovery import scan_paths
from agenomic.prompts.errors import PromptImportError, PromptIntegrityError
from agenomic.prompts.importer import (
    apply_import,
    build_apply_request,
    build_import_request,
    check_report,
    complete_content,
    default_decisions,
    load_prompt_file,
    load_prompts_file,
    load_yaml,
    new_idempotency_key,
    plan_summary,
    upload_report,
    verify_plan,
)

FIXTURES = Path(__file__).parent / "fixtures" / "prompt_imports"
SOURCES = Path(__file__).parent / "fixtures" / "prompt_sources"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def reason(error: pytest.ExceptionInfo[PromptImportError]) -> str:
    assert error.value.code == "prompt_import_invalid"
    return str(error.value.details["errors"][0]["code"])


def resign(plan: dict[str, Any]) -> dict[str, Any]:
    plan.pop("plan_digest", None)
    plan["plan_digest"] = prompt_digest(plan)
    return plan


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("a: 1\n1: b\n", "invalid_field_type"),
        ("base: &b {x: 1}\nother:\n  <<: *b\n", "yaml_alias_unsupported"),
        ("a: .inf\n", "float_not_allowed"),
        ("a: -.5e3\n", "float_not_allowed"),
        ("a: [unclosed\n", "yaml_syntax_error"),
        ("a: !custom x\n", "yaml_tag_unsupported"),
        ("a: 9007199254740992\n", "integer_out_of_range"),
    ],
)
def test_yaml_profile_refusals(text: str, code: str) -> None:
    with pytest.raises(PromptImportError) as raised:
        load_yaml(text)
    assert reason(raised) == code


def test_yaml_profile_scalars() -> None:
    document = load_yaml(
        "﻿null_word: null\ntilde: ~\nempty:\nflag: false\nnumber: -12\n"
        "hex: 0x1F\noctal: 0o7\nquoted: '1.5'\nword: Off\n"
    )
    assert document == {
        "null_word": None,
        "tilde": None,
        "empty": None,
        "flag": False,
        "number": -12,
        "hex": "0x1F",
        "octal": "0o7",
        "quoted": "1.5",
        "word": "Off",
    }


def test_yaml_needs_the_extra_and_json_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(PromptImportError) as raised:
        load_yaml("a: 1\n")
    assert raised.value.code == "yaml_support_not_installed"
    document = load_prompts_file('{"schema": "agenomic.prompts_file/v1", "prompts": []}')
    assert document == {"schema": "agenomic.prompts_file/v1", "prompts": []}


def test_json_documents_follow_the_same_subset() -> None:
    with pytest.raises(PromptImportError) as duplicate:
        load_prompts_file('{"schema": "agenomic.prompts_file/v1", "prompts": [], "prompts": []}')
    assert reason(duplicate) == "duplicate_key"
    with pytest.raises(PromptImportError) as floating:
        load_prompts_file(
            '{"schema": "agenomic.prompts_file/v1", "prompts": [], "agent_id": 1.5}',
            format="json",
        )
    assert reason(floating) == "float_not_allowed"
    with pytest.raises(PromptImportError) as broken:
        load_prompts_file("{not json", format="json")
    assert reason(broken) == "invalid_json"
    with pytest.raises(PromptImportError) as encoding:
        load_prompts_file(b"\xff\xfe", format="yaml")
    assert reason(encoding) == "invalid_unicode"


def test_family_file_digest_matches_the_spec_plan() -> None:
    document = load_prompts_file(FIXTURES / "family.yaml")
    plan = fixture("prompts-file-plan.json")
    assert plan["source"] == {"kind": "prompts_file", "digest": prompt_digest(document)}
    assert verify_plan(plan) is plan
    assert document["prompts"][1]["content"]["fragments"] == {
        "safety": {"prompt_id": "prm_support_safety"}
    }


def prompts_file(*prompts: dict[str, Any]) -> dict[str, Any]:
    return {"schema": "agenomic.prompts_file/v1", "prompts": list(prompts)}


def prompt(prompt_id: str, **fragments: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "kind": "fragment",
        "name": prompt_id,
        "content": {"kind": "text", "body": "x", "variables": {}, "fragments": fragments},
    }


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ({"schema": "agenomic.prompts_file/v2", "prompts": []}, "unsupported_schema"),
        ({"schema": "agenomic.prompts_file/v1", "prompts": {}}, "invalid_field_type"),
        (prompts_file(prompt("prm_a"), prompt("prm_a")), "duplicate_prompt_id"),
        (prompts_file(prompt("prm_a", f={"prompt_id": "prm_elsewhere"})), "fragment_not_found"),
        (
            prompts_file(
                prompt("prm_a", f={"prompt_id": "prm_b"}),
                prompt("prm_b", f={"prompt_id": "prm_c"}),
                prompt("prm_c", f={"prompt_id": "prm_a"}),
            ),
            "fragment_cycle",
        ),
        (prompts_file({"prompt_id": "prm_a", "content": "x"}), "missing_field"),
    ],
)
def test_prompts_file_checks(document: dict[str, Any], code: str) -> None:
    with pytest.raises(PromptImportError) as raised:
        load_prompts_file(json.dumps(document))
    assert reason(raised) == code


def test_prompts_file_accepts_pinned_and_shared_fragments() -> None:
    pinned = {"prompt_id": "prm_other", "version": 2}
    document = prompts_file(
        prompt("prm_a", f={"prompt_id": "prm_b"}, g=pinned),
        prompt("prm_b"),
        prompt("prm_c", f={"prompt_id": "prm_b"}),
    )
    assert load_prompts_file(json.dumps(document)) == document


def test_complete_content_fills_only_the_five_defaults() -> None:
    content = complete_content({"kind": "text", "body": "Hi {name}", "variables": {}})
    assert content == {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": "Hi {name}",
        "variables": {},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


def test_prompt_file_checks_kind_and_digest(tmp_path: Path) -> None:
    content = complete_content(
        {
            "kind": "text",
            "body": "Hi {name}",
            "variables": {"name": {"type": "string", "required": True}},
        }
    )
    document = {
        "schema": "agenomic.prompt_file/v1",
        "prompt_id": "prm_greeter",
        "kind": "text",
        "content": content,
        "content_digest": prompt_digest(content),
    }
    path = tmp_path / "greeter.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_prompt_file(path)["content"] == content
    yaml_path = tmp_path / "bare.yaml"
    yaml_path.write_text(
        "schema: agenomic.prompt_content/v1\ntemplate_format: agenomic-fstring/v1\n"
        "renderer_version: '1'\nkind: text\nbody: Hi\nvariables: {}\npartials: {}\n"
        "output_contract: null\nfragments: {}\n",
        encoding="utf-8",
    )
    assert load_prompt_file(yaml_path)["content"]["body"] == "Hi"
    with pytest.raises(PromptIntegrityError) as mismatch:
        load_prompt_file(json.dumps({**document, "content_digest": "sha256:" + "0" * 64}))
    assert mismatch.value.code == "prompt_digest_mismatch"
    with pytest.raises(PromptImportError) as kind:
        load_prompt_file(json.dumps({**document, "kind": "chat"}))
    assert reason(kind) == "prompt_kind_mismatch"
    with pytest.raises(PromptImportError) as schema:
        load_prompt_file(json.dumps({"schema": "agenomic.other/v1"}))
    assert reason(schema) == "unsupported_schema"


def test_reports_are_checked_before_upload() -> None:
    report = fixture("discovery-report.json")
    body = build_import_request(report, agent_id=AGENT, options={"prompt_id_prefix": "prm_x_"})
    assert body == {"report": report, "agent_id": AGENT, "options": {"prompt_id_prefix": "prm_x_"}}
    scanned = scan_paths([SOURCES], root=SOURCES)
    assert check_report(scanned)["candidates"]
    leaked = copy.deepcopy(report)
    blocked = next(c for c in leaked["candidates"] if c["status"] == "blocked_secret")
    blocked["content"] = {"schema": "agenomic.prompt_content/v1"}
    secret = copy.deepcopy(report)
    supported = next(c for c in secret["candidates"] if c["status"] == "supported")
    supported["content"]["partials"]["locale"] = "sk-" + "a" * 24
    supported["content_digest"] = prompt_digest(supported["content"])
    tampered = copy.deepcopy(report)
    tampered["candidates"][0]["content"]["partials"]["locale"] = "de"
    absolute = copy.deepcopy(report)
    absolute["files"][0]["path"] = "/home/me/app/graph.py"
    traversal = copy.deepcopy(report)
    traversal["candidates"][0]["source"]["path"] = "../outside.py"
    home = copy.deepcopy(report)
    home["root"]["label"] = "~/agents"
    for label in ("team/../agent", "team\\agent", "a" * 128):
        free_text = copy.deepcopy(report)
        free_text["root"]["label"] = label
        assert check_report(free_text)["root"]["label"] == label
    long_label = copy.deepcopy(report)
    long_label["root"]["label"] = "a" * 129
    drive = copy.deepcopy(report)
    drive["root"]["label"] = "C:agents"
    for document, code in [
        (long_label, "invalid_field_type"),
        (drive, "invalid_field_type"),
        (leaked, "invalid_field_type"),
        (secret, "secret_detected"),
        (tampered, "prompt_digest_mismatch"),
        (absolute, "invalid_field_type"),
        (traversal, "invalid_field_type"),
        (home, "invalid_field_type"),
        ({**report, "schema": "agenomic.other/v1"}, "unsupported_schema"),
        ({**report, "generated_at": 1.5}, "float_not_allowed"),
    ]:
        with pytest.raises(PromptImportError) as raised:
            check_report(document)
        assert reason(raised) == code


def test_plans_are_verified_before_review() -> None:
    plan = fixture("import-plan.json")
    assert verify_plan(plan) is plan
    assert plan_summary(plan["items"]) == plan["summary"]
    with pytest.raises(PromptIntegrityError) as digest:
        verify_plan({**plan, "plan_digest": "sha256:" + "0" * 64})
    assert digest.value.code == "prompt_digest_mismatch"
    counted = resign({**copy.deepcopy(plan), "summary": {**plan["summary"], "skip": 0}})
    with pytest.raises(ApiError) as summary:
        verify_plan(counted)
    assert summary.value.code == "invalid_response"
    content = copy.deepcopy(plan)
    content["items"][0]["content"]["partials"]["locale"] = "de"
    with pytest.raises(PromptIntegrityError):
        verify_plan(resign(content))
    unresolved = copy.deepcopy(plan)
    item = next(i for i in unresolved["items"] if i["slot"] and i["slot"]["status"] == "unresolved")
    item["action"] = "create_prompt"
    unresolved["summary"] = plan_summary(unresolved["items"])
    with pytest.raises(ApiError) as managed:
        verify_plan(resign(unresolved))
    assert managed.value.code == "invalid_response"
    with pytest.raises(ApiError):
        verify_plan({"schema": "agenomic.prompt_import_plan/v1", "items": "x"})


def test_default_decisions_never_apply_a_blocked_item() -> None:
    plan = fixture("import-plan.json")
    decisions = default_decisions(plan)
    assert len(decisions) == len(plan["items"])
    by_id = {decision["item_id"]: decision for decision in decisions}
    for item in plan["items"]:
        decision = by_id[item["item_id"]]
        if item["action"] == "blocked":
            assert decision["action"] == "skip"
            assert "prompt_id" not in decision
        if item["action"] == "create_version":
            assert decision["base_version"] == item["base_version"]
    body = build_apply_request(plan, idempotency_key="k-1", agent_id=AGENT, declare_slots=True)
    assert body["plan_digest"] == plan["plan_digest"]
    assert body["mode"] == "publish"
    assert {decision["action"] for decision in body["items"]} <= {
        "create_prompt",
        "create_version",
        "reuse_version",
        "map_slot_only",
        "skip",
    }
    assert new_idempotency_key().startswith("import-apply-")
    assert new_idempotency_key() != new_idempotency_key()


class FakeImports:
    def __init__(self) -> None:
        self.plan = fixture("import-plan.json")
        self.requests: list[httpx.Request] = []
        self.plan_status = 201

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert "idempotency-key" not in request.headers
        body = json.loads(request.content)
        if request.url.path == "/v1/prompts/imports":
            assert set(body) <= {"report", "agent_id", "options"}
            assert body["report"]["schema"] == "agenomic.prompt_discovery_report/v1"
            record = {"status": "planned", "report_digest": "sha256:" + "a" * 64, "plan": self.plan}
            return httpx.Response(self.plan_status, json={"replayed": False, "import": record})
        if request.url.path == f"/v1/prompts/imports/{self.plan['plan_id']}/apply":
            if body["plan_digest"] != self.plan["plan_digest"]:
                error = {"code": "prompt_import_plan_stale", "message": "stale plan"}
                return httpx.Response(409, json={"error": error})
            results = [{"item_id": i["item_id"], "outcome": "skipped"} for i in body["items"]]
            answer = {"import_id": self.plan["plan_id"], "replayed": False, "results": results}
            return httpx.Response(200, json=answer)
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "no"}})

    def client(self) -> Client:
        return Client(
            api_key="agm_test",
            base_url="https://registry.test",
            transport=httpx.MockTransport(self.handle),
        )


def test_upload_and_apply_cite_the_reviewed_plan() -> None:
    fake = FakeImports()
    client = fake.client()
    record = upload_report(client, fixture("discovery-report.json"), agent_id=AGENT)
    assert record["plan"]["plan_digest"] == fake.plan["plan_digest"]
    assert record["replayed"] is False
    result = apply_import(
        client,
        record["plan"],
        idempotency_key="import-apply-test",
        agent_id=AGENT,
        declare_slots=True,
        expected_slots_revision=4,
    )
    assert [r["outcome"] for r in result["results"]] == ["skipped"] * len(fake.plan["items"])
    apply_request = fake.requests[-1]
    assert apply_request.headers["if-match"] == '"4"'
    sent = json.loads(apply_request.content)
    assert sent["idempotency_key"] == "import-apply-test"
    assert sent["declare_slots"] is True
    with pytest.raises(ValueError):
        apply_import(client, record["plan"], idempotency_key="k", declare_slots=True)
    stale = copy.deepcopy(record["plan"])
    stale["plan_digest"] = "sha256:" + "f" * 64
    with pytest.raises(PromptImportError) as conflict:
        apply_import(client, stale, idempotency_key="k")
    assert conflict.value.code == "prompt_import_plan_stale"
    fake.plan = {**fake.plan, "summary": {**fake.plan["summary"], "blocked": 9}}
    with pytest.raises(ApiError):
        upload_report(client, fixture("discovery-report.json"))


def test_cli_scan_import_and_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    assert cli.main(["prompts", "scan", str(SOURCES), "--out", str(report_path)]) == 0
    assert "candidates:" in capsys.readouterr().err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["root"]["label"] == "prompt_sources"
    fake = FakeImports()
    monkeypatch.setattr(cli, "_cloud_client", fake.client)
    assert cli.main(["prompts", "import", str(report_path), "--agent-id", AGENT]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["plan"]["plan_digest"] == fake.plan["plan_digest"]
    assert fake.plan["plan_id"] in captured.err
    assert len(fake.requests) == 1
    sent = json.loads(fake.requests[0].content)
    assert sent["report"] == report
    code = cli.main(
        ["prompts", "import", str(report_path), "--agent-id", AGENT, "--apply", "--mode", "draft"]
    )
    assert code == 0
    applied = json.loads(fake.requests[-1].content)
    assert applied["mode"] == "draft"
    assert applied["idempotency_key"].startswith("import-apply-")
    assert (
        cli.main(["prompts", "import", str(report_path), "--agent-id", AGENT, "--declare-slots"])
        == 2
    )
    fake.plan_status = 403
    fake.plan = {"error": "x"}
    assert cli.main(["prompts", "import", str(report_path), "--agent-id", AGENT]) == 1
    monkeypatch.setattr(cli, "_cloud_client", lambda: None)
    assert cli.main(["prompts", "import", str(report_path), "--agent-id", AGENT]) == 2
    assert cli.main(["prompts", "import", str(tmp_path / "missing.json"), "--agent-id", AGENT]) == 2
    assert cli.main(["prompts", "scan", str(tmp_path / "missing")]) == 2
    capsys.readouterr()


def test_cli_refuses_a_missing_api_key_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent: list[httpx.Request] = []

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        sent.append(request)
        raise AssertionError(f"no request may be sent: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    for variable in ("AGENOMIC_API_KEY", "AGENOMIC_WORKSPACE_ID", "AGENOMIC_PROMPT_CACHE_DIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("AGENOMIC_ENDPOINT", "https://registry.invalid")
    report_path = tmp_path / "report.json"
    assert cli.main(["prompts", "scan", str(SOURCES), "--out", str(report_path)]) == 0
    capsys.readouterr()
    assert cli.main(["prompts", "import", str(report_path), "--agent-id", AGENT]) == 2
    assert "set AGENOMIC_ENDPOINT and AGENOMIC_API_KEY" in capsys.readouterr().err
    assert cli.main(["prompts", "render", "prm_writer:1"]) == 2
    assert "set AGENOMIC_ENDPOINT and AGENOMIC_API_KEY" in capsys.readouterr().err
    assert sent == []
    monkeypatch.setenv("AGENOMIC_API_KEY", "agm_test")
    configured = cli._cloud_client()
    assert configured is not None
    assert configured.api_key == "agm_test"
    configured.close()
    monkeypatch.delenv("AGENOMIC_ENDPOINT")
    assert cli._cloud_client() is None
    assert sent == []


def test_cli_scan_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    target = SOURCES / "graph_node_single_arg.py"
    assert (
        cli.main(["prompts", "scan", str(target), "--label", "triage", "--commit", "b" * 40]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert [entry["path"] for entry in report["files"]] == ["graph_node_single_arg.py"]
    assert report["root"] == {"label": "triage", "vcs": {"kind": "git", "commit": "b" * 40}}
    assert cli.main(["prompts", "scan", str(target), "--commit", "HEAD"]) == 2


def test_cli_render_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content = complete_content(
        {
            "kind": "chat",
            "body": [{"role": "system", "content": "Plan for {customer}."}],
            "variables": {"customer": {"type": "string", "required": True}},
        }
    )
    path = tmp_path / "planner.json"
    path.write_text(
        json.dumps({"schema": "agenomic.prompt_file/v1", "kind": "chat", "content": content}),
        encoding="utf-8",
    )
    variables = tmp_path / "vars.json"
    variables.write_text(json.dumps({"customer": "Acme"}), encoding="utf-8")
    assert cli.main(["prompts", "digest", str(path)]) == 0
    assert capsys.readouterr().out.strip() == prompt_digest(content)
    assert cli.main(["prompts", "render", str(path), "--vars", str(variables)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["messages"] == [{"role": "system", "content": "Plan for Acme."}]
    assert output["content_digest"] == prompt_digest(content)
    assert output["rendered_hash"].startswith("sha256:")
    assert cli.main(["prompts", "render", str(path)]) == 1
    assert "missing_variable" in capsys.readouterr().err
    variables.write_text("[1]", encoding="utf-8")
    assert cli.main(["prompts", "render", str(path), "--vars", str(variables)]) == 2
    floating = tmp_path / "float.json"
    floating.write_text('{"schema": "agenomic.prompt_content/v1", "x": 1.5}', encoding="utf-8")
    assert cli.main(["prompts", "digest", str(floating)]) == 1
    capsys.readouterr()
    shapeless = tmp_path / "shapeless.json"
    shapeless.write_text('{"schema": "agenomic.prompt_content/v1", "x": 1}', encoding="utf-8")
    assert cli.main(["prompts", "digest", str(shapeless)]) == 1
    assert "prompt_template_invalid" in capsys.readouterr().err
    incomplete = tmp_path / "incomplete.json"
    authored = {key: content[key] for key in ("kind", "body", "variables")}
    incomplete.write_text(
        json.dumps({"schema": "agenomic.prompt_file/v1", "content": authored}), encoding="utf-8"
    )
    assert cli.main(["prompts", "digest", str(incomplete)]) == 1
    assert "prompt_template_invalid" in capsys.readouterr().err
    pinned_content = complete_content(
        {
            "kind": "text",
            "body": "Plan. {>safety}",
            "variables": {},
            "fragments": {
                "safety": {
                    "prompt_id": "prm_safety",
                    "version": 1,
                    "content_digest": "sha256:" + "1" * 64,
                }
            },
        }
    )
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps(pinned_content), encoding="utf-8")
    assert cli.main(["prompts", "digest", str(pinned)]) == 0
    assert capsys.readouterr().out.strip() == prompt_digest(pinned_content)
    engine = seeded_engine()

    class FakePrompts:
        def get(self, ref: str) -> Any:
            prompt_id, version = ref.split(":")
            return engine.get_version(prompt_id, int(version))

    class FakeClient:
        prompts = FakePrompts()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "_cloud_client", FakeClient)
    variables.write_text(json.dumps({"topic": "tea"}), encoding="utf-8")
    assert cli.main(["prompts", "render", "prm_writer:1", "--vars", str(variables)]) == 0
    assert json.loads(capsys.readouterr().out)["text"] == "Write about tea."
    monkeypatch.setattr(cli, "_cloud_client", lambda: None)
    assert cli.main(["prompts", "render", "prm_writer:1"]) == 2
    capsys.readouterr()


def test_cli_bundle_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    engine = seeded_engine()
    release, _ = release_with_child(engine)
    key = SigningKey.generate("orgkey_cli")
    bundle = engine.export_bundle(
        AGENT, signer=key, release_id=release, now=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    key_path = tmp_path / "orgkey_cli.pem"
    key_path.write_text(key.public_pem(), encoding="utf-8")
    base = [
        "prompts",
        "bundle-verify",
        str(bundle_path),
        "--workspace",
        WORKSPACE,
        "--agent",
        AGENT,
    ]
    assert cli.main([*base, "--trust-key", str(key_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["prompt_manifest_digest"] == bundle["prompt_manifest_digest"]
    assert summary["slots"]
    digest = bundle["prompt_bundle_digest"]
    assert cli.main([*base, "--expect-bundle-digest", digest]) == 0
    capsys.readouterr()
    assert cli.main([*base, "--expect-bundle-digest", "sha256:" + "0" * 64]) == 1
    assert "prompt_digest_mismatch" in capsys.readouterr().err
    other = SigningKey.generate("orgkey_cli")
    key_path.write_text(other.public_pem(), encoding="utf-8")
    assert cli.main([*base, "--trust-key", str(key_path)]) == 1
    assert "bundle_signature_invalid" in capsys.readouterr().err
    assert cli.main([*base, "--trust-key", str(tmp_path / "absent.pem")]) == 2
    with pytest.raises(SystemExit):
        cli.main(base)
