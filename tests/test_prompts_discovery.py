from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agenomic.prompts.digest import canonical_json_v1, prompt_digest
from agenomic.prompts.discovery import scan_paths

SOURCES = Path(__file__).parent / "fixtures" / "prompt_sources"
SCHEMAS = Path(__file__).parent / "schemas" / "v0.4"
NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)
FAKE_KEY_TAIL = "test0fake0key0for0discovery0only00"


def scan_fixtures(**kwargs: Any) -> dict[str, Any]:
    return scan_paths([SOURCES], root=SOURCES, now=NOW, **kwargs)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return scan_fixtures(label="support-agent")


def candidate(report: dict[str, Any], path: str, **match: Any) -> dict[str, Any]:
    found = [
        item
        for item in report["candidates"]
        if item["source"]["path"] == path
        and all(
            item["source"].get(key) == value
            if key in ("symbol", "enclosing_function", "line")
            else item.get(key) == value
            for key, value in match.items()
        )
    ]
    assert len(found) == 1, [c["source"] for c in found]
    return found[0]


def codes(item: dict[str, Any]) -> list[str]:
    return [issue["code"] for issue in item["issues"]]


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def schema_errors(document: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    documents = [json.loads(path.read_text()) for path in sorted(SCHEMAS.glob("*.json"))]
    registry = Registry().with_resources(
        (item["$id"], Resource.from_contents(item)) for item in documents
    )
    schema = next(d for d in documents if d["$id"].endswith("prompt-discovery-report.schema.json"))
    validator = Draft202012Validator(schema, registry=registry)
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(document)]


def test_scan_never_imports_or_executes_the_scanned_code(report: dict[str, Any]) -> None:
    before = set(sys.modules)
    again = scan_fixtures()
    loaded = set(sys.modules) - before
    assert not [name for name in loaded if name.startswith(("app", "graph_"))]
    assert not (SOURCES / "app" / "executed.marker").exists()
    entry = next(f for f in again["files"] if f["path"] == "app/raises_on_import.py")
    assert entry["status"] == "scanned"
    onboarding = candidate(again, "app/raises_on_import.py", symbol="ONBOARDING_PROMPT")
    assert onboarding["status"] == "supported"
    assert onboarding["content"]["body"] == "Welcome {customer} to the support desk."


def test_report_matches_the_discovery_schema(report: dict[str, Any]) -> None:
    assert schema_errors(report) == []
    assert report["root"] == {"label": "support-agent", "vcs": None}
    assert report["generated_at"] == "2026-10-05T12:00:00Z"
    assert report["scanner"]["secret_patterns"] == "agenomic-secrets/1"
    assert report["limits"] == {"max_files": 4000, "max_file_bytes": 524288}


def test_files_are_listed_with_the_hash_of_their_bytes(report: dict[str, Any]) -> None:
    paths = [entry["path"] for entry in report["files"]]
    assert paths == sorted(paths)
    assert "graph_nodes_literal.py" in paths
    for entry in report["files"]:
        data = (SOURCES / entry["path"]).read_bytes()
        assert entry["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()


def test_candidate_ids_are_stable_and_follow_the_formula(report: dict[str, Any]) -> None:
    again = scan_fixtures(label="support-agent")
    assert [c["candidate_id"] for c in again["candidates"]] == [
        c["candidate_id"] for c in report["candidates"]
    ]
    for item in report["candidates"]:
        identity = {
            "path": item["source"]["path"],
            "line": item["source"]["line"],
            "column": item["source"]["column"],
            "construct": item["construct"],
        }
        expected = hashlib.sha256(canonical_json_v1(identity).encode()).hexdigest()[:16]
        assert item["candidate_id"] == "cand_" + expected
    slots = [c["proposal"]["slot_path"] for c in report["candidates"] if c["proposal"]["slot_path"]]
    assert len(slots) == len(set(slots))


def test_string_constants_become_text_prompts(report: dict[str, Any]) -> None:
    router = candidate(report, "app/prompts.py", symbol="ROUTER_PROMPT")
    assert router["construct"] == "python.string_constant"
    assert router["status"] == "supported"
    assert router["content"]["body"] == "Route the request to billing, shipping or returns."
    assert router["content_digest"] == prompt_digest(router["content"])
    assert router["source"]["line"] == 14
    assert router["source"]["column"] == 17
    writer = candidate(report, "app/prompts.py", symbol="WRITER_PROMPT")
    assert (
        writer["content"]["body"]
        == "Write a short answer to {question}.\nKeep it under 120 words.\n"
    )
    assert writer["content"]["variables"] == {"question": {"type": "string", "required": True}}
    assert "literal_transform_applied" in codes(writer)
    assert "variable_types_defaulted" in codes(writer)
    assert not [c for c in report["candidates"] if c["source"]["symbol"] == "MAX_TURNS"]
    assert not [c for c in report["candidates"] if c["source"]["symbol"] == "COMPANY"]


def test_chat_template_with_placeholder_and_static_partials(report: dict[str, Any]) -> None:
    planner = candidate(report, "app/templates.py", symbol="PLANNER_PROMPT")
    assert planner["construct"] == "langchain.chat_prompt_template"
    assert planner["status"] == "supported"
    assert planner["proposal"]["prompt_kind"] == "chat"
    assert planner["proposal"]["usage"] == "chat"
    assert planner["content"]["body"] == [
        {
            "role": "system",
            "content": "You plan support work for {customer_name}.\n"
            "Answer in {locale} and never promise refunds.",
        },
        {"placeholder": "history", "optional": True},
        {"role": "user", "content": "{question}"},
    ]
    assert planner["content"]["variables"] == {
        "customer_name": {"type": "string", "required": True},
        "history": {"type": "messages", "required": False},
        "locale": {"type": "string", "required": False},
        "question": {"type": "string", "required": True},
    }
    assert planner["content"]["partials"] == {"locale": "en"}
    consumed = [c for c in report["candidates"] if c["source"]["symbol"] == "PLANNER_SYSTEM_PROMPT"]
    assert consumed == []
    summary = candidate(report, "app/templates.py", symbol="SUMMARY_PROMPT")
    assert summary["content"]["partials"] == {"style": "neutral"}
    assert summary["content"]["variables"]["style"] == {"type": "string", "required": False}


def test_static_messages_keep_their_braces_literal(report: dict[str, Any]) -> None:
    handoff = candidate(report, "app/templates.py", symbol="HANDOFF_PROMPT")
    assert handoff["status"] == "supported"
    assert handoff["content"]["body"][0] == {
        "role": "system",
        "content": "Hand-offs follow the {{severity}} matrix verbatim.",
    }
    assert "severity" not in handoff["content"]["variables"]
    assert handoff["content"]["body"][1] == {"placeholder": "history", "optional": True}
    assert "static_message_escaped" in codes(handoff)


def test_agent_constructor_prompts_are_system_text(report: dict[str, Any]) -> None:
    react = candidate(
        report, "app/agents.py", construct="langgraph.create_react_agent.prompt", status="supported"
    )
    assert react["proposal"]["usage"] == "system"
    assert react["proposal"]["slot_path"] == "researcher.system"
    assert react["content"]["kind"] == "text"
    assert (
        react["content"]["body"] == 'You research orders. Reply with JSON like {{"status": "..."}}.'
    )
    assert react["content"]["variables"] == {}
    agent = candidate(report, "app/agents.py", construct="langchain.create_agent.system_prompt")
    assert agent["status"] == "supported"
    assert agent["content"]["body"] == "You handle refunds within policy."
    assert agent["proposal"]["slot_path"] == "refunds.system"


def test_unresolved_constructs_are_reported_never_managed(report: dict[str, Any]) -> None:
    expected = {
        ("app/templates.py", "build_research_prompt"): "python_fstring",
        ("app/prompts.py", "BANNER_PROMPT"): "dynamic_template",
        ("app/templates.py", "REVIEW_PROMPT"): "remote_prompt",
        ("app/billing.py", "STRIPE_PROMPT"): "dynamic_template",
        ("app/agents.py", "_tiered_prompt"): "python_fstring",
        ("app/templates.py", "signature"): "dynamic_template",
    }
    for (path, name), code in expected.items():
        key = "enclosing_function" if name.islower() else "symbol"
        item = candidate(report, path, **{key: name})
        assert item["status"] == "unresolved", (path, name)
        assert item["content"] is None
        assert item["content_digest"] is None
        assert item["proposal"]["prompt_id"] is None
        assert code in codes(item), (path, name, codes(item))
    callable_prompt = candidate(
        report,
        "app/agents.py",
        construct="langgraph.create_react_agent.prompt",
        status="unresolved",
    )
    assert "dynamic_template" in codes(callable_prompt)
    remote = candidate(report, "app/templates.py", symbol="REVIEW_PROMPT")
    assert remote["construct"] == "dynamic"
    for item in report["candidates"]:
        if item["status"] != "supported":
            assert item["content"] is None
            assert all(
                issue["severity"] in ("error", "warning", "info") for issue in item["issues"]
            )


def test_refused_features_are_unsupported(report: dict[str, Any]) -> None:
    triage = candidate(report, "app/templates.py", symbol="TRIAGE_PROMPT")
    assert triage["status"] == "unsupported"
    assert codes(triage) == ["unsupported_template_format"]
    assert triage["content"] is None
    dated = candidate(report, "app/templates.py", symbol="DATED_PROMPT")
    assert dated["status"] == "unsupported"
    assert "callable_partial" in codes(dated)
    issue = next(i for i in dated["issues"] if i["code"] == "callable_partial")
    assert (issue["line"], issue["column"]) == (43, 33)


def test_secret_candidates_carry_location_but_no_text(report: dict[str, Any]) -> None:
    billing = candidate(report, "app/billing.py", symbol="BILLING_PROMPT")
    assert billing["status"] == "blocked_secret"
    assert billing["content"] is None
    assert billing["content_digest"] is None
    assert billing["secret_findings"] == [
        {"pattern": "openai_key", "line": 7, "column": 8, "length": 3 + len(FAKE_KEY_TAIL)}
    ]
    assert codes(billing)[0] == "secret_detected"
    serialized = json.dumps(report)
    assert FAKE_KEY_TAIL not in serialized
    assert "sk-test0" not in serialized
    stripe = candidate(report, "app/billing.py", symbol="STRIPE_PROMPT")
    assert stripe["secret_findings"] == []


def test_add_node_literal_links_node_path(report: dict[str, Any]) -> None:
    planner = candidate(report, "graph_nodes_literal.py", symbol="PLANNER_TEMPLATE")
    assert planner["proposal"]["node_path"] == "planner"
    assert planner["proposal"]["slot_path"] == "planner.chat"
    reviewer = candidate(report, "graph_nodes_literal.py", enclosing_function="reviewer")
    assert reviewer["construct"] == "langchain.message"
    assert reviewer["proposal"]["node_path"] == "reviewer"
    assert reviewer["proposal"]["slot_path"] == "reviewer.system"
    assert reviewer["content"]["body"] == [
        {"role": "system", "content": "Check the draft for tone and accuracy."}
    ]
    writer = candidate(report, "app/prompts.py", symbol="WRITER_PROMPT")
    assert writer["proposal"]["node_path"] == "writer"
    assert writer["proposal"]["slot_path"] == "writer.instructions"
    router = candidate(report, "app/prompts.py", symbol="ROUTER_PROMPT")
    assert router["proposal"]["node_path"] is None


def test_single_argument_add_node_uses_the_function_name(report: dict[str, Any]) -> None:
    triage = candidate(report, "graph_node_single_arg.py", enclosing_function="triage")
    assert triage["proposal"]["node_path"] == "triage"
    assert triage["proposal"]["slot_path"] == "triage.system"
    assert triage["content"]["body"][0]["content"] == (
        "Sort the request into billing, shipping or returns."
    )
    consumed = [c for c in report["candidates"] if c["source"]["symbol"] == "TRIAGE_SYSTEM_PROMPT"]
    assert consumed == []


def test_compiled_subgraph_is_unmapped_subagent(report: dict[str, Any]) -> None:
    subagents = [
        c
        for c in report["candidates"]
        if c["source"]["path"] == "graph_subagent_compiled.py"
        and c["construct"] == "langgraph.subagent_node"
    ]
    assert [c["proposal"]["node_path"] for c in subagents] == ["research", "investigate"]
    for item in subagents:
        assert item["status"] == "unresolved"
        assert item["content"] is None
        assert item["proposal"]["slot_path"] is None
        assert item["proposal"]["prompt_id"] is None
        assert codes(item) == ["subagent_unmapped"]
        assert item["issues"][0]["severity"] == "warning"
    prompt = candidate(
        report, "graph_subagent_compiled.py", construct="langgraph.create_react_agent.prompt"
    )
    assert prompt["proposal"]["node_path"] == "investigate"
    assert prompt["content"]["body"] == "You research the order history before anyone answers."
    summarize = candidate(report, "graph_subagent_compiled.py", enclosing_function="summarize")
    assert summarize["proposal"]["node_path"] == "summarize"


def test_dynamic_add_node_reports_node_unresolved(report: dict[str, Any]) -> None:
    escalate = candidate(report, "graph_node_dynamic.py", enclosing_function="escalate")
    assert escalate["proposal"]["node_path"] is None
    assert "node_unresolved" in codes(escalate)
    issue = next(i for i in escalate["issues"] if i["code"] == "node_unresolved")
    assert issue["severity"] == "info"
    apologize = candidate(report, "graph_node_dynamic.py", enclosing_function="Nodes.apologize")
    assert apologize["proposal"]["node_path"] is None
    assert "node_unresolved" in codes(apologize)
    assert apologize["status"] == "supported"


def test_skipped_files_carry_a_reason(tmp_path: Path) -> None:
    write(tmp_path, "pkg/ok.py", 'A_PROMPT = "Say {x}."\n')
    write(tmp_path, "pkg/broken.py", "def broken(:\n")
    write(tmp_path, "pkg/big.py", "B_PROMPT = '" + "y" * 200 + "'\n")
    (tmp_path / "pkg" / "latin.py").write_bytes(b'C_PROMPT = "caf\xe9"\n')
    write(tmp_path, "pkg/generated_pb2.py", 'D_PROMPT = "never read"\n')
    write(tmp_path, ".venv/lib/site.py", 'E_PROMPT = "pruned"\n')
    write(tmp_path, "pkg/zz_last.py", 'F_PROMPT = "over the limit"\n')
    result = scan_paths(
        [tmp_path],
        root=tmp_path,
        max_files=5,
        max_file_bytes=120,
        exclude=(".venv", "*_pb2.py"),
        now=NOW,
    )
    reasons = {entry["path"]: entry["skip_reason"] for entry in result["files"]}
    assert reasons == {
        "pkg/big.py": "too_large",
        "pkg/broken.py": "syntax_error",
        "pkg/generated_pb2.py": "excluded",
        "pkg/latin.py": "not_utf8",
        "pkg/ok.py": None,
        "pkg/zz_last.py": "limit_reached",
    }
    assert [c["source"]["symbol"] for c in result["candidates"]] == ["A_PROMPT"]
    assert result["root"]["label"] == tmp_path.name


def test_a_file_too_deep_to_analyze_is_skipped_not_the_scan(tmp_path: Path) -> None:
    chain = " + ".join(['"a"'] * 1000)
    write(tmp_path, "deep_constant.py", f"DEEP_PROMPT = {chain}\n")
    write(tmp_path, "deep_body.py", f"def build():\n    return {chain}\n")
    write(
        tmp_path,
        "deep_template.py",
        "from langchain_core.prompts import PromptTemplate\n"
        f"DEEP = PromptTemplate.from_template({chain})\n",
    )
    write(tmp_path, "ok.py", 'OK_PROMPT = "Say {x}."\n')
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    reasons = {entry["path"]: entry["skip_reason"] for entry in result["files"]}
    assert reasons == {
        "deep_body.py": "syntax_error",
        "deep_constant.py": "syntax_error",
        "deep_template.py": "syntax_error",
        "ok.py": None,
    }
    assert [c["source"]["path"] for c in result["candidates"]] == ["ok.py"]
    assert schema_errors(result) == []


def test_a_skipped_file_releases_the_constants_it_used(tmp_path: Path) -> None:
    write(tmp_path, "shared.py", 'SHARED_PROMPT = "Shared {x}."\n')
    chain = " + ".join(['"a"'] * 1000)
    write(
        tmp_path,
        "user.py",
        "from langchain_core.prompts import PromptTemplate\n"
        "from shared import SHARED_PROMPT\n"
        "T = PromptTemplate.from_template(SHARED_PROMPT)\n"
        f"def build():\n    return {chain}\n",
    )
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    reasons = {entry["path"]: entry["skip_reason"] for entry in result["files"]}
    assert reasons == {"shared.py": None, "user.py": "syntax_error"}
    shared = candidate(result, "shared.py", symbol="SHARED_PROMPT")
    assert shared["content"]["body"] == "Shared {x}."


def test_names_outside_the_report_bounds_are_never_reported(tmp_path: Path) -> None:
    long_function = "build_" + "x" * 300
    write(
        tmp_path,
        "graph.py",
        "from langgraph.graph import StateGraph\n"
        'EMPTY_PROMPT = "Empty {a}."\n'
        'LONG_PROMPT = "Long {b}."\n'
        'NUL_PROMPT = "Nul {c}."\n'
        'NAMED_PROMPT = "Named {d}."\n'
        "def empty(state):\n    return EMPTY_PROMPT\n"
        "def long(state):\n    return LONG_PROMPT\n"
        "def nul(state):\n    return NUL_PROMPT\n"
        f"def {long_function}(state):\n    return NAMED_PROMPT\n"
        "g = StateGraph(dict)\n"
        'g.add_node("", empty)\n'
        f'g.add_node("{"n" * 257}", long)\n'
        'g.add_node("a\\x00b", nul)\n'
        f"g.add_node({long_function})\n",
    )
    long_symbol = "A" * 250 + "_PROMPT"
    write(
        tmp_path,
        "names.py",
        "from langchain_core.prompts import PromptTemplate\n"
        f'{long_symbol} = "Hello {{x}}."\n'
        f"def {long_function}():\n"
        '    return PromptTemplate.from_template("Inside {e}.")\n',
    )
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    assert schema_errors(result) == []
    assert [entry["path"] for entry in result["files"]] == ["graph.py", "names.py"]
    for symbol in ("EMPTY_PROMPT", "LONG_PROMPT", "NUL_PROMPT", "NAMED_PROMPT"):
        item = candidate(result, "graph.py", symbol=symbol)
        assert item["proposal"]["node_path"] is None
        assert "node_unresolved" in codes(item)
    constant = candidate(result, "names.py", construct="python.string_constant")
    assert constant["source"]["symbol"] is None
    assert constant["status"] == "supported"
    inner = candidate(result, "names.py", construct="langchain.prompt_template")
    assert inner["source"]["enclosing_function"] is None
    assert inner["status"] == "supported"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows treats a backslash as a path separator, so no such file name can exist",
)
def test_file_names_with_a_backslash_are_never_reported(tmp_path: Path) -> None:
    write(tmp_path, "plain.py", 'PLAIN_PROMPT = "Plain {x}."\n')
    write(tmp_path, "odd\\name.py", 'ODD_PROMPT = "Odd {x}."\n')
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    assert schema_errors(result) == []
    assert [entry["path"] for entry in result["files"]] == ["plain.py"]


FAKE_AWS_KEY = "AKIA" + "TESTFAKEKEY00000"


def strings_outside_content(document: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(document, str):
        return [(path, document)]
    if isinstance(document, list):
        return [
            pair
            for index, item in enumerate(document)
            for pair in strings_outside_content(item, f"{path}/{index}")
        ]
    if isinstance(document, dict):
        return [
            pair
            for key, item in document.items()
            if key != "content"
            for pair in strings_outside_content(item, f"{path}/{key}")
        ]
    return []


def assert_no_secret_outside_content(document: dict[str, Any]) -> None:
    from agenomic.prompts.secrets import scan

    serialized = json.dumps(document)
    assert FAKE_AWS_KEY not in serialized
    assert FAKE_AWS_KEY.lower() not in serialized.lower()
    assert [(path, text) for path, text in strings_outside_content(document) if scan(text)] == []
    assert schema_errors(document) == []


def test_secrets_in_names_paths_and_labels_never_reach_the_report(tmp_path: Path) -> None:
    from agenomic.prompts.importer import check_report

    key = FAKE_AWS_KEY
    graph = (
        "from langgraph.graph import StateGraph\n"
        "from langgraph.prebuilt import create_react_agent\n"
        'PLAN_PROMPT = "Plan {task}."\n'
        'REVIEW_PROMPT = "Review {draft}."\n'
        "def plan(state):\n    return PLAN_PROMPT\n"
        "def review(state):\n    return REVIEW_PROMPT\n"
        "research = create_react_agent(model, tools=[], prompt='Research the order.')\n"
        "g = StateGraph(dict)\n"
        f'g.add_node("planner-{key}", plan)\n'
        f'g.add_node("research-{key}", research)\n'
        f'g.add_node("review-{key}", review)\n'
        f'g.add_node("review-{key[:-1]}1", review)\n'
    )
    write(tmp_path, "graph.py", graph)
    write(
        tmp_path,
        "names.py",
        "from langchain_core.prompts import PromptTemplate\n"
        f'{key} = PromptTemplate.from_template("Hello {{name}}.")\n'
        f"def {key[:-1]}1():\n"
        '    return PromptTemplate.from_template("Inside {e}.")\n'
        "def handler():\n"
        '    return PromptTemplate.from_template("Plain {f}.")\n',
    )
    write(tmp_path, f"keys/{key}.py", 'KEY_PROMPT = "Use {x}."\n')
    result = scan_paths([tmp_path], root=tmp_path, now=NOW, label=f"agent {key}")
    assert_no_secret_outside_content(result)
    assert check_report(result) == result
    assert result["root"]["label"] == "agent [REDACTED:aws_access_key]"
    paths = [entry["path"] for entry in result["files"]]
    assert paths == ["graph.py", "keys/[REDACTED:aws_access_key].py", "names.py"]
    finding = {"pattern": "aws_access_key", "length": len(key)}

    planner = candidate(result, "graph.py", symbol="PLAN_PROMPT")
    assert planner["status"] == "blocked_secret"
    assert planner["content"] is None
    assert planner["content_digest"] is None
    assert codes(planner)[0] == "secret_detected"
    assert planner["secret_findings"] == [
        {**finding, "line": planner["source"]["line"], "column": planner["source"]["column"]}
    ]
    assert planner["proposal"]["node_path"] == "planner-[REDACTED:aws_access_key]"
    assert planner["proposal"]["slot_path"] == "planner_redacted_aws_access_key.instructions"
    assert planner["proposal"]["prompt_id"] == "prm_planner_redacted_aws_access_key_instructions"

    review = candidate(result, "graph.py", symbol="REVIEW_PROMPT")
    assert review["status"] == "supported"
    assert review["proposal"]["node_path"] is None
    assert "node_unresolved" in codes(review)
    assert review["secret_findings"] == []

    subagent = candidate(result, "graph.py", construct="langgraph.subagent_node")
    assert subagent["status"] == "unresolved"
    assert subagent["proposal"]["node_path"] == "research-[REDACTED:aws_access_key]"
    assert {"subagent_unmapped", "secret_detected"} <= set(codes(subagent))
    assert [item["pattern"] for item in subagent["secret_findings"]] == ["aws_access_key"]
    agent = candidate(result, "graph.py", construct="langgraph.create_react_agent.prompt")
    assert agent["status"] == "blocked_secret"
    assert agent["proposal"]["node_path"] == "research-[REDACTED:aws_access_key]"

    named = candidate(result, "names.py", line=2)
    assert named["source"]["symbol"] == "[REDACTED:aws_access_key]"
    assert named["status"] == "blocked_secret"
    assert named["proposal"]["slot_path"] == "redacted_aws_access_key.instructions"
    inner = candidate(result, "names.py", line=4)
    assert inner["source"]["enclosing_function"] == "[REDACTED:aws_access_key]"
    assert inner["status"] == "blocked_secret"
    plain = candidate(result, "names.py", enclosing_function="handler")
    assert plain["status"] == "supported"

    stored = candidate(result, "keys/[REDACTED:aws_access_key].py", symbol="KEY_PROMPT")
    assert stored["status"] == "blocked_secret"
    assert [item["pattern"] for item in stored["secret_findings"]] == ["aws_access_key"]


def test_an_issue_message_never_carries_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agenomic.prompts import importer

    monkeypatch.setitem(
        importer.ISSUE_MESSAGES, "variable_types_defaulted", f"types near {FAKE_AWS_KEY}"
    )
    write(tmp_path, "note.py", 'NOTE_PROMPT = "Note {x}."\n')
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    assert_no_secret_outside_content(result)
    note = candidate(result, "note.py", symbol="NOTE_PROMPT")
    assert note["status"] == "blocked_secret"
    assert note["content"] is None
    messages = [issue["message"] for issue in note["issues"]]
    assert "types near [REDACTED:aws_access_key]" in messages
    assert note["secret_findings"] == [
        {"pattern": "aws_access_key", "line": 1, "column": 15, "length": len(FAKE_AWS_KEY)}
    ]


def test_slugs_that_look_like_secrets_are_never_proposed(tmp_path: Path) -> None:
    token = "GHP_" + "A" * 36
    write(
        tmp_path,
        "hub.py",
        "from langchain_core.prompts import PromptTemplate\n"
        f'{token} = PromptTemplate.from_template("Hi {{x}}.")\n',
    )
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    assert_no_secret_outside_content(result)
    hub = candidate(result, "hub.py", construct="langchain.prompt_template")
    assert hub["status"] == "supported"
    assert hub["source"]["symbol"] == token
    assert hub["proposal"]["slot_path"] == "redacted_github_token.instructions"


def test_columns_count_code_points_and_follow_escapes(tmp_path: Path) -> None:
    line = 'TITLE = "café"; GREETING_PROMPT = "Hello {name}"\n'
    secret = '"Bearer abcdefghijklmnopqrstuvwxyz0123"'
    token_line = 'TOKEN_PROMPT = ("a\\tb " ' + secret + ")\n"
    write(tmp_path, "mod.py", line + token_line)
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    greeting = candidate(result, "mod.py", symbol="GREETING_PROMPT")
    assert greeting["source"]["column"] == line.index('"Hello') + 1
    assert greeting["source"]["end_column"] == line.index("\n")
    token = candidate(result, "mod.py", symbol="TOKEN_PROMPT")
    assert token["status"] == "blocked_secret"
    assert token["secret_findings"] == [
        {
            "pattern": "bearer_token",
            "line": 2,
            "column": token_line.index("Bearer") + 1,
            "length": 37,
        }
    ]


def test_scan_of_one_file_and_root_checks(tmp_path: Path) -> None:
    target = write(tmp_path, "agent/main.py", 'SYSTEM_PROMPT = "Be brief."\n')
    write(tmp_path, "agent/other.py", 'OTHER_PROMPT = "Not scanned."\n')
    result = scan_paths([target], root=tmp_path, commit="a" * 40, now=NOW)
    assert [entry["path"] for entry in result["files"]] == ["agent/main.py"]
    assert result["root"]["vcs"] == {"kind": "git", "commit": "a" * 40}
    only = result["candidates"][0]
    assert only["proposal"]["usage"] == "system"
    assert only["proposal"]["slot_path"] == "main.system"
    with pytest.raises(ValueError):
        scan_paths([SOURCES], root=tmp_path)
    with pytest.raises(ValueError):
        scan_paths([target], root=tmp_path, label="/home/me/agent")
    with pytest.raises(ValueError):
        scan_paths([target], root=tmp_path, commit="HEAD")
    with pytest.raises(ValueError):
        scan_paths([target], root=tmp_path, max_files=0)


def test_literal_forms_and_runtime_values(tmp_path: Path) -> None:
    write(
        tmp_path,
        "forms.py",
        "import inspect\n"
        "\n"
        "from langchain_core.prompts import ChatPromptTemplate, PromptTemplate\n"
        "\n"
        'BASE_PROMPT = "Answer {question}."\n'
        "ALIAS_PROMPT = BASE_PROMPT\n"
        'JOINED_PROMPT = "Hi " + "{name}"\n'
        'CLEAN_PROMPT = inspect.cleandoc("""\n    Line one.\n    Line two.\n""")\n'
        'STRIPPED_PROMPT = "  padded  ".strip()\n'
        'PERCENT_PROMPT = "%s rules" % "house"\n'
        'PLAIN_FSTRING_PROMPT = f"no fields {{x}}"\n'
        "CHAT = ChatPromptTemplate.from_template('{topic} please')\n"
        "TOOLY = ChatPromptTemplate.from_messages([('tool', 'x'), ('system', 'y')])\n"
        "COMPOSED_PROMPT = CHAT + CHAT\n"
        "PARTIAL = PromptTemplate.from_template('{a} {b}', partial_variables={'a': 3, 'b': None})\n"
        "NESTED = PromptTemplate.from_template('{a}', partial_variables={'a': [1]})\n"
        "UNUSED = PromptTemplate.from_template('{a}', partial_variables={'z': 'q'})\n"
        "SPEC = PromptTemplate.from_template('{a:>3}')\n"
        "FRAG = PromptTemplate.from_template('{>frag}')\n",
    )
    result = scan_paths([tmp_path], root=tmp_path, now=NOW)
    by_symbol = {c["source"]["symbol"]: c for c in result["candidates"]}
    assert "BASE_PROMPT" not in by_symbol
    assert by_symbol["ALIAS_PROMPT"]["content"]["body"] == "Answer {question}."
    assert by_symbol["JOINED_PROMPT"]["content"]["body"] == "Hi {name}"
    assert by_symbol["CLEAN_PROMPT"]["content"]["body"] == "Line one.\nLine two."
    assert by_symbol["STRIPPED_PROMPT"]["content"]["body"] == "padded"
    assert "dynamic_template" in codes(by_symbol["PERCENT_PROMPT"])
    assert by_symbol["PLAIN_FSTRING_PROMPT"]["content"]["body"] == "no fields {x}"
    assert by_symbol["CHAT"]["content"]["body"] == [{"role": "user", "content": "{topic} please"}]
    assert codes(by_symbol["TOOLY"]) == ["unsupported_role"]
    assert codes(by_symbol["COMPOSED_PROMPT"]) == ["template_composition"]
    assert by_symbol["PARTIAL"]["content"]["partials"] == {"a": "3", "b": "None"}
    assert codes(by_symbol["PARTIAL"]).count("partial_coerced_to_string") == 2
    assert "partial_not_scalar" in codes(by_symbol["NESTED"])
    assert "unused_partial_dropped" in codes(by_symbol["UNUSED"])
    assert by_symbol["UNUSED"]["status"] == "supported"
    assert "format_spec" in codes(by_symbol["SPEC"])
    assert by_symbol["SPEC"]["status"] == "unsupported"
    assert "fragment_syntax_in_import" in codes(by_symbol["FRAG"])
