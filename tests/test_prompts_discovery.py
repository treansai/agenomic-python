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
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    documents = [json.loads(path.read_text()) for path in sorted(SCHEMAS.glob("*.json"))]
    registry = Registry().with_resources(
        (document["$id"], Resource.from_contents(document)) for document in documents
    )
    schema = next(d for d in documents if d["$id"].endswith("prompt-discovery-report.schema.json"))
    validator = Draft202012Validator(schema, registry=registry)
    errors = [f"{e.json_path}: {e.message}" for e in validator.iter_errors(report)]
    assert errors == []
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
