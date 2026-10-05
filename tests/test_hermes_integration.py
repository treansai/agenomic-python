"""Integration tests with the REAL pinned Hermes Agent (v2026.9.24, 0.21.5).

Skipped unless ``run_agent`` is importable, i.e. unless pytest runs in the
Hermes venv (editable install of the pinned clone plus this repository):

    /home/user/upstream/hermes-venv/bin/python -m pytest -m hermes

Each test starts a fake Agenomic server (runtime API + Model Gateway with a
scripted model) and runs one real ``AIAgent`` conversation in a fresh process
whose ``HERMES_HOME`` enables the ``agenomic`` plugin by its entry point.
Assertions are on external effects: files on disk and requests received.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import pytest
from hermes_fakes import FakeAgenomic, ScriptedLLM

from agenomic.integrations.hermes.canonical import arguments_hash

pytestmark = [
    pytest.mark.hermes,
    pytest.mark.skipif(
        importlib.util.find_spec("run_agent") is None, reason="Hermes Agent not installed"
    ),
]

DRIVER = Path(__file__).with_name("hermes_agent_driver.py")
TOKEN = "agmhr_integration"
GUARD = Path(sys.executable).with_name("agenomic-hermes-guard")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "home").mkdir()
    return tmp_path


def write_config(
    home: Path, server: FakeAgenomic, *, enable_plugin: bool = True, guard: bool = False
) -> None:
    lines = [
        "model:",
        "  default: demo-model",
        "  provider: custom",
        f'  base_url: "{server.model_base_url}"',
        "  api_key: ${AGENOMIC_HERMES_RUNTIME_TOKEN}",
        "  api_mode: chat_completions",
        "plugins:",
        f"  enabled: [{'agenomic' if enable_plugin else ''}]",
        "  hook_callback_timeout: 30",
        "  entries:",
        "    agenomic:",
        "      settings:",
        f'        endpoint: "{server.url}"',
        "        timeouts: {decision_s: 5}",
        "        buffer: {flush_interval_s: 0.1}",
        "skills:",
        "  write_approval: true",
    ]
    if guard:
        lines += [
            "hooks:",
            "  pre_tool_call:",
            f'    - command: "{GUARD}"',
            "      fail_closed: true",
            "      timeout: 10",
        ]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_agent(workdir: Path, server: FakeAgenomic, *, shell_hooks: bool = False) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "AGENOMIC_"))}
    env.update(HERMES_HOME=str(workdir / "home"), AGENOMIC_HERMES_RUNTIME_TOKEN=TOKEN)
    cfg = {
        "base_url": server.model_base_url,
        "api_key": TOKEN,
        "prompt": "do the task",
        "shell_hooks": shell_hooks,
    }
    proc = subprocess.run(
        [sys.executable, str(DRIVER), json.dumps(cfg)],
        env=env,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert lines, proc.stderr[-4000:]
    out: dict[str, Any] = json.loads(lines[-1][len("RESULT:") :])
    return out


@pytest.fixture
def server_factory() -> Iterator[Any]:
    servers: list[FakeAgenomic] = []

    def make(tool_calls: list[tuple[str, dict[str, Any]]], **kwargs: Any) -> FakeAgenomic:
        server = FakeAgenomic(llm=ScriptedLLM(tool_calls), **kwargs)
        servers.append(server)
        return server

    yield make
    for s in servers:
        s.close()


def tool_results_seen_by_model(server: FakeAgenomic) -> list[str]:
    out: list[str] = []
    for body in server.llm.requests:
        for m in body.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool":
                out.append(str(m.get("content")))
    return out


def chat_requests(server: FakeAgenomic) -> list[Any]:
    return [r for r in server.requests if r.path.endswith("/model/v1/chat/completions")]


def test_plugin_loaded_by_entry_point_allow_writes_once(workdir: Path, server_factory: Any) -> None:
    target = workdir / "allowed.txt"
    server = server_factory([("write_file", {"path": str(target), "content": "hello from hermes"})])
    write_config(workdir / "home", server)
    result = run_agent(workdir, server)

    plugin = result["plugins"]["agenomic"]
    assert plugin == {"source": "entrypoint", "enabled": True, "error": None}
    assert target.read_text() == "hello from hermes"
    authorize = server.authorize_calls()
    assert len(authorize) == 1
    assert authorize[0].body["tool"] == "write_file"
    assert authorize[0].body["schema_hash"].startswith("blake3:")
    reports = server.reports()
    assert len(reports) == 1
    report = reports[0].body
    assert report["logical_call_id"] == authorize[0].body["tool_call_id"]
    assert report["permit"]["document"]["record_id"] == f"rec-{report['logical_call_id']}-1"
    assert report["arguments"] == authorize[0].body["arguments"]
    assert report["permit"]["document"]["arguments_hash"] == arguments_hash(report["arguments"])
    assert report["is_error"] is False

    hello = server.calls("/hello")[0].body
    assert hello["hermes"]["version"] == "0.21.5"
    assert hello["hermes"]["release_date"] == "2026.9.24"
    assert hello["hermes"]["commit"] == "f97608f178d1ffeca59860195ab7da295f7c8e5f"
    assert hello["contracts"]["pre_tool_call"]
    assert hello["contracts"]["tool_execution"]
    assert hello["contracts"]["llm_request"]
    assert hello["foreign_mutators"] == []
    checks = {c["check"]: c["status"] for c in hello["compat_results"]}
    assert checks["hermes_version_compatible"] == "pass"
    assert checks["pre_tool_call_fail_closed"] == "pass"
    assert checks["model_gateway_provider"] == "pass"
    assert checks["skills_write_approval"] == "pass"

    discovered = server.calls("/tools/discovered")[0].body["tools"]
    assert "write_file" in {t["tool_name"] for t in discovered}

    sid = result["session_id"]
    admitted = server.calls("/v1/hermes/runtime/sessions")
    assert admitted[0].body["hermes_session_id"] == sid
    headers = [r.headers.get("X-Agenomic-Hermes-Session") for r in chat_requests(server)]
    assert headers
    assert all(h == sid for h in headers)

    types = server.event_types()
    for expected in (
        "adapter.loaded",
        "session.started",
        "model.call.started",
        "model.call.completed",
        "tool.call.decision",
        "tool.call.executed",
        "tool.call.completed",
        "session.turn_ended",
    ):
        assert expected in types
    assert "hello from hermes" not in json.dumps(server.events)


def test_deny_file_not_created_and_model_receives_block(workdir: Path, server_factory: Any) -> None:
    target = workdir / "denied.txt"
    server = server_factory(
        [("write_file", {"path": str(target), "content": "nope"})], decide=lambda b: "deny"
    )
    write_config(workdir / "home", server)
    run_agent(workdir, server)

    assert not target.exists()
    assert server.reports() == []
    seen = tool_results_seen_by_model(server)
    assert any("Agenomic denied write_file: writes are not allowed" in s for s in seen)


def test_require_approval_blocks_then_controlled_retry(workdir: Path, server_factory: Any) -> None:
    target = workdir / "approved.txt"
    args = {"path": str(target), "content": "after approval"}
    server = server_factory(
        [("write_file", args), ("write_file", args)],
        decide=lambda b: "require_approval",
        auto_approve=True,
    )
    write_config(workdir / "home", server)
    run_agent(workdir, server)

    seen = tool_results_seen_by_model(server)
    approval_id = next(iter(server.approvals))
    assert any(f"Agenomic approval {approval_id} required" in s for s in seen)
    assert target.read_text() == "after approval"
    calls = server.authorize_calls()
    assert len(calls) == 2
    assert calls[1].body["tool_call_id"] == calls[0].body["tool_call_id"]  # identity reused
    assert server.approvals[approval_id]["status"] == "consumed"
    assert len(server.reports()) == 1


def test_observe_mode_never_authorizes(workdir: Path, server_factory: Any) -> None:
    target = workdir / "observed.txt"
    server = server_factory(
        [("write_file", {"path": str(target), "content": "o"})], effective_state="observe"
    )
    write_config(workdir / "home", server)
    run_agent(workdir, server)
    assert target.read_text() == "o"
    assert server.authorize_calls() == []
    assert "tool.call.requested" in server.event_types()


def test_guard_allows_with_plugin_and_blocks_without(workdir: Path, server_factory: Any) -> None:
    assert GUARD.exists(), "agenomic-hermes-guard is not installed in this venv"
    target = workdir / "guarded.txt"
    server = server_factory([("write_file", {"path": str(target), "content": "g"})])
    write_config(workdir / "home", server, guard=True)
    run_agent(workdir, server, shell_hooks=True)
    assert target.read_text() == "g"

    target.unlink()
    other: Optional[FakeAgenomic] = server_factory(
        [("write_file", {"path": str(target), "content": "g"})]
    )
    assert other is not None
    fresh = workdir / "second"
    (fresh / "home").mkdir(parents=True)
    write_config(fresh / "home", other, enable_plugin=False, guard=True)
    run_agent(fresh, other, shell_hooks=True)
    assert not target.exists()
    seen = tool_results_seen_by_model(other)
    assert any("Agenomic adapter status is missing" in s for s in seen)
