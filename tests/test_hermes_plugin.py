"""Adapter behaviour against a fake Agenomic runtime API (no Hermes needed).

The two Hermes call orders are simulated faithfully:

* agent loop (``agent/tool_executor.py``): ``tool_execution`` middleware wraps
  a terminal that runs ``pre_tool_call`` then the tool;
* direct dispatch (``model_tools.handle_function_call``): ``pre_tool_call``
  first, then the middleware around the tool.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from hermes_fakes import FakeAgenomic, FakeCtx
from pydantic import SecretStr

from agenomic.integrations.hermes import plugin as plugin_mod
from agenomic.integrations.hermes.canonical import arguments_hash
from agenomic.integrations.hermes.config import AdapterConfig, ConfigError
from agenomic.integrations.hermes.plugin import APPROVAL_MESSAGE, HermesAdapter


@pytest.fixture(autouse=True)
def _shutdown_adapters(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop every adapter a test creates, so no exporter thread outlives its test and
    posts into another test's HTTP mock."""
    created: list[HermesAdapter] = []
    original = HermesAdapter.__init__

    def tracking_init(self: HermesAdapter, *args: Any, **kwargs: Any) -> None:
        created.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(HermesAdapter, "__init__", tracking_init)
    yield
    for adapter in created:
        if hasattr(adapter, "exporter"):
            adapter.shutdown()


@pytest.fixture
def server() -> Iterator[FakeAgenomic]:
    s = FakeAgenomic()
    yield s
    s.close()


PINNED = {
    "version": "0.21.5",
    "release_date": "2026.9.24",
    "commit": "f97608f178d1ffeca59860195ab7da295f7c8e5f",
}


def make_adapter(
    server_url: str, tmp_path: Path, identity: Optional[dict[str, Any]] = None, **overrides: Any
) -> HermesAdapter:
    doc: dict[str, Any] = {
        "endpoint": server_url,
        "timeouts": {"connect_s": 1.0, "decision_s": 1.0, "report_s": 1.0},
        "buffer": {"flush_interval_s": 0.05},
    }
    doc.update(overrides)
    config = AdapterConfig.model_validate(doc)
    ctx = FakeCtx()
    adapter = HermesAdapter(
        config,
        SecretStr("agmhr_test"),
        ctx=ctx,
        hermes_home=tmp_path / "home",
        start_threads=False,
        identity=identity or dict(PINNED),
    )
    adapter.install(ctx)
    return adapter


class Runner:
    """Drives the adapter callbacks in Hermes' orders and counts real executions."""

    def __init__(self, adapter: HermesAdapter) -> None:
        self.adapter = adapter
        self.executions = 0

    def _kw(self, tool: str, sid: str, tcid: str) -> dict[str, Any]:
        return {
            "tool_name": tool,
            "session_id": sid,
            "tool_call_id": tcid,
            "task_id": "task",
            "turn_id": f"{sid}:task:turn",
            "api_request_id": f"{sid}:task:turn:api:1",
            "telemetry_schema_version": "hermes.observer.v1",
        }

    def _execute(
        self, args: dict[str, Any], effect: Optional[Callable[[dict[str, Any]], None]]
    ) -> str:
        self.executions += 1
        if effect is not None:
            effect(args)
        return json.dumps({"success": True})

    def agent_loop(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        sid: str = "s1",
        tcid: str = "call_1",
        effect: Optional[Callable[[dict[str, Any]], None]] = None,
        modify: Optional[dict[str, Any]] = None,
    ) -> str:
        a = self.adapter
        kw = self._kw(tool, sid, tcid)
        blocked: dict[str, bool] = {}

        def terminal(next_args: Optional[dict[str, Any]] = None) -> str:
            final = dict(args if next_args is None else next_args)
            directive = a.pre_tool_call(args=final, **kw)
            if directive:
                result = json.dumps({"error": directive["message"]})
                blocked["b"] = True
                a.post_tool_call(args=final, result=result, status="blocked", **kw)
                return result
            if modify:
                final.update(modify)
            blocked["final"] = final  # type: ignore[assignment]
            return self._execute(final, effect)

        result = a.tool_execution(args=args, next_call=terminal, **kw)
        if not blocked.get("b"):
            executed = blocked.get("final", args)
            a.post_tool_call(args=executed, result=result, status="ok", duration_ms=1, **kw)
        return str(result)

    def direct(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        sid: str = "s1",
        tcid: str = "call_1",
        exec_args: Optional[dict[str, Any]] = None,
        effect: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        a = self.adapter
        kw = self._kw(tool, sid, tcid)
        directive = a.pre_tool_call(args=args, **kw)
        if directive:
            return json.dumps({"error": directive["message"]})
        mw_args = args if exec_args is None else exec_args
        return str(
            a.tool_execution(
                args=mw_args, next_call=lambda p=None: self._execute(mw_args, effect), **kw
            )
        )


def write_effect(path: Path) -> Callable[[dict[str, Any]], None]:
    def _effect(args: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(str(args.get("content", "")))

    return _effect


def wait_for(cond: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ---------------------------------------------------------------- allow


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_allow_executes_once_and_reports_with_permit(
    server: FakeAgenomic, tmp_path: Path, order: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "out.txt"
    args = {"path": str(target), "content": "hello"}
    result = getattr(runner, order)("write_file", args, effect=write_effect(target))

    assert json.loads(result) == {"success": True}
    assert runner.executions == 1
    assert target.read_text() == "hello"
    assert len(server.authorize_calls()) == 1
    reports = server.reports()
    assert len(reports) == 1
    body = reports[0].body
    assert body["logical_call_id"] == "call_1"
    assert body["permit"]["document"]["arguments_hash"] == arguments_hash(args)
    assert body["arguments"] == args
    assert body["is_error"] is False
    assert body["result_hash"].startswith("blake3:")
    assert body["result_preview"] is None
    adapter.exporter.flush(3.0)
    types = server.event_types()
    assert "tool.call.decision" in types
    assert "tool.call.executed" in types
    serialized = json.dumps(server.events)
    assert "hello" not in serialized  # content never leaves without redacted_preview


def test_session_admitted_before_authorize(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="s1", model="demo-model", platform="cli")
    Runner(adapter).agent_loop("read_file", {"path": "/tmp/x"})
    paths = [r.path for r in server.requests if r.method == "POST"]
    assert paths.index("/v1/hermes/runtime/hello") < paths.index("/v1/hermes/runtime/sessions")
    assert paths.index("/v1/hermes/runtime/sessions") < paths.index(
        "/v1/hermes/runtime/sessions/s1/actions/authorize"
    )
    assert len(server.calls("/v1/hermes/runtime/sessions")) == 1


# ---------------------------------------------------------------- deny


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_deny_never_calls_next_call(server: FakeAgenomic, tmp_path: Path, order: str) -> None:
    server.decide = lambda body: "deny"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "denied.txt"
    result = getattr(runner, order)(
        "write_file", {"path": str(target), "content": "x"}, effect=write_effect(target)
    )
    assert runner.executions == 0
    assert not target.exists()
    message = json.loads(result)["error"]
    assert message.startswith("Agenomic denied write_file: writes are not allowed (decision dec-")
    assert server.reports() == []


# ---------------------------------------------------------------- approval


def test_require_approval_blocks_then_retry_reuses_identity(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "approved.txt"
    args = {"path": str(target), "content": "x"}

    first = json.loads(
        runner.agent_loop("write_file", args, tcid="call_1", effect=write_effect(target))
    )
    approval_id = next(iter(server.approvals))
    assert first["error"] == APPROVAL_MESSAGE.format(approval_id=approval_id)
    assert runner.executions == 0
    assert not target.exists()

    # Still pending: blocked again, the gateway is not asked to authorize.
    second = json.loads(runner.agent_loop("write_file", args, tcid="call_2"))
    assert "still pending" in second["error"]
    assert len(server.authorize_calls()) == 1

    server.approve(approval_id)
    runner.agent_loop("write_file", args, tcid="call_3", effect=write_effect(target))
    assert runner.executions == 1
    assert target.read_text() == "x"
    retry = server.authorize_calls()[-1].body
    assert retry["tool_call_id"] == "call_1"  # identity of the pending action, not call_3
    assert retry["attempt"] == 1
    assert server.approvals[approval_id]["status"] == "consumed"
    assert server.reports()[0].body["logical_call_id"] == "call_1"


def test_rejected_approval_blocks_and_forgets_identity(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    args = {"path": "/tmp/r", "content": "x"}
    runner.agent_loop("write_file", args, tcid="call_1")
    approval_id = next(iter(server.approvals))
    server.approve(approval_id, "rejected")
    out = json.loads(runner.agent_loop("write_file", args, tcid="call_2"))
    assert out["error"].startswith(f"Agenomic approval {approval_id} was rejected")
    assert runner.executions == 0


# ---------------------------------------------------------------- failure safety


def test_middleware_gate_exception_blocks_without_next_call(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(adapter, "_execution_gate", boom)
    calls: list[int] = []
    out = adapter.tool_execution(
        tool_name="write_file",
        args={"path": "/tmp/a"},
        tool_call_id="c",
        session_id="s",
        next_call=lambda p=None: calls.append(1),
    )
    assert calls == []
    assert json.loads(out) == {"error": "Agenomic: no valid authorization for this action"}


def test_authorize_exception_blocks_in_both_paths(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(adapter, "authorize", boom)
    runner = Runner(adapter)
    assert "error" in json.loads(runner.agent_loop("write_file", {"path": "/tmp/a"}))
    assert "error" in json.loads(runner.direct("write_file", {"path": "/tmp/a"}, tcid="c2"))
    assert runner.executions == 0
    directive = adapter.pre_tool_call(
        tool_name="terminal", args={"command": "ls"}, session_id="s", tool_call_id="c3"
    )
    assert directive is not None
    assert directive["action"] == "block"


def test_argument_change_after_authorization_is_blocked(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "changed.txt"
    out = runner.direct(
        "write_file",
        {"path": str(target), "content": "safe"},
        exec_args={"path": str(target), "content": "evil"},
        effect=write_effect(target),
    )
    assert runner.executions == 0
    assert not target.exists()
    assert "changed after authorization" in json.loads(out)["error"]
    adapter.exporter.flush(3.0)
    assert "authorization.argument_mismatch" in server.event_types()


def test_post_tool_call_mismatch_emits_incident(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    # A foreign pre_tool_call "modify" applied after our authorization (agent loop order).
    runner.agent_loop("read_file", {"path": "/tmp/a"}, modify={"path": "/etc/passwd"})
    adapter.exporter.flush(3.0)
    incidents = [e for e in server.events if e["type"] == "authorization.argument_mismatch"]
    assert len(incidents) == 1
    assert incidents[0]["reason"] == "arguments changed after authorization (post_tool_call)"


def test_server_timeout_in_enforce_blocks(server: FakeAgenomic, tmp_path: Path) -> None:
    server.authorize_delay_s = 0.6
    adapter = make_adapter(server.url, tmp_path, timeouts={"decision_s": 0.2, "connect_s": 0.2})
    runner = Runner(adapter)
    out = json.loads(runner.agent_loop("write_file", {"path": "/tmp/t"}))
    assert (
        out["error"] == "Agenomic authorization unavailable (timeout); the action was not executed."
    )
    assert runner.executions == 0


def test_server_error_and_invalid_decision_block(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    server.authorize_status = 500
    assert (
        "unavailable (boom)"
        in json.loads(runner.agent_loop("write_file", {"path": "/tmp/a"}))["error"]
    )
    server.authorize_status = None
    server.decide = lambda body: "invalid"
    out = json.loads(runner.agent_loop("write_file", {"path": "/tmp/b"}, tcid="call_2"))
    assert "invalid_response" in out["error"]
    assert runner.executions == 0


def test_unreachable_server_with_unknown_mode_blocks(tmp_path: Path) -> None:
    adapter = make_adapter(
        "http://127.0.0.1:9", tmp_path, timeouts={"decision_s": 0.3, "connect_s": 0.3}
    )
    runner = Runner(adapter)
    out = json.loads(runner.agent_loop("write_file", {"path": "/tmp/a"}))
    assert "Agenomic authorization unavailable" in out["error"]
    assert runner.executions == 0
    adapter.shutdown()


def test_incompatible_hermes_blocks_in_enforce_only(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(
        server.url, tmp_path, identity={"version": "0.22.0", "release_date": None, "commit": None}
    )
    assert not adapter.hermes_compatible
    runner = Runner(adapter)
    out = json.loads(runner.agent_loop("read_file", {"path": "/tmp/a"}))
    assert "0.22.0 is not in the adapter compatibility table" in out["error"]
    server.effective_state = "shadow"
    adapter._effective_state = None
    runner.agent_loop("read_file", {"path": "/tmp/a"}, tcid="call_2")
    assert runner.executions == 1


# ---------------------------------------------------------------- modes


def test_shadow_never_blocks(server: FakeAgenomic, tmp_path: Path) -> None:
    server.effective_state = "shadow"
    server.decide = lambda body: "deny"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "shadow.txt"
    runner.agent_loop(
        "write_file", {"path": str(target), "content": "s"}, effect=write_effect(target)
    )
    assert runner.executions == 1
    assert target.exists()
    server.authorize_status = 503
    runner.agent_loop(
        "write_file",
        {"path": str(target), "content": "t"},
        tcid="call_2",
        effect=write_effect(target),
    )
    assert runner.executions == 2
    adapter.exporter.flush(3.0)
    decisions = [e for e in server.events if e["type"] == "tool.call.decision"]
    assert decisions[0]["extra"]["counterfactual"] == {"outcome": "deny", "reason_codes": []}
    assert "authorization.unavailable" in server.event_types()


def test_observe_never_calls_authorize(server: FakeAgenomic, tmp_path: Path) -> None:
    server.effective_state = "observe"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    runner.agent_loop("write_file", {"path": "/tmp/o", "content": "o"})
    runner.direct("terminal", {"command": "ls"}, tcid="call_2")
    assert runner.executions == 2
    assert server.authorize_calls() == []
    assert server.reports() == []
    adapter.exporter.flush(3.0)
    assert "tool.call.requested" in server.event_types()


# ---------------------------------------------------------------- local defence and delegation


def test_protected_path_denied_locally_even_if_server_allows(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    skill = tmp_path / "home" / "skills" / "evil" / "SKILL.md"
    out = json.loads(
        runner.agent_loop(
            "write_file", {"path": str(skill), "content": "x"}, effect=write_effect(skill)
        )
    )
    assert "protected Hermes paths" in out["error"]
    assert runner.executions == 0
    assert len(server.authorize_calls()) == 1  # the server still saw and recorded the request
    patch = "*** Begin Patch\n*** Update File: " + str(tmp_path / "home" / "config.yaml") + "\n"
    assert adapter.protected_targets("patch", {"patch": patch})
    assert adapter.protected_targets("read_file", {"path": str(skill)}) == []


def test_delegate_task_reserves_then_links_child_session(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="parent", platform="cli", model="m")
    Runner(adapter).agent_loop(
        "delegate_task", {"tasks": [{"goal": "a"}, {"goal": "b"}]}, sid="parent"
    )
    paths = [r.path for r in server.requests if r.method == "POST"]
    deleg = paths.index("/v1/hermes/runtime/sessions/parent/delegations")
    assert deleg < paths.index("/v1/hermes/runtime/sessions/parent/actions/authorize")
    assert server.calls("/delegations")[0].body == {"count": 2, "tool_call_id": "call_1"}
    adapter.subagent_start(
        parent_session_id="parent",
        child_session_id="child",
        child_subagent_id="sa-0-x",
        child_goal="a",
    )
    adapter.on_session_start(session_id="child", platform="subagent", model="m")
    child = [
        r.body
        for r in server.calls("/v1/hermes/runtime/sessions")
        if r.body["hermes_session_id"] == "child"
    ]
    assert child[0]["parent_hermes_session_id"] == "parent"
    assert child[0]["subagent_id"] == "sa-0-x"
    assert child[0]["delegation_id"]
    adapter.exporter.flush(3.0)
    started = [
        e
        for e in server.events
        if e["type"] == "session.started" and e["hermes_session_id"] == "child"
    ]
    assert started[0]["trace_id"] == "parent"


def test_delegate_control_actions_do_not_reserve(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    Runner(adapter).agent_loop("delegate_task", {"action": "list"})
    assert server.calls("/delegations") == []


def test_chain_run_twice_gets_new_attempt(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    runner.agent_loop("read_file", {"path": "/tmp/a"}, tcid="dup")
    runner.agent_loop("read_file", {"path": "/tmp/a"}, tcid="dup")
    attempts = [r.body["attempt"] for r in server.authorize_calls()]
    assert attempts == [1, 2]
    assert len(server.reports()) == 2


def test_report_failure_is_retried_not_reexecuted(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    original = adapter.client.report
    failures = {"n": 1}

    def flaky(sid: str, body: dict[str, Any]) -> dict[str, Any]:
        if failures["n"]:
            failures["n"] -= 1
            from agenomic.integrations.hermes.client import HermesApiError

            raise HermesApiError("unavailable", "down", 503)
        return original(sid, body)

    monkeypatch.setattr(adapter.client, "report", flaky)
    runner.agent_loop("read_file", {"path": "/tmp/a"})
    assert runner.executions == 1
    assert server.reports() == []
    adapter.tick()
    assert len(server.reports()) == 1
    assert runner.executions == 1
    adapter.exporter.flush(3.0)
    failed = [e for e in server.events if e["type"] == "action.report_failed"]
    assert failed[0]["extra"]["external_state"] == "unknown"


def test_redacted_preview_capture(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(
        server.url, tmp_path, capture={"content": "redacted_preview", "preview_chars": 60}
    )
    runner = Runner(adapter)
    runner.agent_loop(
        "terminal",
        {"command": "curl -H 'Authorization: Bearer abc.def' x", "api_key": "sk-supersecret123"},
    )
    adapter.exporter.flush(3.0)
    blob = json.dumps(server.events)
    assert "sk-supersecret123" not in blob
    assert "abc.def" not in blob
    requested = [e for e in server.events if e["type"] == "tool.call.requested"][0]
    assert "***" in requested["extra"]["previews"]["input"]
    assert len(requested["extra"]["previews"]["input"]) <= 60


# ---------------------------------------------------------------- llm_request, probes, commands


def test_llm_request_adds_header_only_for_gateway(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    request = {"model": "m", "messages": [], "extra_headers": {"X-Other": "1"}}
    out = adapter.llm_request(request=request, session_id="s1", base_url=server.model_base_url)
    assert out is not None
    assert out["request"]["extra_headers"] == {"X-Other": "1", "X-Agenomic-Hermes-Session": "s1"}
    assert {k: v for k, v in out["request"].items() if k != "extra_headers"} == {
        "model": "m",
        "messages": [],
    }
    assert request["extra_headers"] == {"X-Other": "1"}  # input untouched
    assert (
        adapter.llm_request(request=request, session_id="s1", base_url="https://api.openai.com/v1")
        is None
    )
    assert adapter.llm_request(request="x", session_id="s1", base_url=server.model_base_url) is None


def test_contract_probe_and_foreign_mutators(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    assert adapter._contracts["pre_tool_call"]
    assert adapter._contracts["tool_execution"]
    assert adapter.foreign_mutators() == []

    def shell(**kwargs: Any) -> None:
        return None

    shell.__qualname__ = "shell_hook[pre_tool_call:agenomic-hermes-guard]"

    def other_plugin(**kwargs: Any) -> dict[str, Any]:
        return {"action": "modify", "args": {}}

    manager = adapter.ctx._manager
    manager._hooks["pre_tool_call"] += [shell, other_plugin]
    manager._middleware.setdefault("tool_request", []).append(other_plugin)
    found = adapter.foreign_mutators()
    assert [(f["kind"], f["name"]) for f in found] == [
        ("hook", "pre_tool_call"),
        ("middleware", "tool_request"),
    ]
    adapter._ensure_started("cli")
    hello = server.calls("/hello")[0].body
    assert hello["adapter"] == {
        "version": "1.0.0",
        "config_schema": "agenomic.hermes.adapter_config/v1",
    }
    assert len(hello["foreign_mutators"]) == 2
    assert hello["contracts"]["observer_hooks"] == list(plugin_mod.OBSERVER_HOOKS)
    checks = {c["check"]: c["status"] for c in hello["compat_results"]}
    assert "hermes_version_compatible" in checks


def test_commands_pause_resume_unknown(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    server.commands = [
        {
            "id": "c1",
            "kind": "pause",
            "target_kind": "instance",
            "target_ref": "",
            "status": "requested",
        },
        {
            "id": "c2",
            "kind": "frobnicate",
            "target_kind": "instance",
            "target_ref": "",
            "status": "requested",
        },
    ]
    adapter.tick()
    acks = [(cid, b["status"]) for cid, b in server.acks]
    assert acks == [("c1", "received"), ("c1", "applied"), ("c2", "received"), ("c2", "refused")]
    status = json.loads(adapter.status_file.read_text())
    assert status["instance_status"] == "paused"
    assert status["loaded"] is True
    assert adapter.local_mode() == "enforce"
    server.commands = [
        {
            "id": "c3",
            "kind": "resume",
            "target_kind": "instance",
            "target_ref": "",
            "status": "requested",
        }
    ]
    adapter.tick()
    assert ("c3", "applied") in [(cid, b["status"]) for cid, b in server.acks]
    assert json.loads(adapter.status_file.read_text())["instance_status"] == "active"
    heartbeat = server.calls("/heartbeat")[-1].body
    assert set(heartbeat["exporter"]) == {"buffered", "dropped", "buffer_full", "last_flush_error"}


def test_cancel_subagent_applied_only_after_stop(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    interrupted: list[str] = []
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: interrupted.append(sid) or True)
    adapter.handle_command(
        {
            "id": "k1",
            "kind": "cancel",
            "target_kind": "subagent",
            "target_ref": "sa-1",
            "status": "requested",
        }
    )
    assert interrupted == ["sa-1"]
    assert [b["status"] for _, b in server.acks] == ["received"]
    adapter.subagent_start(parent_session_id="p", child_session_id="c", child_subagent_id="sa-1")
    adapter.subagent_stop(parent_session_id="p", child_session_id="c", child_status="interrupted")
    assert [b["status"] for _, b in server.acks] == ["received", "applied"]
    assert server.acks[-1][1]["detail"]["observed"] == "subagent_stop"


def test_cancel_subagent_not_running_is_refused(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: False)
    adapter.handle_command(
        {"id": "k2", "kind": "cancel", "target_kind": "subagent", "target_ref": "sa-9"}
    )
    assert server.acks[-1][1]["status"] == "refused"


def test_cancel_root_session_blocks_tools_until_terminal(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="root", platform="cli")
    agenomic_id = adapter._sessions["root"].agenomic_id
    adapter.handle_command(
        {
            "id": "k3",
            "kind": "cancel",
            "target_kind": "session",
            "target_ref": agenomic_id,
            "status": "requested",
        }
    )
    assert [b["status"] for _, b in server.acks] == ["received"]
    runner = Runner(adapter)
    out = json.loads(runner.agent_loop("read_file", {"path": "/tmp/a"}, sid="root"))
    assert "cancelled this session" in out["error"]
    adapter.on_session_end(session_id="root", completed=True, interrupted=False)
    assert [b["status"] for _, b in server.acks] == ["received"]
    adapter.on_session_end(session_id="root", completed=False, interrupted=True)
    assert [b["status"] for _, b in server.acks] == ["received", "applied"]


def test_observer_hooks_emit_events_and_never_raise(server: FakeAgenomic, tmp_path: Path) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.pre_api_request(
        session_id="s",
        api_request_id="a1",
        turn_id="t",
        model="m",
        provider="custom",
        request_messages=[{"role": "user", "content": "secret prompt"}],
    )
    adapter.post_api_request(
        session_id="s",
        api_request_id="a1",
        api_duration=0.5,
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )
    adapter.api_request_error(
        session_id="s", api_request_id="a2", error="not a dict", status_code=500
    )
    adapter.pre_auxiliary_call(session_id="s", aux_task="title")
    adapter.post_auxiliary_call(session_id="s", aux_task="title", usage=None)
    adapter.pre_approval_request(session_id="s", command="rm -rf /", surface="cli")
    adapter.post_approval_response(session_id="s", choice="deny", surface="cli")
    adapter.agent_loop_stopped(platform="telegram", reason="stop")
    adapter.on_session_reset(session_id="s2", old_session_id="s")
    adapter.on_session_finalize(session_id="s", reason="shutdown")
    adapter.on_session_finalize(platform="gateway", reason="shutdown")
    adapter.exporter.flush(3.0)
    types = server.event_types()
    for expected in (
        "model.call.started",
        "model.call.completed",
        "model.call.failed",
        "model.aux.started",
        "approval.requested",
        "approval.responded",
        "agent.loop_stopped",
        "session.reset",
        "session.finalized",
    ):
        assert expected in types
    completed = [e for e in server.events if e["type"] == "model.call.completed"][0]
    assert completed["usage"] == {"input_tokens": 3, "output_tokens": 2, "known": True}
    assert completed["latency_ms"] == 500
    assert "secret prompt" not in json.dumps(server.events)
    assert "rm -rf" not in json.dumps(server.events)
    ends = server.calls("/sessions/s/end")
    assert ends[-1].body == {"final": True, "status": "completed", "reason": "shutdown"}


def test_register_entry_point_function(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    monkeypatch.delenv("AGENOMIC_HERMES_CONFIG", raising=False)
    monkeypatch.setenv("AGENOMIC_HERMES_RUNTIME_TOKEN", "agmhr_entry")
    ctx = FakeCtx({"endpoint": server.url})
    plugin_mod.register(ctx)
    adapter = plugin_mod.current_adapter()
    assert adapter is not None
    assert adapter.ctx is ctx
    assert len(ctx._manager._hooks["pre_tool_call"]) == 1
    assert ctx.unload
    assert json.loads((tmp_path / "hh" / "agenomic" / "status.json").read_text())["loaded"]
    adapter.shutdown()

    monkeypatch.delenv("AGENOMIC_HERMES_RUNTIME_TOKEN")
    with pytest.raises(ConfigError, match="AGENOMIC_HERMES_RUNTIME_TOKEN"):
        plugin_mod.register(FakeCtx({"endpoint": server.url}))
    assert json.loads((tmp_path / "hh" / "agenomic" / "status.json").read_text())["loaded"] is False


def test_plugin_source_has_no_kind_markers() -> None:
    # Hermes classifies entry point plugins by scanning their first 8 KiB of source.
    source = Path(plugin_mod.__file__).read_text(encoding="utf-8")[:8192]
    for marker in (
        "register_memory_provider",
        "MemoryProvider",
        "register_cron_scheduler",
        "CronScheduler",
    ):
        assert marker not in source
    assert not ("register_provider" in source and "ProviderProfile" in source)


def test_heartbeat_thread_starts_and_writes_status(server: FakeAgenomic, tmp_path: Path) -> None:
    config = AdapterConfig.model_validate(
        {"endpoint": server.url, "buffer": {"flush_interval_s": 0.05}}
    )
    ctx = FakeCtx()
    adapter = HermesAdapter(
        config, SecretStr("agmhr_t"), ctx=ctx, hermes_home=tmp_path / "h", identity=dict(PINNED)
    )
    adapter.install(ctx)
    server.heartbeat_interval_secs = 1
    server_state = adapter.status_file
    adapter.on_session_start(session_id="s", platform="cli")
    assert adapter._thread is not None
    assert adapter._thread.is_alive()
    assert wait_for(lambda: len(server.calls("/heartbeat")) >= 1)
    assert json.loads(server_state.read_text())["effective_state"] == "enforce"
    adapter.shutdown()


def test_staged_skill_write_becomes_a_proposal_never_an_approval(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    record = {
        "id": "ab12cd34",
        "summary": "new skill",
        "payload": {
            "action": "create",
            "name": "summarize",
            "content": "---\nname: summarize\n---\nbody",
        },
    }
    fake = types.ModuleType("tools.write_approval")
    fake.list_pending = lambda subsystem: [record] if subsystem == "skills" else []  # type: ignore[attr-defined]
    fake.skill_pending_diff = lambda r: r["payload"]["content"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.write_approval", fake)
    adapter = make_adapter(server.url, tmp_path)
    adapter.post_tool_call(
        tool_name="skill_manage",
        args={"action": "create", "name": "summarize"},
        result=json.dumps({"success": True, "staged": True, "pending_id": "ab12cd34"}),
        session_id="s1",
        tool_call_id="call_s",
        status="ok",
    )
    sent = server.calls("/proposals")
    assert len(sent) == 1
    assert sent[0].body["kind"] == "skill"
    assert sent[0].body["target"] == "skills/summarize/SKILL.md"
    assert sent[0].body["content"].startswith("---")
    adapter.post_tool_call(
        tool_name="skill_manage",
        args={},
        result=json.dumps({"success": True}),
        session_id="s1",
        tool_call_id="call_t",
        status="ok",
    )
    assert len(server.calls("/proposals")) == 1, "an unstaged write is not proposed"
    assert not [r for r in server.requests if "decide" in r.path or "publish" in r.path]


class _GatelessCtx(FakeCtx):
    def register_hook(self, name: str, cb: Any) -> None:
        if name == "pre_tool_call":
            raise RuntimeError("no gate")
        super().register_hook(name, cb)

    def register_middleware(self, kind: str, cb: Any) -> None:
        raise RuntimeError("no middleware")


def test_guard_keeps_blocking_when_no_gate_registers(server: FakeAgenomic, tmp_path: Path) -> None:
    config = AdapterConfig.model_validate({"endpoint": server.url})
    ctx = _GatelessCtx()
    adapter = HermesAdapter(
        config,
        SecretStr("agmhr_t"),
        ctx=ctx,
        hermes_home=tmp_path / "h",
        start_threads=False,
        identity=dict(PINNED),
    )
    adapter.install(ctx)
    status = json.loads(adapter.status_file.read_text())
    assert status["loaded"] is False


def test_delegation_reservation_is_queued_only_after_the_action_is_allowed(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.decide = lambda body: "deny"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "a"}]}, sid="p", tcid="d1")
    assert len(server.calls("/delegations")) == 1
    assert not adapter._delegations.get("p"), "a denied action leaves no reservation for a child"
    assert not adapter._provisional_delegations

    server.decide = lambda body: "require_approval"
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "b"}]}, sid="p", tcid="d2")
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "b"}]}, sid="p", tcid="d2")
    assert len(server.calls("/delegations")) == 2, "a retry after approval reuses its reservation"
    assert not adapter._delegations.get("p")

    server.decide = lambda body: "allow"
    for approval in server.approvals.values():
        approval["status"] = "approved"
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "b"}]}, sid="p", tcid="d2")
    assert len(server.calls("/delegations")) == 2
    assert len(adapter._delegations["p"]) == 1, "the allowed action queues its reservation"


def test_cached_authorization_never_serves_another_session(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    kw = Runner(adapter)._kw("read_file", "sess-a", "call_shared")
    assert adapter.pre_tool_call(args={"path": "/tmp/a"}, **kw) is None
    other = Runner(adapter)._kw("read_file", "sess-b", "call_shared")
    assert adapter.pre_tool_call(args={"path": "/tmp/a"}, **other) is None
    sessions = [r.path.split("/sessions/")[1].split("/")[0] for r in server.authorize_calls()]
    assert sessions == ["sess-a", "sess-b"], "each session asks the gateway for its own call"
