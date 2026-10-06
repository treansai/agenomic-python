"""Adapter behaviour against a fake Agenomic runtime API (no Hermes needed).

The two Hermes call orders are simulated faithfully:

* agent loop (``agent/tool_executor.py``): ``tool_execution`` middleware wraps
  a terminal that runs ``pre_tool_call`` then the tool;
* direct dispatch (``model_tools.handle_function_call``): ``pre_tool_call``
  first, then the middleware around the tool.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from hermes_fakes import FakeAgenomic, FakeCtx
from pydantic import SecretStr

from agenomic.integrations.hermes import guard as guard_mod
from agenomic.integrations.hermes import plugin as plugin_mod
from agenomic.integrations.hermes.canonical import arguments_hash
from agenomic.integrations.hermes.client import HermesApiError
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


def _approved_write(
    server: FakeAgenomic, tmp_path: Path
) -> tuple[HermesAdapter, Runner, dict[str, Any], Path, str]:
    """A write that required an approval which a human then granted."""
    server.decide = lambda body: "require_approval"
    server.replay_consumed = True
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "approved.txt"
    args = {"path": str(target), "content": "x"}
    runner.agent_loop("write_file", args, tcid="call_1")
    approval_id = next(iter(server.approvals))
    server.approve(approval_id)
    return adapter, runner, args, target, approval_id


def test_one_approval_authorizes_one_concurrent_invocation(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, runner, args, target, approval_id = _approved_write(server, tmp_path)
    original = adapter.client.authorize
    concurrent: list[str] = []
    interleaved: list[bool] = []

    def authorize_then_interleave(sid: str, body: Any) -> Any:
        answer = original(sid, body)
        if not interleaved:
            interleaved.append(True)
            # A second identical invocation (another tool_call_id) arrives while the
            # first retry holds the granted approval and has not executed yet.
            concurrent.append(
                runner.agent_loop("write_file", args, tcid="call_3", effect=write_effect(target))
            )
        return answer

    monkeypatch.setattr(adapter.client, "authorize", authorize_then_interleave)
    runner.agent_loop("write_file", args, tcid="call_2", effect=write_effect(target))

    assert runner.executions == 1
    assert target.read_text() == "x"
    message = json.loads(concurrent[0])["error"]
    assert message == plugin_mod.APPROVAL_IN_USE_MESSAGE.format(approval_id=approval_id)
    assert [c.body["tool_call_id"] for c in server.authorize_calls()] == ["call_1", "call_1"]
    assert len(server.reports()) == 1


def test_consumed_approval_is_not_reused_by_a_later_identical_invocation(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter, runner, args, target, approval_id = _approved_write(server, tmp_path)
    runner.agent_loop("write_file", args, tcid="call_2", effect=write_effect(target))
    assert runner.executions == 1
    later = json.loads(
        runner.agent_loop("write_file", args, tcid="call_3", effect=write_effect(target))
    )
    assert runner.executions == 1
    assert target.read_text() == "x"
    retry = server.authorize_calls()[-1].body
    assert retry["tool_call_id"] == "call_3", "a new logical action, not the approved one"
    fresh = server.pending_by_call["call_3"]
    assert fresh != approval_id
    assert later["error"] == APPROVAL_MESSAGE.format(approval_id=fresh)


def test_lost_answer_releases_the_approval_for_the_next_retry(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, runner, args, target, approval_id = _approved_write(server, tmp_path)
    original = adapter.client.authorize
    lost: list[bool] = []

    def lose_first_answer(sid: str, body: Any) -> Any:
        answer = original(sid, body)
        if not lost:
            lost.append(True)
            raise HermesApiError("timeout", "authorize timed out", 0)
        return answer

    monkeypatch.setattr(adapter.client, "authorize", lose_first_answer)
    first = json.loads(runner.agent_loop("write_file", args, tcid="call_2"))
    assert "error" in first
    assert server.approvals[approval_id]["status"] == "consumed"
    runner.agent_loop("write_file", args, tcid="call_3", effect=write_effect(target))
    assert runner.executions == 1
    assert server.authorize_calls()[-1].body["tool_call_id"] == "call_1"


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


def test_cancel_subagent_applied_when_only_finalize_reports_the_end(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: True)
    adapter.subagent_start(parent_session_id="p", child_session_id="c", child_subagent_id="sa-2")
    adapter.on_session_start(session_id="c", platform="subagent")
    adapter.handle_command(
        {
            "id": "k4",
            "kind": "cancel",
            "target_kind": "subagent",
            "target_ref": "sa-2",
            "status": "requested",
        }
    )
    assert [b["status"] for _, b in server.acks] == ["received"]
    # No subagent_stop and no interrupted turn end: only the final end reaches the plugin.
    adapter.on_session_finalize(session_id="c", reason="exit")
    assert [b["status"] for _, b in server.acks] == ["received", "applied"]
    assert server.acks[-1][1]["detail"]["observed"] == "on_session_finalize"
    ends = [(c.body["final"], c.body["status"]) for c in server.calls("/end")]
    assert ends == [(True, "cancelled")], "the gateway sees a terminal cancelled session"
    assert not adapter._cancel_subagents


def test_finalize_without_a_pending_cancel_is_completed(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="s", platform="cli")
    adapter.on_session_finalize(session_id="s", reason="exit")
    assert [c.body["status"] for c in server.calls("/end")] == ["completed"]
    assert server.acks == []


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
    # The cancel-driven interrupt is reported as "cancelled", a terminal state for the
    # gateway, before the ack: the gateway only applies a cancel whose session ended.
    ends = [c.body["status"] for c in server.calls("/end")]
    assert ends == ["completed", "cancelled"]


def test_interrupt_without_a_pending_cancel_stays_interrupted(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="s", platform="cli")
    adapter.on_session_end(session_id="s", completed=False, interrupted=True)
    ends = [c.body["status"] for c in server.calls("/end")]
    assert ends == ["interrupted"]


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
    # Read like the guard does: the heartbeat thread may be replacing the file right now.
    status = guard_mod._read_status(server_state)
    assert status is not None
    assert status["effective_state"] == "enforce"
    adapter.shutdown()


def test_status_write_retries_a_transient_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = plugin_mod.os.replace
    calls: list[int] = []

    def flaky_replace(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "sharing violation")
        real_replace(src, dst)

    monkeypatch.setattr(plugin_mod.os, "replace", flaky_replace)
    path = tmp_path / "agenomic" / "status.json"
    plugin_mod.write_status(path, loaded=True, instance_status="active", effective_state="observe")
    assert len(calls) == 2
    assert json.loads(path.read_text())["effective_state"] == "observe"
    assert [p.name for p in path.parent.iterdir()] == ["status.json"]


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


@pytest.mark.parametrize("missing", ["pre_tool_call", "tool_execution"])
def test_guard_keeps_blocking_when_one_gate_is_missing(
    server: FakeAgenomic, tmp_path: Path, missing: str
) -> None:
    class _OneGateCtx(FakeCtx):
        def register_hook(self, name: str, cb: Any) -> None:
            if name == missing:
                raise RuntimeError("no gate")
            super().register_hook(name, cb)

        def register_middleware(self, kind: str, cb: Any) -> None:
            if kind == missing:
                raise RuntimeError("no middleware")
            super().register_middleware(kind, cb)

    config = AdapterConfig.model_validate({"endpoint": server.url})
    ctx = _OneGateCtx()
    adapter = HermesAdapter(
        config,
        SecretStr("agmhr_t"),
        ctx=ctx,
        hermes_home=tmp_path / "h",
        start_threads=False,
        identity=dict(PINNED),
    )
    adapter.install(ctx)
    assert not adapter._contracts[missing]
    status = json.loads(adapter.status_file.read_text())
    assert status["loaded"] is False, f"without {missing} the guard stays closed"


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


def test_concurrent_identical_delegations_each_reserve_for_their_own_children(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="p", platform="cli", model="m")
    runner = Runner(adapter)
    args: dict[str, Any] = {"tasks": [{"goal": "same"}]}
    original = adapter.client.authorize
    interleaved: list[str] = []

    def authorize_then_interleave(sid: str, body: Any) -> Any:
        answer = original(sid, body)
        if body["tool_call_id"] == "d1" and not interleaved:
            interleaved.append("d2")
            # A second identical delegate_task (another tool_call_id) is decided and
            # allowed while the first one has not settled yet.
            runner.agent_loop("delegate_task", args, sid="p", tcid="d2")
        return answer

    monkeypatch.setattr(adapter.client, "authorize", authorize_then_interleave)
    runner.agent_loop("delegate_task", args, sid="p", tcid="d1")
    assert runner.executions == 2, "both invocations were allowed"
    reserved = server.calls("/delegations")
    assert [r.body["tool_call_id"] for r in reserved] == ["d1", "d2"], "one reservation each"
    assert len(adapter._delegations["p"]) == 2
    assert not adapter._provisional_delegations

    for child in ("c1", "c2"):
        adapter.subagent_start(parent_session_id="p", child_session_id=child)
        adapter.on_session_start(session_id=child, platform="subagent", model="m")
    admitted = {
        r.body["hermes_session_id"]: r.body.get("delegation_id")
        for r in server.calls("/v1/hermes/runtime/sessions")
    }
    assert admitted["c1"]
    assert admitted["c2"]
    assert admitted["c1"] != admitted["c2"], "each child consumes its own reservation"


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


def test_heartbeat_stays_within_the_guard_deadline(server: FakeAgenomic, tmp_path: Path) -> None:
    server.heartbeat_interval_secs = 300
    adapter = make_adapter(server.url, tmp_path)
    adapter.tick()
    assert adapter._heartbeat_s <= plugin_mod.DEFAULT_MAX_AGE_S / 3


def test_heartbeat_before_hello_stays_within_a_short_guard_deadline(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Until the server sets an interval the default one must still refresh the status file
    # within a third of the guard's deadline.
    monkeypatch.setenv("AGENOMIC_HERMES_GUARD_MAX_AGE_S", "9")
    adapter = make_adapter(server.url, tmp_path)
    assert adapter._heartbeat_s <= 3.0
    monkeypatch.setenv("AGENOMIC_HERMES_GUARD_MAX_AGE_S", "0.5")
    assert plugin_mod._guard_max_age_s() / 3 >= 1.0, "the clamp matches the guard minimum"


def test_failed_command_ack_is_retried(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    real = adapter.client.ack_command
    failures = {"left": 1}

    def flaky(command_id: str, status: str, detail: Any) -> Any:
        if status == "applied" and failures["left"]:
            failures["left"] -= 1
            raise HermesApiError("unreachable", "connection refused", 0)
        return real(command_id, status, detail)

    monkeypatch.setattr(adapter.client, "ack_command", flaky)
    adapter.handle_command({"id": "c9", "kind": "pause", "target_kind": "instance"})
    assert [(c, b.get("status")) for c, b in server.acks] == [("c9", "received")]
    adapter.tick()
    assert ("c9", "applied") in [(c, b.get("status")) for c, b in server.acks]


def test_post_status_of_another_session_never_hides_an_execution(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    blocked_kw = runner._kw("read_file", "sess-x", "call_same")
    adapter.post_tool_call(args={"path": "/tmp/a"}, result="{}", status="blocked", **blocked_kw)
    runner.agent_loop("read_file", {"path": "/tmp/a"}, sid="sess-y", tcid="call_same")
    assert runner.executions == 1
    assert len(server.reports()) == 1, "the executed call of sess-y is reported"


def test_rejected_approval_drops_its_delegation_reservation(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "x"}]}, sid="p", tcid="d1")
    assert adapter._provisional_delegations
    for approval in server.approvals.values():
        approval["status"] = "rejected"
    runner.agent_loop("delegate_task", {"tasks": [{"goal": "x"}]}, sid="p", tcid="d1")
    assert not adapter._provisional_delegations, "a rejected action never keeps its reservation"


def test_stale_blocked_status_never_describes_a_later_execution(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    kw = runner._kw("read_file", "s1", "call_reused")
    adapter.post_tool_call(args={"path": "/tmp/a"}, result="{}", status="blocked", **kw)
    runner.direct("read_file", {"path": "/tmp/a"}, tcid="call_reused")
    assert runner.executions == 1
    assert len(server.reports()) == 1, "the later execution is reported"


def test_later_ack_supersedes_a_queued_earlier_one(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    real = adapter.client.ack_command

    def flaky(command_id: str, status: str, detail: Any) -> Any:
        if status == "received":
            raise HermesApiError("unreachable", "connection refused", 0)
        return real(command_id, status, detail)

    monkeypatch.setattr(adapter.client, "ack_command", flaky)
    adapter.handle_command({"id": "c7", "kind": "pause", "target_kind": "instance"})
    assert not adapter._ack_retries, "received is dropped once applied was accepted"


def test_failed_hello_after_a_mutator_change_is_resent(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.tick()
    assert server.calls("/hello")[-1].body["foreign_mutators"] == []

    def other_plugin(**kwargs: Any) -> dict[str, Any]:
        return {"action": "modify", "args": {}}

    adapter.ctx._manager._hooks["pre_tool_call"].append(other_plugin)
    real = adapter.client.hello
    failures = {"left": 1}

    def flaky(body: Any) -> Any:
        if failures["left"]:
            failures["left"] -= 1
            raise HermesApiError("unreachable", "connection refused", 0)
        return real(body)

    monkeypatch.setattr(adapter.client, "hello", flaky)
    adapter.tick()
    assert failures["left"] == 0, "the changed mutator list triggered a hello"
    adapter.tick()
    hello = server.calls("/hello")[-1].body
    assert len(hello["foreign_mutators"]) == 1, "the server learns about the new mutator"


@pytest.mark.parametrize("outcome", ["blocked", "raised", "error"])
def test_unexecuted_delegation_drops_its_reservation(
    server: FakeAgenomic, tmp_path: Path, outcome: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter.on_session_start(session_id="parent", platform="cli", model="m")
    args: dict[str, Any] = {"tasks": [{"goal": "a"}]}
    kw = Runner(adapter)._kw("delegate_task", "parent", "d1")

    def terminal(next_args: Optional[dict[str, Any]] = None) -> str:
        assert adapter.pre_tool_call(args=args, **kw) is None
        if outcome == "blocked":
            # Another Hermes hook blocks the call after Agenomic allowed it.
            result = json.dumps({"error": "blocked by another plugin"})
            adapter.post_tool_call(args=args, result=result, status="blocked", **kw)
            return result
        if outcome == "raised":
            raise RuntimeError("delegation failed")
        return json.dumps({"error": "no child started"})

    try:
        adapter.tool_execution(args=args, next_call=terminal, **kw)
    except RuntimeError:
        assert outcome == "raised"
    assert len(server.calls("/delegations")) == 1
    assert not adapter._delegations.get("parent"), "no reservation outlives the failed call"

    adapter.subagent_start(
        parent_session_id="parent",
        child_session_id="later-child",
        child_subagent_id="sa-1",
        child_goal="b",
    )
    adapter.on_session_start(session_id="later-child", platform="subagent", model="m")
    child = [
        r.body
        for r in server.calls("/v1/hermes/runtime/sessions")
        if r.body["hermes_session_id"] == "later-child"
    ]
    assert "delegation_id" not in child[0], "an unrelated child never takes a stale delegation"


def test_shutdown_makes_the_guard_block(server: FakeAgenomic, tmp_path: Path) -> None:
    config = AdapterConfig.model_validate(
        {"endpoint": server.url, "buffer": {"flush_interval_s": 0.05}}
    )
    ctx = FakeCtx()
    adapter = HermesAdapter(
        config, SecretStr("agmhr_t"), ctx=ctx, hermes_home=tmp_path / "h", identity=dict(PINNED)
    )
    adapter.install(ctx)
    server.heartbeat_interval_secs = 1
    adapter.on_session_start(session_id="s", platform="cli")
    assert wait_for(lambda: len(server.calls("/heartbeat")) >= 1)
    status = guard_mod._read_status(adapter.status_file)
    assert os.environ[guard_mod.GUARD_EPOCH_ENV] == adapter._guard_epoch
    epoch = adapter._guard_epoch
    assert guard_mod.evaluate(status, epoch=epoch, max_age_s=60) is None, (
        "the loaded adapter allows"
    )
    assert guard_mod.evaluate(status, epoch="restarted", max_age_s=60) is not None

    assert ctx.unload
    for callback in ctx.unload:
        callback()
    assert adapter._thread is not None
    assert not adapter._thread.is_alive()
    status = guard_mod._read_status(adapter.status_file)
    assert status is not None
    assert status["loaded"] is False
    assert guard_mod.evaluate(status, epoch=epoch, max_age_s=60) is not None, "blocks after unload"
    adapter.tick()  # a late heartbeat never reopens the guard
    assert guard_mod._read_status(adapter.status_file)["loaded"] is False  # type: ignore[index]


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_new_mutator_is_confirmed_before_authorizing(
    server: FakeAgenomic, tmp_path: Path, order: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")  # the first hello; no heartbeat tick follows
    assert server.calls("/hello")[-1].body["foreign_mutators"] == []

    def other_plugin(**kwargs: Any) -> dict[str, Any]:
        return {"action": "modify", "args": {}}

    adapter.ctx._manager._hooks["pre_tool_call"].append(other_plugin)
    result = getattr(Runner(adapter), order)("read_file", {"path": "/tmp/x"})
    assert json.loads(result) == {"success": True}
    paths = [r.path for r in server.requests if r.method == "POST"]
    last_hello = len(paths) - 1 - paths[::-1].index("/v1/hermes/runtime/hello")
    first_authorize = next(i for i, p in enumerate(paths) if p.endswith("/actions/authorize"))
    assert last_hello < first_authorize, "hello with the new mutator precedes the authorization"
    assert len(server.calls("/hello")[-1].body["foreign_mutators"]) == 1


@pytest.mark.parametrize("state", ["enforce", "shadow"])
def test_unconfirmed_mutator_change_blocks_in_enforce(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")  # the first hello; no heartbeat tick follows

    def other_plugin(**kwargs: Any) -> dict[str, Any]:
        return {"action": "modify", "args": {}}

    adapter.ctx._manager._hooks["pre_tool_call"].append(other_plugin)

    def down(body: Any) -> Any:
        raise HermesApiError("unreachable", "connection refused", 0)

    monkeypatch.setattr(adapter.client, "hello", down)
    runner = Runner(adapter)
    result = runner.direct("read_file", {"path": "/tmp/x"})
    adapter.exporter.flush(3.0)
    decisions = [e for e in server.events if e.get("type") == "tool.call.decision"]
    assert any(e["extra"].get("foreign_mutators_unconfirmed") for e in decisions)
    if state == "enforce":
        assert runner.executions == 0
        assert "not confirmed" in json.loads(result)["error"]
        assert server.authorize_calls() == []
    else:
        assert runner.executions == 1, "shadow records the change but does not block"
        assert len(server.authorize_calls()) == 1


def test_shutdown_closes_the_client_and_releases_the_atexit_hook(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks: list[Callable[[], None]] = []
    monkeypatch.setattr(plugin_mod.atexit, "register", hooks.append)
    monkeypatch.setattr(
        plugin_mod.atexit, "unregister", lambda f: [hooks.remove(h) for h in list(hooks) if h == f]
    )
    config = AdapterConfig.model_validate(
        {"endpoint": server.url, "buffer": {"flush_interval_s": 0.05}}
    )
    ctx = FakeCtx()
    adapter = HermesAdapter(
        config, SecretStr("agmhr_t"), ctx=ctx, hermes_home=tmp_path / "h", identity=dict(PINNED)
    )
    adapter.install(ctx)
    adapter.on_session_start(session_id="s", platform="cli")
    assert hooks == [adapter.shutdown]
    assert not adapter.client.closed

    closes: list[float] = []
    original_close = adapter.exporter.close
    monkeypatch.setattr(
        adapter.exporter, "close", lambda timeout=None: closes.append(1) or original_close(timeout)
    )
    adapter.shutdown()
    assert hooks == [], "the unloaded adapter is no longer kept alive by atexit"
    assert adapter.client.closed
    assert closes == [1]
    adapter.shutdown()  # idempotent: nothing is drained or closed twice
    assert closes == [1]
    # A callback after the unload neither restarts the adapter nor runs the tool.
    out = Runner(adapter).agent_loop("terminal", {"command": "ls"}, sid="s", tcid="late")
    assert "error" in json.loads(out)
    assert hooks == []


# ---------------------------------------------------------------- shadow never changes execution


def _local_decisions(server: FakeAgenomic, reason: str) -> list[dict[str, Any]]:
    return [
        e
        for e in server.events
        if e.get("type") == "tool.call.decision" and e.get("reason") == reason
    ]


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
@pytest.mark.parametrize("state", ["observe", "shadow", "enforce"])
@pytest.mark.parametrize(
    "bad",
    [float("nan"), 10**400, {"a", "b"}, object()],
    ids=["nan", "huge_int", "set", "object"],
)
def test_arguments_without_canonical_form_block_only_in_enforce(
    server: FakeAgenomic, tmp_path: Path, order: str, state: str, bad: object
) -> None:
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    runner = Runner(adapter)
    result = getattr(runner, order)("read_file", {"path": "/tmp/x", "injected": bad})
    adapter.exporter.flush(3.0)
    recorded = _local_decisions(server, "arguments_not_canonical")
    assert len(recorded) == 1, "recorded once, whichever gate sees the call first"
    event = recorded[0]
    assert event["decision"] == "deny"
    assert event["extra"]["local"] is True
    assert event["extra"]["local_mode"] == state
    assert server.authorize_calls() == []
    if state == "enforce":
        assert runner.executions == 0
        assert "no canonical form" in json.loads(result)["error"]
        assert "counterfactual" not in event["extra"]
    else:
        assert runner.executions == 1, f"{state} never changes execution"
        assert event["extra"]["counterfactual"] == {
            "outcome": "deny",
            "reason_codes": ["arguments_not_canonical"],
        }


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
@pytest.mark.parametrize("state", ["observe", "shadow", "enforce"])
def test_reused_call_id_with_non_canonical_arguments_is_recorded_each_time(
    server: FakeAgenomic, tmp_path: Path, order: str, state: str
) -> None:
    # A later invocation reusing a (session, tool, tool_call_id) is a new action: it gets
    # its own audit record instead of being taken for the second gate of the first one.
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    runner = Runner(adapter)
    for _ in range(2):
        getattr(runner, order)("read_file", {"path": "/tmp/x", "bad": float("nan")}, tcid="call_1")
    adapter.exporter.flush(3.0)
    assert len(_local_decisions(server, "arguments_not_canonical")) == 2


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_pending_approval_from_enforce_never_blocks_in_shadow(
    server: FakeAgenomic, tmp_path: Path, order: str
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    args = {"path": "/tmp/x", "content": "x"}
    first = json.loads(getattr(runner, order)("write_file", args, tcid="call_1"))
    assert "approval" in first["error"]
    assert runner.executions == 0

    server.effective_state = "shadow"
    adapter.tick()  # the server now answers shadow
    assert adapter.local_mode() == "shadow"
    getattr(runner, order)("write_file", args, tcid="call_2")
    assert runner.executions == 1, "a still pending approval does not block in shadow"
    assert server.authorize_calls()[-1].body["tool_call_id"] == "call_2"
    assert adapter._pending, "the enforce approval stays for a later enforce"


@pytest.mark.parametrize("state", ["shadow", "enforce"])
def test_refused_delegation_is_recorded_and_blocks_only_in_enforce(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    monkeypatch.setattr(
        adapter.client,
        "reserve_delegation",
        lambda sid, body: {"decision": "deny", "explanation": "delegation limit reached"},
    )
    runner = Runner(adapter)
    runner.direct("delegate_task", {"tasks": [{"goal": "a"}]})
    adapter.exporter.flush(3.0)
    assert _local_decisions(server, "Agenomic denied delegate_task: delegation limit reached")
    assert runner.executions == (1 if state == "shadow" else 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_status_file_is_owner_only_even_over_a_leftover_temporary(tmp_path: Path) -> None:
    import os
    import threading

    path = tmp_path / "agenomic" / "status.json"
    path.parent.mkdir()
    leftover = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    leftover.write_text("stale")
    os.chmod(leftover, 0o666)
    plugin_mod.write_status(path, loaded=True, instance_status="active", effective_state="observe")
    assert oct(path.stat().st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------- subagent cancel races


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
@pytest.mark.parametrize("state", ["observe", "shadow", "enforce"])
@pytest.mark.parametrize("known", ["session", "link_only"])
def test_pending_subagent_cancel_blocks_child_calls_in_every_mode(
    server: FakeAgenomic,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    state: str,
    known: str,
) -> None:
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == state
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: True)
    adapter.subagent_start(parent_session_id="p", child_session_id="c", child_subagent_id="sa-1")
    if known == "session":
        adapter.on_session_start(session_id="c", platform="subagent")
    adapter.handle_command(
        {
            "id": "k9",
            "kind": "cancel",
            "target_kind": "subagent",
            "target_ref": "sa-1",
            "status": "requested",
        }
    )
    runner = Runner(adapter)
    target = tmp_path / "raced.txt"
    # The interrupt is asynchronous: the child's next tool call races with it.
    out = json.loads(
        getattr(runner, order)(
            "write_file",
            {"path": str(target), "content": "x"},
            sid="c",
            effect=write_effect(target),
        )
    )
    assert runner.executions == 0, f"a pending subagent cancel blocks in {state}"
    assert not target.exists()
    assert out["error"] == "Agenomic cancelled this subagent; the action was not executed."
    assert server.authorize_calls() == []
    adapter.exporter.flush(3.0)
    recorded = _local_decisions(server, out["error"])
    assert len(recorded) == 1
    assert recorded[0]["extra"]["local"] is True
    assert recorded[0]["extra"]["reason_codes"] == ["cancel_pending"]
    assert [b["status"] for _, b in server.acks] == ["received"], "not applied before the end"
    # Another session is not affected.
    getattr(runner, order)("read_file", {"path": "/tmp/a"}, sid="p", tcid="call_2")
    assert runner.executions == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symbolic links")
def test_status_write_ignores_a_link_planted_at_the_old_temporary_name(tmp_path: Path) -> None:
    import os
    import threading

    path = tmp_path / "agenomic" / "status.json"
    path.parent.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    planted = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    planted.symlink_to(victim)
    plugin_mod.write_status(path, loaded=True, instance_status="active", effective_state="observe")
    assert victim.read_text() == "precious"
    assert json.loads(path.read_text())["loaded"] is True
    assert oct(path.stat().st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------- concurrent approvals


def test_every_concurrently_issued_approval_keeps_its_identity(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "approved.txt"
    args = {"path": str(target), "content": "x"}
    original = adapter.client.authorize
    inner: list[str] = []

    def answer_then_interleave(sid: str, body: Any) -> Any:
        answer = original(sid, body)
        if body["tool_call_id"] == "call_a" and not inner:
            # Both identical calls reached the gateway before either answer was stored:
            # call_b's approval is stored first, then call_a's answer arrives.
            inner.append(runner.agent_loop("write_file", args, tcid="call_b"))
        return answer

    monkeypatch.setattr(adapter.client, "authorize", answer_then_interleave)
    outer = json.loads(runner.agent_loop("write_file", args, tcid="call_a"))
    monkeypatch.undo()
    a1 = server.pending_by_call["call_b"]  # stored first
    a2 = server.pending_by_call["call_a"]
    assert a1 != a2
    assert json.loads(inner[0])["error"] == APPROVAL_MESSAGE.format(approval_id=a1)
    assert outer["error"] == APPROVAL_MESSAGE.format(approval_id=a2)
    server.approve(a2)  # only the approval whose message named a2 is granted
    runner.agent_loop("write_file", args, tcid="call_c", effect=write_effect(target))
    assert runner.executions == 1
    assert target.read_text() == "x"
    retry = server.authorize_calls()[-1].body
    assert retry["tool_call_id"] == "call_a", "resumed under a2's identity, not a1's"
    assert server.approvals[a2]["status"] == "consumed"
    assert server.approvals[a1]["status"] == "pending"
    assert server.reports()[0].body["logical_call_id"] == "call_a"
    remaining = adapter._pending[("s1", "write_file", arguments_hash(args))]
    assert [p.approval_id for p in remaining] == [a1], "a1 still waits for its own retry"

    still = json.loads(runner.agent_loop("write_file", args, tcid="call_d"))
    assert "still pending" in still["error"]
    assert a1 in still["error"]
    assert runner.executions == 1
    server.approve(a1)
    runner.agent_loop("write_file", args, tcid="call_e", effect=write_effect(target))
    assert runner.executions == 2
    assert server.authorize_calls()[-1].body["tool_call_id"] == "call_b"
    assert not adapter._pending


def test_concurrent_approvals_each_keep_their_delegation_reservation(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    args: dict[str, Any] = {"tasks": [{"goal": "same"}]}
    original = adapter.client.authorize
    inner: list[str] = []

    def answer_then_interleave(sid: str, body: Any) -> Any:
        answer = original(sid, body)
        if body["tool_call_id"] == "d_a" and not inner:
            inner.append(runner.agent_loop("delegate_task", args, sid="p", tcid="d_b"))
        return answer

    reserved: list[str] = []

    def reserve(sid: str, body: Any) -> Any:
        reserved.append(body["tool_call_id"])
        return {"decision": "allow", "delegation_id": f"del-{body['tool_call_id']}"}

    monkeypatch.setattr(adapter.client, "reserve_delegation", reserve)
    monkeypatch.setattr(adapter.client, "authorize", answer_then_interleave)
    runner.agent_loop("delegate_task", args, sid="p", tcid="d_a")
    monkeypatch.setattr(adapter.client, "authorize", original)
    assert sorted(reserved) == ["d_a", "d_b"]
    a2 = server.pending_by_call["d_a"]
    server.decide = lambda body: "allow"
    server.approve(a2)
    runner.agent_loop("delegate_task", args, sid="p", tcid="d_c")
    assert runner.executions == 1
    assert sorted(reserved) == ["d_a", "d_b"], "the retry reuses a reservation"
    assert server.authorize_calls()[-1].body["tool_call_id"] == "d_a"
    queued = adapter._delegations["p"]
    assert [r[0] for r in queued] == ["del-d_a"], "the reservation made under a2 follows a2"
    (slot,) = adapter._provisional_delegations
    assert slot[3] == server.pending_by_call["d_b"]
    (waiting,) = adapter._provisional_delegations[slot]
    assert waiting.reservation[0] == "del-d_b"


def _observe_decisions(server: FakeAgenomic) -> list[dict[str, Any]]:
    return [
        e
        for e in server.events
        if e.get("type") == "tool.call.decision" and e["extra"].get("local_mode") == "observe"
    ]


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
@pytest.mark.parametrize("check", ["protected_path", "hermes_incompatible", "mutator"])
def test_observe_records_local_checks_as_counterfactuals(
    server: FakeAgenomic, tmp_path: Path, order: str, check: str
) -> None:
    server.effective_state = "observe"
    identity = (
        {"version": "0.22.0", "release_date": None, "commit": None}
        if check == "hermes_incompatible"
        else None
    )
    adapter = make_adapter(server.url, tmp_path, identity=identity)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "observe"
    hellos = len(server.calls("/hello"))
    target = tmp_path / "out.txt"
    if check == "protected_path":
        target = tmp_path / "home" / "skills" / "evil" / "SKILL.md"
        target.parent.mkdir(parents=True)
    if check == "mutator":

        def other_plugin(**kwargs: Any) -> dict[str, Any]:
            return {"action": "modify", "args": {}}

        adapter.ctx._manager._hooks["pre_tool_call"].append(other_plugin)
    runner = Runner(adapter)
    getattr(runner, order)(
        "write_file", {"path": str(target), "content": "x"}, effect=write_effect(target)
    )
    assert runner.executions == 1, "observe never changes execution"
    assert target.read_text() == "x"
    adapter.exporter.flush(3.0)
    assert server.authorize_calls() == []
    assert server.calls("/delegations") == []
    assert len(server.calls("/hello")) == hellos, "observe asks the gateway nothing"
    code = "foreign_mutators_unconfirmed" if check == "mutator" else check
    recorded = _observe_decisions(server)
    assert len(recorded) == 1, "recorded once, whichever gate sees the call first"
    event = recorded[0]
    assert event["decision"] == "deny"
    assert event["extra"]["local"] is True
    assert event["extra"]["reason_codes"] == [code]
    assert event["extra"]["counterfactual"] == {"outcome": "deny", "reason_codes": [code]}


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_observe_records_an_outstanding_enforce_approval_without_touching_it(
    server: FakeAgenomic, tmp_path: Path, order: str
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    target = tmp_path / "out.txt"
    args = {"path": str(target), "content": "x"}
    first = json.loads(
        getattr(runner, order)("write_file", args, tcid="call_1", effect=write_effect(target))
    )
    approval_id = next(iter(server.approvals))
    assert first["error"] == APPROVAL_MESSAGE.format(approval_id=approval_id)
    assert runner.executions == 0
    (key,) = adapter._pending
    (entry,) = adapter._pending[key]
    before = (entry.logical_call_id, entry.attempt, entry.approval_id, entry.claimed_by)
    authorizations = len(server.authorize_calls())
    approval_reads = len(server.calls(f"/approvals/{approval_id}", "GET"))

    adapter._effective_state = "observe"
    getattr(runner, order)("write_file", args, tcid="call_2", effect=write_effect(target))
    assert runner.executions == 1, "observe never changes execution"
    assert target.read_text() == "x"
    assert len(server.authorize_calls()) == authorizations
    assert len(server.calls(f"/approvals/{approval_id}", "GET")) == approval_reads
    assert server.calls("/delegations") == []
    assert adapter._pending[key] == [entry], "the approval stays for a later enforce retry"
    assert (entry.logical_call_id, entry.attempt, entry.approval_id, entry.claimed_by) == before
    assert server.approvals[approval_id]["status"] == "pending"
    adapter.exporter.flush(3.0)
    recorded = _observe_decisions(server)
    assert len(recorded) == 1, "recorded once, whichever gate sees the call first"
    event = recorded[0]
    assert event["decision"] == "require_approval"
    assert event["span_id"] == "call_2"
    assert event["extra"]["approval_id"] == approval_id
    assert event["extra"]["reason_codes"] == ["approval_pending"]
    assert event["extra"]["counterfactual"] == {
        "outcome": "require_approval",
        "reason_codes": ["approval_pending"],
    }


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_observe_without_a_local_finding_records_no_decision(
    server: FakeAgenomic, tmp_path: Path, order: str
) -> None:
    server.effective_state = "observe"
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    runner = Runner(adapter)
    getattr(runner, order)("read_file", {"path": "/tmp/x"})
    assert runner.executions == 1
    adapter.exporter.flush(3.0)
    assert _observe_decisions(server) == []


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_hello_switching_to_observe_fails_open_on_authorization_errors(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "enforce"

    def other_plugin(**kwargs: Any) -> dict[str, Any]:
        return {"action": "modify", "args": {}}

    adapter.ctx._manager._hooks["pre_tool_call"].append(other_plugin)
    # The hello confirming the new mutator answers observe; authorize is unreachable.
    server.effective_state = "observe"
    authorize_attempts: list[str] = []

    def down(sid: str, body: Any) -> Any:
        authorize_attempts.append(sid)
        raise HermesApiError("unreachable", "connection refused", 0)

    monkeypatch.setattr(adapter.client, "authorize", down)
    runner = Runner(adapter)
    result = getattr(runner, order)("read_file", {"path": "/tmp/x"})
    assert json.loads(result) == {"success": True}
    assert runner.executions == 1, "observe never changes execution"
    assert adapter.local_mode() == "observe"
    assert len(server.calls("/hello")[-1].body["foreign_mutators"]) == 1
    assert authorize_attempts == [], "observe never asks for an authorization"
    adapter.exporter.flush(3.0)
    assert "tool.call.requested" in server.event_types()


@pytest.mark.parametrize("state", ["observe", "shadow"])
def test_authorization_outage_fails_open_outside_enforce(
    server: FakeAgenomic, tmp_path: Path, state: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._effective_state = state
    exc = HermesApiError("unreachable", "connection refused", 0)
    assert adapter._unavailable("s1", "read_file", "call_1", exc) is None
    adapter._effective_state = "enforce"
    assert adapter._unavailable("s1", "read_file", "call_1", exc)
    adapter.exporter.flush(3.0)
    assert "authorization.unavailable" in server.event_types()


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
@pytest.mark.parametrize("state", ["observe", "shadow", "enforce"])
@pytest.mark.parametrize("kind", ["pause", "quarantine", "revoke"])
def test_stop_command_between_the_gates_blocks_a_cached_authorization(
    server: FakeAgenomic,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    state: str,
    kind: str,
) -> None:
    server.effective_state = state
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    # The command lands after the first gate answered, before the second gate runs.
    second_gate = "pre_tool_call" if order == "agent_loop" else "tool_execution"
    real = getattr(adapter, second_gate)

    def stop_then_gate(**kwargs: Any) -> Any:
        adapter.handle_command({"id": "stop-1", "kind": kind, "target_kind": "instance"})
        return real(**kwargs)

    monkeypatch.setattr(adapter, second_gate, stop_then_gate)
    runner = Runner(adapter)
    target = tmp_path / "raced.txt"
    out = getattr(runner, order)(
        "write_file", {"path": str(target), "content": "x"}, effect=write_effect(target)
    )
    status = {"pause": "paused", "quarantine": "quarantined", "revoke": "revoked"}[kind]
    message = f"Agenomic: this instance is {status}; the action was not executed."
    assert runner.executions == 0, f"a local {kind} stops the call in {state}"
    assert not target.exists()
    assert message in out
    adapter.exporter.flush(3.0)
    recorded = _local_decisions(server, message)
    assert len(recorded) == 1
    assert recorded[0]["extra"]["reason_codes"] == ["instance_stopped"]


@pytest.mark.parametrize("order", ["agent_loop", "direct"])
def test_each_child_takes_the_reservation_of_the_invocation_that_built_it(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")

    def reserve(sid: str, body: Any) -> Any:
        return {"decision": "allow", "delegation_id": f"del-{body['tool_call_id']}"}

    monkeypatch.setattr(adapter.client, "reserve_delegation", reserve)
    args = {"tasks": [{"goal": "x"}]}
    kw = Runner(adapter)._kw

    def start_child(child: str) -> Callable[..., str]:
        def build(*_: Any) -> str:
            # Hermes builds the children on the thread running delegate_task.
            adapter.subagent_start(parent_session_id="p", child_session_id=child)
            adapter.on_session_start(session_id=child, platform="subagent", model="m")
            return json.dumps({"success": True})

        return build

    if order == "direct":
        # Both calls are authorized (A first), then B builds its child before A.
        assert adapter.pre_tool_call(args=args, **kw("delegate_task", "p", "d_a")) is None
        assert adapter.pre_tool_call(args=args, **kw("delegate_task", "p", "d_b")) is None
        adapter.tool_execution(
            args=args, next_call=start_child("c_b"), **kw("delegate_task", "p", "d_b")
        )
        adapter.tool_execution(
            args=args, next_call=start_child("c_a"), **kw("delegate_task", "p", "d_a")
        )
    else:
        # A is authorized first; while A runs, B is authorized and builds its child, then A.
        def run_a(*_: Any) -> str:
            assert adapter.pre_tool_call(args=args, **kw("delegate_task", "p", "d_a")) is None
            adapter.tool_execution(
                args=args,
                next_call=lambda *_: (
                    adapter.pre_tool_call(args=args, **kw("delegate_task", "p", "d_b")),
                    start_child("c_b")(),
                )[1],
                **kw("delegate_task", "p", "d_b"),
            )
            return start_child("c_a")()

        adapter.tool_execution(args=args, next_call=run_a, **kw("delegate_task", "p", "d_a"))
    admitted = {
        r.body["hermes_session_id"]: r.body.get("delegation_id")
        for r in server.calls("/v1/hermes/runtime/sessions")
    }
    assert admitted["c_a"] == "del-d_a"
    assert admitted["c_b"] == "del-d_b"
    assert not adapter._delegations.get("p"), "both reservations are used up"


def test_failed_identical_delegations_each_keep_their_reservation(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    runner = Runner(adapter)
    args = {"tasks": [{"goal": "x"}]}

    def reserve(sid: str, body: Any) -> Any:
        return {"decision": "allow", "delegation_id": f"del-{body['tool_call_id']}"}

    monkeypatch.setattr(adapter.client, "reserve_delegation", reserve)
    original = adapter.client.authorize

    def unreachable_then_interleave(sid: str, body: Any) -> Any:
        # d2 reserves while d1 is still deciding; both authorizations then fail.
        if body["tool_call_id"] == "d1":
            runner.direct("delegate_task", args, sid="p", tcid="d2")
        raise HermesApiError("unreachable", "connection refused", 0)

    monkeypatch.setattr(adapter.client, "authorize", unreachable_then_interleave)
    runner.direct("delegate_task", args, sid="p", tcid="d1")
    assert runner.executions == 0
    (waiting,) = adapter._provisional_delegations.values()
    assert [p.reservation[0] for p in waiting] == ["del-d1", "del-d2"], "neither is dropped"
    monkeypatch.setattr(adapter.client, "authorize", original)
    # A retry under d2 takes back d2's reservation; another retry takes the oldest one.
    runner.direct("delegate_task", args, sid="p", tcid="d2")
    runner.direct("delegate_task", args, sid="p", tcid="d3")
    assert runner.executions == 2
    assert [r[0] for r in adapter._delegations["p"]] == ["del-d2", "del-d1"]
    assert not adapter._provisional_delegations


def test_call_blocked_before_the_middleware_retires_its_authorization(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")

    def reserve(sid: str, body: Any) -> Any:
        return {"decision": "allow", "delegation_id": f"del-{body['tool_call_id']}"}

    monkeypatch.setattr(adapter.client, "reserve_delegation", reserve)
    args = {"tasks": [{"goal": "x"}]}
    kw = Runner(adapter)._kw("delegate_task", "p", "d1")
    # Direct dispatch: this adapter authorizes, then a later pre_tool_call callback blocks
    # and Hermes reports the call blocked; tool_execution is never entered.
    assert adapter.pre_tool_call(args=args, **kw) is None
    assert [r[0] for r in adapter._delegations["p"]] == ["del-d1"]
    adapter.post_tool_call(args=args, result="{}", status="blocked", **kw)
    assert adapter._auth[("p", "delegate_task", "d1")].state == "done"
    assert not adapter._delegations.get("p"), "the blocked call's reservation is dropped"
    # A child started later without a reservation of its own never takes d1's delegation.
    adapter.subagent_start(parent_session_id="p", child_session_id="c1")
    adapter.on_session_start(session_id="c1", platform="subagent", model="m")
    admitted = {
        r.body["hermes_session_id"]: r.body.get("delegation_id")
        for r in server.calls("/v1/hermes/runtime/sessions")
    }
    assert admitted["c1"] is None
    adapter.exporter.flush(3.0)
    assert [e for e in server.events if e.get("type") == "tool.call.not_executed"]


@pytest.mark.parametrize("same_args", [True, False], ids=["same_args", "other_args"])
def test_observe_between_the_gates_never_leaves_the_authorization_reusable(
    server: FakeAgenomic, tmp_path: Path, same_args: bool
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "enforce"
    runner = Runner(adapter)
    kw = runner._kw("read_file", "s1", "call_1")
    args = {"path": "/tmp/a"}
    assert adapter.pre_tool_call(args=args, **kw) is None
    first = len(server.authorize_calls())
    # The server switches to observe before the execution gate runs.
    adapter._set_state("observe")
    exec_args = args if same_args else {"path": "/tmp/b"}
    adapter.tool_execution(args=exec_args, next_call=lambda *_: "{}", **kw)
    assert adapter._auth[("s1", "read_file", "call_1")].state == "done"
    # Back in enforce, a call reusing the id asks the gateway again.
    adapter._set_state("enforce")
    runner.direct("read_file", args, tcid="call_1")
    assert len(server.authorize_calls()) == first + 1, "no reuse of the old permit"


@pytest.mark.parametrize("first", ["subagent", "session"])
def test_session_and_subagent_cancels_of_one_child_are_both_applied(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: True)
    adapter.subagent_start(parent_session_id="p", child_session_id="c", child_subagent_id="sa-1")
    adapter.on_session_start(session_id="c", platform="subagent", model="m")
    commands = {
        "subagent": {
            "id": "k_sub",
            "kind": "cancel",
            "target_kind": "subagent",
            "target_ref": "sa-1",
        },
        "session": {"id": "k_ses", "kind": "cancel", "target_kind": "session", "target_ref": "c"},
    }
    for kind in (first, "session" if first == "subagent" else "subagent"):
        adapter.handle_command({**commands[kind], "status": "requested"})
    adapter.subagent_stop(parent_session_id="p", child_session_id="c", child_status="interrupted")
    applied = sorted(c for c, b in server.acks if b["status"] == "applied")
    assert applied == ["k_ses", "k_sub"], "every cancel waiting for this end is acknowledged"


@pytest.mark.parametrize("target_kind", ["session", "subagent"])
def test_every_cancel_of_one_target_is_applied(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_kind: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    monkeypatch.setattr(adapter, "_interrupt_subagent", lambda sid: True)
    adapter.subagent_start(parent_session_id="p", child_session_id="c", child_subagent_id="sa-1")
    adapter.on_session_start(session_id="c", platform="subagent", model="m")
    target = "c" if target_kind == "session" else "sa-1"
    for command_id in ("k1", "k2"):
        adapter.handle_command(
            {
                "id": command_id,
                "kind": "cancel",
                "target_kind": target_kind,
                "target_ref": target,
                "status": "requested",
            }
        )
    adapter.subagent_stop(parent_session_id="p", child_session_id="c", child_status="interrupted")
    applied = [c for c, b in server.acks if b["status"] == "applied"]
    assert applied == ["k1", "k2"], "each waiting cancel is acknowledged once"


def test_shadow_decision_is_asked_again_when_enforce_starts_between_the_gates(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.effective_state = "shadow"
    server.decide = lambda body: "deny"
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "shadow"
    runner = Runner(adapter)
    kw = runner._kw("write_file", "s1", "call_1")
    args = {"path": "/tmp/x", "content": "y"}
    # Direct dispatch: shadow records the deny and lets the call through...
    assert adapter.pre_tool_call(args=args, **kw) is None
    asked = len(server.authorize_calls())
    # ...then enforce becomes active before the execution gate.
    server.effective_state = "enforce"
    adapter._set_state("enforce")
    out = adapter.tool_execution(args=args, next_call=lambda *_: runner._execute(args, None), **kw)
    assert len(server.authorize_calls()) == asked + 1, "decided again under enforce"
    assert runner.executions == 0, "the enforce deny blocks"
    assert "error" in json.loads(str(out))


def test_enforce_starting_inside_a_shadow_execution_blocks_at_the_second_gate(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.effective_state = "shadow"
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    real = adapter.pre_tool_call

    def enforce_then_gate(**kwargs: Any) -> Any:
        server.effective_state = "enforce"
        adapter._set_state("enforce")
        return real(**kwargs)

    monkeypatch.setattr(adapter, "pre_tool_call", enforce_then_gate)
    runner = Runner(adapter)
    out = json.loads(runner.agent_loop("write_file", {"path": "/tmp/x", "content": "y"}))
    assert runner.executions == 0
    assert "enforce became active" in out["error"]


def test_argument_change_never_blocks_once_shadow_is_active(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "enforce"
    real = adapter.pre_tool_call

    def shadow_then_gate(**kwargs: Any) -> Any:
        server.effective_state = "shadow"
        adapter._set_state("shadow")
        # Another plugin changed the arguments before this gate sees them.
        return real(**{**kwargs, "args": {"path": "/tmp/b"}})

    monkeypatch.setattr(adapter, "pre_tool_call", shadow_then_gate)
    runner = Runner(adapter)
    # Agent loop: the middleware authorized under enforce; shadow is active when the
    # second gate sees the changed arguments.
    runner.agent_loop("read_file", {"path": "/tmp/a"})
    assert runner.executions == 1, "shadow never changes execution"


@pytest.mark.parametrize("newer", ["enforce_blocked", "paused", "quarantined", "revoked"])
def test_stale_observe_answer_never_downgrades_a_newer_blocking_state(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newer: str
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "enforce"
    original = adapter.client.authorize

    def observe_answer_delayed(sid: str, body: Any) -> Any:
        server.effective_state = "observe"
        answer = original(sid, body)
        # The heartbeat thread receives a newer blocking state before this answer lands.
        adapter._set_state(newer)
        return answer

    monkeypatch.setattr(adapter.client, "authorize", observe_answer_delayed)
    runner = Runner(adapter)
    out = json.loads(runner.direct("write_file", {"path": "/tmp/x", "content": "y"}))
    assert adapter._effective_state == newer, "the newer state is kept"
    assert runner.executions == 0
    assert newer in out["error"]


def test_shadow_records_an_outstanding_enforce_approval_without_touching_it(
    server: FakeAgenomic, tmp_path: Path
) -> None:
    server.decide = lambda body: "require_approval"
    adapter = make_adapter(server.url, tmp_path)
    runner = Runner(adapter)
    args = {"path": str(tmp_path / "out.txt"), "content": "x"}
    runner.direct("write_file", args, tcid="call_1")
    approval_id = next(iter(server.approvals))
    (key,) = adapter._pending
    (entry,) = adapter._pending[key]
    # The server switches to shadow: the retry runs and the approval is recorded.
    server.effective_state = "shadow"
    server.decide = lambda body: "allow"
    adapter._set_state("shadow")
    runner.direct("write_file", args, tcid="call_2")
    assert runner.executions == 1, "shadow never changes execution"
    assert adapter._pending[key] == [entry], "the approval stays for a later enforce retry"
    assert entry.claimed_by is None
    adapter.exporter.flush(3.0)
    recorded = [
        e
        for e in server.events
        if e.get("type") == "tool.call.decision"
        and e["extra"].get("local_mode") == "shadow"
        and e["extra"].get("reason_codes") == ["approval_pending"]
    ]
    assert len(recorded) == 1
    assert recorded[0]["decision"] == "require_approval"
    assert recorded[0]["extra"]["approval_id"] == approval_id
    assert recorded[0]["span_id"] == "call_2"


def test_dropped_acknowledgement_lets_the_command_be_delivered_again(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = make_adapter(server.url, tmp_path)
    real = adapter.client.ack_command

    def ack_endpoint_down(command_id: str, status: str, detail: Any) -> Any:
        raise HermesApiError("unreachable", "connection refused", 0)

    monkeypatch.setattr(adapter.client, "ack_command", ack_endpoint_down)
    limit = adapter._ack_retries.maxlen
    assert limit is not None
    # Each command queues two acknowledgements (received, applied): the oldest are dropped.
    for i in range(limit):
        adapter.handle_command({"id": f"c{i}", "kind": "pause", "target_kind": "instance"})
    assert "c0" not in adapter._commands_seen, "its acknowledgements were dropped"
    assert f"c{limit - 1}" in adapter._commands_seen
    monkeypatch.setattr(adapter.client, "ack_command", real)
    adapter.handle_command({"id": "c0", "kind": "pause", "target_kind": "instance"})
    assert ("c0", "applied") in [(c, b.get("status")) for c, b in server.acks]


@pytest.mark.parametrize("request_kind", ["hello", "create_session"])
def test_delayed_state_response_never_replaces_a_newer_heartbeat_state(
    server: FakeAgenomic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_kind: str
) -> None:
    server.effective_state = "observe"
    adapter = make_adapter(server.url, tmp_path)
    adapter._ensure_started("cli")
    assert adapter.local_mode() == "observe"
    original = getattr(adapter.client, request_kind)

    def overtaken_by_a_heartbeat(*args: Any) -> Any:
        answer = original(*args)  # carries effective_state "observe"
        # The heartbeat thread sends a later request and applies enforce first.
        adapter._set_state("enforce", adapter._state_request())
        return answer

    monkeypatch.setattr(adapter.client, request_kind, overtaken_by_a_heartbeat)
    if request_kind == "hello":
        assert adapter._hello()
    else:
        adapter.on_session_start(session_id="s-late", platform="cli")
    assert adapter._effective_state == "enforce", "the stale observe answer is ignored"
    assert adapter.local_mode() == "enforce"
