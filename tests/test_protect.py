"""Tests for Protect: pending and denied calls, resume, hooks, overlays and ``client.protect``."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from agenomic import Client
from agenomic.integrations.anthropic import inject_anthropic_overlay, instrument_anthropic
from agenomic.integrations.openai import inject_openai_overlay, instrument_openai
from agenomic.protect import ProtectOverlay, ProtectResource
from agenomic.tools import (
    ToolApprovalPending,
    ToolCallDenied,
    ToolCallResult,
    ToolExecutionError,
)

BASE = "https://api.test"
RUN = "11111111-2222-4333-8444-555555555555"
APPROVAL = "aaaaaaaa-0000-4000-8000-000000000001"
INVOKE = f"{BASE}/v1/tool-execution/runs/{RUN}/invoke"
AUTHORIZE = f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize"
REPORT = f"{BASE}/v1/tool-execution/runs/{RUN}/report-local"
APPROVAL_URL = f"{BASE}/v1/protect/approvals/{APPROVAL}"

PROTECT = {
    "decision_id": "d1",
    "outcome": "require_approval",
    "effective_mode": "enforce",
    "reason_codes": ["approval_required"],
    "approval_id": APPROVAL,
    "policy_snapshot_digest": "blake3:snap",
    "evaluated_at": "2026-09-14T10:00:00Z",
}
DENY = {**PROTECT, "outcome": "deny", "reason_codes": ["rule_matched"], "approval_id": None}
PERMIT = {
    "document": {"schema_version": "agenomic.protect.permit/v1", "record_id": "rec_1"},
    "signature": {"algorithm": "ed25519", "value": "sig", "public_key_pem": "pem"},
}


def _envelope(status: str, protect: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "result": None,
        "agenomic": {
            "record_id": "rec_1",
            "status": status,
            "provenance": {"source": "unrouted", "binding_mode": "live"},
            "external_state": "none",
            "effects": [],
            "duration_ms": 0,
            "expected_error": False,
            "protect": protect,
            "decision": protect["outcome"],
            **extra,
        },
    }


def _ok(result: Any) -> dict[str, Any]:
    return {
        "result": result,
        "agenomic": {
            "record_id": "rec_2",
            "status": "success",
            "provenance": {"source": "live", "binding_mode": "live"},
            "external_state": "confirmed",
            "effects": [],
            "duration_ms": 1,
            "expected_error": False,
        },
    }


def _pending_envelope() -> dict[str, Any]:
    return _envelope("pending", PROTECT, approval_id=APPROVAL)


def _cloud() -> Client:
    return Client(api_key="key", base_url=BASE)


def _paths(httpx_mock: Any) -> list[str]:
    return [r.url.path.rsplit("/", 1)[-1] for r in httpx_mock.get_requests()]


def test_from_response_accepts_a_null_result() -> None:
    envelope = ToolCallResult.from_response(_pending_envelope())
    assert envelope.result is None
    assert envelope.status == "pending"
    assert envelope.approval_id == APPROVAL
    assert envelope.protect is not None
    assert envelope.protect.outcome == "require_approval"


def test_pending_invoke_returns_pending_and_router_raises_without_second_request(
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    client = _cloud()
    out = client.tools.invoke(RUN, "payments.refund", {"amount_minor": 900})
    assert out.status == "pending"
    assert out.ok is False
    router = client.tools.router(RUN)
    with pytest.raises(ToolApprovalPending) as exc:
        router.call("payments.refund", {"amount_minor": 900})
    assert exc.value.code == "approval_pending"
    assert exc.value.status == 202
    assert exc.value.approval_id == APPROVAL
    assert exc.value.record_id == "rec_1"
    assert router.calls[0].status == "pending"
    assert len(httpx_mock.get_requests()) == 2


def test_denied_invoke_raises_tool_call_denied_with_the_proposal(httpx_mock: Any) -> None:
    proposal = {"kind": "redact_arguments", "fields": ["body"]}
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=403,
        json=_envelope(
            "denied", DENY, transformation=proposal, safe_explanation="secret in arguments"
        ),
    )
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolCallDenied) as exc:
        router.call("email.send", {"body": "token"})
    assert exc.value.code == "policy_denied"
    assert exc.value.status == 403
    assert exc.value.transformation == proposal
    assert exc.value.decision is not None
    assert exc.value.decision.reason_codes == ["rule_matched"]
    assert "secret in arguments" in str(exc.value)
    assert router.calls[0].status == "denied"
    assert router.summary()["by_source"] == {"unrouted": 1}


def test_plain_403_without_envelope_stays_a_generic_error(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=403,
        json={"error": {"code": "forbidden", "message": "no"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.invoke(RUN, "email.send")
    assert not isinstance(exc.value, ToolCallDenied)
    assert exc.value.code == "forbidden"


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ({"decision": "denied", "record_id": "rec_1", "protect": DENY}, ToolCallDenied),
        (
            {
                "decision": "pending",
                "record_id": "rec_1",
                "approval_id": APPROVAL,
                "protect": PROTECT,
            },
            ToolApprovalPending,
        ),
        ({"decision": "maybe", "record_id": "rec_1"}, ToolCallDenied),
        ({"record_id": "rec_1"}, ToolCallDenied),
    ],
)
def test_local_function_never_runs_unless_the_decision_is_local(
    httpx_mock: Any, decision: dict[str, Any], expected: type[ToolExecutionError]
) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=decision)
    effects: list[str] = []
    router = _cloud().tools.router(
        RUN, local_functions={"crm.update": lambda **_: effects.append("ran")}
    )
    with pytest.raises(expected):
        router.call("crm.update", {"id": "c1"})
    assert effects == []
    assert _paths(httpx_mock) == ["authorize"]
    assert len(router.calls) == 1
    assert router.calls[0].status == ("pending" if expected is ToolApprovalPending else "denied")


def test_denied_authorize_answers_403_with_the_decision(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=403,
        json={"decision": "denied", "record_id": "rec_1", "protect": DENY},
    )
    effects: list[str] = []
    router = _cloud().tools.router(
        RUN, local_functions={"crm.delete": lambda **_: effects.append("ran")}
    )
    with pytest.raises(ToolCallDenied) as exc:
        router.call("crm.delete", {"id": "c1"})
    assert exc.value.code == "policy_denied"
    assert exc.value.decision is not None
    assert exc.value.decision.reason_codes == ["rule_matched"]
    assert effects == []
    assert router.calls[0].status == "denied"
    assert router.calls[0].record_id == "rec_1"


def test_plain_403_on_authorize_stays_a_generic_error(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=403,
        json={"error": {"code": "capability_denied", "message": "plan lapsed"}},
    )
    router = _cloud().tools.router(RUN, local_functions={"crm.delete": lambda **_: None})
    with pytest.raises(ToolExecutionError) as exc:
        router.call("crm.delete")
    assert not isinstance(exc.value, ToolCallDenied)
    assert exc.value.code == "capability_denied"


async def test_async_denied_authorize_answers_403_with_the_decision(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=403,
        json={"decision": "denied", "record_id": "rec_1", "protect": DENY},
    )

    async def delete(**_: Any) -> None:
        raise AssertionError("must not run")

    router = _cloud().tools.arouter(RUN, local_functions={"crm.delete": delete})
    with pytest.raises(ToolCallDenied):
        await router.call("crm.delete")


def test_permit_is_forwarded_verbatim_to_report_local(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        json={"decision": "local", "record_id": "rec_1", "permit": PERMIT},
    )
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    router = _cloud().tools.router(RUN, local_functions={"math.add": lambda a, b: {"sum": a + b}})
    assert router.call("math.add", {"a": 1, "b": 2}) == {"sum": 3}
    reported = json.loads(httpx_mock.get_requests()[1].content)
    assert reported["permit"] == PERMIT
    assert reported["result"] == {"sum": 3}


def test_report_local_without_permit_sends_no_permit_key(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    _cloud().tools.report_local(RUN, "math.add", {}, {"sum": 0}, logical_call_id="c1")
    assert "permit" not in json.loads(httpx_mock.get_requests()[0].content)


def test_resume_polls_then_reissues_the_identical_call(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(
        method="GET", url=APPROVAL_URL, json={"id": APPROVAL, "status": "pending"}
    )
    httpx_mock.add_response(
        method="GET", url=APPROVAL_URL, json={"id": APPROVAL, "status": "approved"}
    )
    httpx_mock.add_response(method="POST", url=INVOKE, json=_ok({"refunded": True}))
    router = _cloud().tools.router(RUN, repetition=2)
    with pytest.raises(ToolApprovalPending) as exc:
        router.call("payments.refund", {"amount_minor": 900}, parent_call_id="parent", attempt=3)
    assert router.resume(exc.value, poll_interval=0) == {"refunded": True}
    requests = httpx_mock.get_requests()
    assert _paths(httpx_mock) == ["invoke", APPROVAL, APPROVAL, "invoke"]
    assert json.loads(requests[0].content) == json.loads(requests[3].content)
    assert json.loads(requests[3].content) == {
        "repetition": 2,
        "logical_call_id": "payments.refund#1",
        "attempt": 3,
        "tool": "payments.refund",
        "arguments": {"amount_minor": 900},
        "parent_call_id": "parent",
    }
    assert requests[0].headers["Idempotency-Key"] == requests[3].headers["Idempotency-Key"]
    assert [c.status for c in router.calls] == ["pending", "success"]
    with pytest.raises(ToolExecutionError) as unknown:
        router.resume(APPROVAL)
    assert unknown.value.code == "approval_unknown"


def test_resume_by_id_reissues_a_local_call_with_its_permit(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        json={
            "decision": "pending",
            "record_id": "rec_1",
            "approval_id": APPROVAL,
            "protect": PROTECT,
        },
    )
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "approved"})
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        json={"decision": "local", "record_id": "rec_1", "permit": PERMIT},
    )
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    effects: list[str] = []

    def update(**kwargs: Any) -> dict[str, Any]:
        effects.append("ran")
        return {"updated": kwargs}

    router = _cloud().tools.router(RUN, local_functions={"crm.update": update})
    with pytest.raises(ToolApprovalPending):
        router.call("crm.update", {"id": "c1"})
    assert effects == []
    assert router.resume(APPROVAL, poll_interval=0) == {"updated": {"id": "c1"}}
    assert effects == ["ran"]
    requests = httpx_mock.get_requests()
    assert json.loads(requests[0].content) == json.loads(requests[2].content)
    assert json.loads(requests[3].content)["permit"] == PERMIT


@pytest.mark.parametrize("status", ["rejected", "expired", "consumed_by_someone"])
def test_resume_on_a_refused_approval_raises_without_reissuing(
    httpx_mock: Any, status: str
) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": status})
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolApprovalPending) as pending:
        router.call("payments.refund", {"amount_minor": 900})
    with pytest.raises(ToolCallDenied) as exc:
        router.resume(pending.value, poll_interval=0)
    assert exc.value.code == status
    assert exc.value.tool == "payments.refund"
    assert _paths(httpx_mock) == ["invoke", APPROVAL]


def test_resume_on_a_consumed_approval_replays_the_identity_and_recovers_the_result(
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "consumed"})
    httpx_mock.add_response(method="POST", url=INVOKE, json=_ok({"refunded": True}))
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolApprovalPending) as pending:
        router.call("payments.refund", {"amount_minor": 900})
    assert router.resume(pending.value, poll_interval=0) == {"refunded": True}
    requests = httpx_mock.get_requests()
    assert _paths(httpx_mock) == ["invoke", APPROVAL, "invoke"]
    assert json.loads(requests[0].content) == json.loads(requests[2].content)
    assert requests[0].headers["Idempotency-Key"] == requests[2].headers["Idempotency-Key"]
    assert [c.status for c in router.calls] == ["pending", "success"]


def test_resume_on_a_consumed_approval_raises_conflict_when_the_replay_is_refused(
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "consumed"})
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=409,
        json={"error": {"code": "approval_invalid", "message": "approval is consumed"}},
    )
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolApprovalPending) as pending:
        router.call("payments.refund", {"amount_minor": 900})
    with pytest.raises(ToolExecutionError) as exc:
        router.resume(pending.value, poll_interval=0)
    assert not isinstance(exc.value, ToolCallDenied)
    assert (exc.value.code, exc.value.status) == ("conflict", 409)
    assert _paths(httpx_mock) == ["invoke", APPROVAL, "invoke"]


async def test_async_resume_on_a_consumed_approval_raises_conflict(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "consumed"})
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=409,
        json={"error": {"code": "approval_invalid", "message": "approval is consumed"}},
    )
    router = _cloud().tools.arouter(RUN)
    with pytest.raises(ToolApprovalPending) as pending:
        await router.call("payments.refund", {"amount_minor": 900})
    with pytest.raises(ToolExecutionError) as exc:
        await router.resume(pending.value, poll_interval=0)
    assert (exc.value.code, exc.value.status) == ("conflict", 409)


def test_resume_times_out_while_still_pending(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "pending"})
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolApprovalPending) as pending:
        router.call("payments.refund")
    with pytest.raises(ToolExecutionError) as exc:
        router.resume(pending.value, poll_interval=0, timeout=0)
    assert exc.value.code == "approval_timeout"
    assert _paths(httpx_mock) == ["invoke", APPROVAL]


async def test_async_router_holds_and_resumes(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=INVOKE, status_code=202, json=_pending_envelope())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "pending"})
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "approved"})
    httpx_mock.add_response(method="POST", url=INVOKE, json=_ok({"refunded": True}))
    router = _cloud().tools.arouter(RUN)
    with pytest.raises(ToolApprovalPending) as exc:
        await router.call("payments.refund", {"amount_minor": 900})
    assert router.calls[0].status == "pending"
    assert await router.resume(exc.value, poll_interval=0) == {"refunded": True}
    requests = httpx_mock.get_requests()
    assert json.loads(requests[0].content) == json.loads(requests[3].content)


async def test_async_router_denies_unknown_local_decision(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json={"decision": "later"})
    effects: list[str] = []

    async def update(**_: Any) -> None:
        effects.append("ran")

    router = _cloud().tools.arouter(RUN, local_functions={"crm.update": update})
    with pytest.raises(ToolCallDenied):
        await router.call("crm.update")
    assert effects == []


def test_before_action_receives_the_identity_and_a_raise_aborts_before_any_request(
    httpx_mock: Any,
) -> None:
    seen: list[dict[str, Any]] = []

    def hook(identity: dict[str, Any]) -> bool:
        seen.append(identity)
        if identity["tool"] == "email.send":
            raise PermissionError("no external send")
        return False

    effects: list[str] = []
    router = _cloud().tools.router(
        RUN,
        local_functions={"email.send": lambda **_: effects.append("sent")},
        before_action=hook,
    )
    with pytest.raises(PermissionError):
        router.call("email.send", {"to": "x@example.test"})
    assert effects == []
    assert httpx_mock.get_requests() == []
    assert seen == [
        {
            "repetition": 1,
            "logical_call_id": "email.send#1",
            "attempt": 1,
            "tool": "email.send",
            "arguments": {"to": "x@example.test"},
        }
    ]
    httpx_mock.add_response(method="POST", url=INVOKE, json=_ok({"ok": True}))
    assert router.call("crm.get", {"id": "c1"}) == {"ok": True}
    assert seen[1]["tool"] == "crm.get"


async def test_async_before_action_raise_aborts(httpx_mock: Any) -> None:
    def hook(_: dict[str, Any]) -> None:
        raise PermissionError("blocked")

    router = _cloud().tools.arouter(RUN, before_action=hook)
    with pytest.raises(PermissionError):
        await router.call("crm.get")
    assert httpx_mock.get_requests() == []


def test_local_engine_refuses_a_protect_configuration() -> None:
    tools = Client().tools
    config = {
        "schema_version": "agenomic.tool_execution/v1",
        "mode": "mock",
        "protect": {"agent_id": "agent://acme/support", "mode": "enforce"},
    }
    for call in (tools.validate, tools.preflight, tools.create_run):
        with pytest.raises(ToolExecutionError) as exc:
            call(config=config)
        assert exc.value.code == "protect_cloud_required"
        assert exc.value.status == 400


def _openai_client() -> Any:
    create = MagicMock(return_value=SimpleNamespace(id="resp_1", system_fingerprint=None))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_openai_overlay_is_prepended_once_and_hashed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    overlay = ProtectOverlay(version="1", digest="blake3:o", text="Only call listed tools.")
    client = _openai_client()
    create = client.chat.completions.create
    instrument_openai(client, overlay=overlay)
    messages = [{"role": "user", "content": "hi"}]
    client.chat.completions.create(model="m", messages=messages)
    sent = create.call_args.kwargs["messages"]
    assert sent == [{"role": "system", "content": "Only call listed tools."}, *messages]
    assert messages == [{"role": "user", "content": "hi"}]
    client.chat.completions.create(model="m", messages=sent)
    assert create.call_args.kwargs["messages"] == sent
    assert inject_openai_overlay({"messages": sent}, "Only call listed tools.")["messages"] == sent
    assert inject_openai_overlay({"messages": messages}, None)["messages"] == messages
    plain = _openai_client()
    plain_create = plain.chat.completions.create
    instrument_openai(plain)
    plain.chat.completions.create(model="m", messages=messages)
    assert plain_create.call_args.kwargs["messages"] == messages


def test_openai_overlay_changes_the_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenomic.trace.context import set_current_recorder
    from agenomic.trace.recorder import TraceRecorder

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    hashes: list[str] = []
    for overlay in (None, "Rule.", "Rule."):
        rec = TraceRecorder("agent://a/b", "r", "t")
        set_current_recorder(rec)
        try:
            client = instrument_openai(_openai_client(), overlay=overlay)
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        finally:
            set_current_recorder(None)
        hashes.append(rec.model_calls[0].prompt_hash or "")
    assert hashes[0] != hashes[1]
    assert hashes[1] == hashes[2]


def _anthropic_client() -> Any:
    create = MagicMock(return_value=SimpleNamespace(id="msg_1"))
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_anthropic_overlay_sets_or_prefixes_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace())
    client = _anthropic_client()
    create = client.messages.create
    instrument_anthropic(client, overlay="Rule.")
    client.messages.create(model="m", messages=[])
    assert create.call_args.kwargs["system"] == "Rule."
    client.messages.create(model="m", messages=[], system="You help.")
    assert create.call_args.kwargs["system"] == "Rule.\n\nYou help."
    client.messages.create(model="m", messages=[], system="Rule.\n\nYou help.")
    assert create.call_args.kwargs["system"] == "Rule.\n\nYou help."
    blocks = [{"type": "text", "text": "You help."}]
    once = inject_anthropic_overlay({"system": blocks}, "Rule.")["system"]
    assert once == [{"type": "text", "text": "Rule."}, *blocks]
    assert inject_anthropic_overlay({"system": once}, "Rule.")["system"] == once
    assert inject_anthropic_overlay({"system": "You help."}, None)["system"] == "You help."


def test_protect_namespace_keeps_the_rmp_stage_and_refuses_cloud_calls_offline() -> None:
    protect = Client().protect
    assert isinstance(protect, ProtectResource)
    assert protect.alerts("mon_missing") == []
    with pytest.raises(ToolExecutionError) as exc:
        protect.coverage()
    assert exc.value.code == "cloud_required"
    with pytest.raises(ToolExecutionError):
        protect.approvals.list()


INTENT = {"action_type": "tool.call", "tool_id": "crm.update_customer", "arguments": {}}
OVERLAY = {"version": "1", "digest": "blake3:o", "text": "Only call listed tools."}

RESOURCE_CASES: list[tuple[str, Callable[[ProtectResource], Any], str, str, Any, Any]] = [
    ("overlay", lambda p: p.overlay(RUN), "GET", f"/v1/protect/runs/{RUN}/overlay", None, OVERLAY),
    (
        "catalog",
        lambda p: p.catalog(RUN),
        "GET",
        f"/v1/protect/runs/{RUN}/catalog",
        None,
        {"tools": []},
    ),
    (
        "approvals.list",
        lambda p: p.approvals.list(status="pending", run_id=RUN),
        "GET",
        f"/v1/protect/approvals?status=pending&run_id={RUN}",
        None,
        {"approvals": [{"id": APPROVAL}]},
    ),
    (
        "approvals.get",
        lambda p: p.approvals.get(APPROVAL),
        "GET",
        f"/v1/protect/approvals/{APPROVAL}",
        None,
        {},
    ),
    (
        "approvals.decide",
        lambda p: p.approvals.decide(APPROVAL, "approve", comment="ok"),
        "POST",
        f"/v1/protect/approvals/{APPROVAL}/decide",
        {"decision": "approve", "comment": "ok"},
        {},
    ),
    (
        "decisions.list",
        lambda p: p.decisions.list(
            run_id=RUN, outcome="deny", since="2026-09-14T00:00:00Z", limit=5
        ),
        "GET",
        f"/v1/protect/decisions?run_id={RUN}&outcome=deny&since=2026-09-14T00%3A00%3A00Z&limit=5",
        None,
        {"decisions": [], "next_cursor": None},
    ),
    (
        "decisions.list cursor",
        lambda p: p.decisions.list(limit=1, cursor="c1"),
        "GET",
        "/v1/protect/decisions?limit=1&cursor=c1",
        None,
        {"decisions": [{"id": "d2"}], "next_cursor": "c2"},
    ),
    ("decisions.get", lambda p: p.decisions.get("d1"), "GET", "/v1/protect/decisions/d1", None, {}),
    (
        "policies.register",
        lambda p: p.policies.register({"policy_id": "crm", "version": "1.0.0"}),
        "POST",
        "/v1/policies",
        {"policy_id": "crm", "version": "1.0.0"},
        {"policy": {}},
    ),
    (
        "policies.register text",
        lambda p: p.policies.register("policy_id: crm\nversion: 1.0.0\n"),
        "POST",
        "/v1/policies",
        {"document_text": "policy_id: crm\nversion: 1.0.0\n"},
        {"policy": {}},
    ),
    (
        "policies.get",
        lambda p: p.policies.get("crm", "1.0.0"),
        "GET",
        "/v1/policies/crm@1.0.0",
        None,
        {},
    ),
    ("policies.list", lambda p: p.policies.list(), "GET", "/v1/policies", None, {"policies": []}),
    (
        "policies.release",
        lambda p: p.policies.release("crm", "1.0.0"),
        "POST",
        "/v1/policies/crm@1.0.0/release",
        {},
        {},
    ),
    (
        "policies.deprecate",
        lambda p: p.policies.deprecate("crm", "1.0.0"),
        "POST",
        "/v1/policies/crm@1.0.0/deprecate",
        {},
        {},
    ),
    (
        "policies.simulate",
        lambda p: p.policies.simulate("crm", "1.0.0", [INTENT]),
        "POST",
        "/v1/policies/crm@1.0.0/simulate",
        {"intents": [INTENT]},
        {"decisions": []},
    ),
    (
        "policies.diff",
        lambda p: p.policies.diff("crm", "2.0.0", against="1.0.0"),
        "GET",
        "/v1/policies/crm@2.0.0/diff?against=1.0.0",
        None,
        {"lines": []},
    ),
    (
        "bindings.list",
        lambda p: p.bindings.list(
            scope_kind="agent", scope_ref="agent://acme/support", status="active"
        ),
        "GET",
        "/v1/protect/bindings?scope_kind=agent&scope_ref=agent%3A%2F%2Facme%2Fsupport&status=active",
        None,
        {"bindings": []},
    ),
    (
        "bindings.create",
        lambda p: p.bindings.create("crm", "1.0.0", "org", mode="shadow"),
        "POST",
        "/v1/protect/bindings",
        {"policy_id": "crm", "version": "1.0.0", "scope_kind": "org", "mode": "shadow"},
        {},
    ),
    (
        "bindings.revoke",
        lambda p: p.bindings.revoke("b1", reason="superseded"),
        "POST",
        "/v1/protect/bindings/b1/revoke",
        {"reason": "superseded"},
        {},
    ),
    (
        "restrictions.list",
        lambda p: p.restrictions.list(status="active"),
        "GET",
        "/v1/protect/restrictions?status=active",
        None,
        {"restrictions": []},
    ),
    (
        "restrictions.create",
        lambda p: p.restrictions.create(
            scope_kind="tool", scope_ref="email.send", kind="block_tool", reason="incident"
        ),
        "POST",
        "/v1/protect/restrictions",
        {
            "scope_kind": "tool",
            "kind": "block_tool",
            "reason": "incident",
            "scope_ref": "email.send",
        },
        {},
    ),
    (
        "restrictions.lift",
        lambda p: p.restrictions.lift("r1"),
        "POST",
        "/v1/protect/restrictions/r1/lift",
        {},
        {},
    ),
    (
        "kill_switch",
        lambda p: p.kill_switch("agent", "agent://acme/support", reason="runaway"),
        "POST",
        "/v1/protect/kill-switch",
        {"scope_kind": "agent", "reason": "runaway", "scope_ref": "agent://acme/support"},
        {},
    ),
    (
        "simulate",
        lambda p: p.simulate([INTENT], policy_refs=["crm@1.0.0"], decisions_from_run=RUN),
        "POST",
        "/v1/protect/simulate",
        {"intents": [INTENT], "policy_refs": ["crm@1.0.0"], "decisions_from_run": RUN},
        {"decisions": []},
    ),
    ("coverage", lambda p: p.coverage(), "GET", "/v1/protect/coverage", None, {"families": []}),
    (
        "metrics_summary",
        lambda p: p.metrics_summary(),
        "GET",
        "/v1/protect/metrics/summary",
        None,
        {},
    ),
]


@pytest.mark.parametrize(
    ("call", "method", "path", "body", "response"),
    [case[1:] for case in RESOURCE_CASES],
    ids=[case[0] for case in RESOURCE_CASES],
)
def test_protect_resources_hit_the_documented_routes(
    httpx_mock: Any,
    call: Callable[[ProtectResource], Any],
    method: str,
    path: str,
    body: Any,
    response: Any,
) -> None:
    httpx_mock.add_response(method=method, url=f"{BASE}{path}", json=response)
    out = call(_cloud().protect)
    request = httpx_mock.get_requests()[0]
    assert request.method == method
    assert str(request.url) == f"{BASE}{path}"
    assert request.headers["Authorization"] == "Bearer key"
    if body is None:
        assert request.content == b""
    else:
        assert json.loads(request.content) == body
    if isinstance(out, ProtectOverlay):
        assert out.text == OVERLAY["text"]
    elif isinstance(out, list):
        assert out == next(v for v in response.values() if isinstance(v, list))
    else:
        assert out == response


async def test_async_protect_resources_share_the_routes(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/protect/runs/{RUN}/overlay", json=OVERLAY)
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/protect/approvals/{APPROVAL}/decide",
        json={"status": "rejected"},
    )
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/v1/policies", json={"policies": [{"policy_id": "crm"}]}
    )
    protect = _cloud().protect
    assert (await protect.aoverlay(RUN)).digest == "blake3:o"
    assert await protect.approvals.adecide(APPROVAL, "reject") == {"status": "rejected"}
    assert await protect.policies.alist() == [{"policy_id": "crm"}]
    assert json.loads(httpx_mock.get_requests()[1].content) == {"decision": "reject"}


@pytest.mark.parametrize(
    ("call", "path", "response"),
    [
        (lambda p: p.approvals.list(), "/v1/protect/approvals", {"items": []}),
        (lambda p: p.policies.list(), "/v1/policies", []),
        (lambda p: p.decisions.list(), "/v1/protect/decisions", {"rows": []}),
    ],
)
def test_list_responses_without_the_documented_array_are_invalid(
    httpx_mock: Any, call: Callable[[ProtectResource], Any], path: str, response: Any
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE}{path}", json=response)
    with pytest.raises(ToolExecutionError) as exc:
        call(_cloud().protect)
    assert exc.value.code == "invalid_response"


def test_protect_resource_errors_carry_the_server_code(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/protect/approvals/{APPROVAL}/decide",
        status_code=409,
        json={"error": {"code": "approval_expired", "message": "too late"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().protect.approvals.decide(APPROVAL, "approve")
    assert (exc.value.code, exc.value.status) == ("approval_expired", 409)
