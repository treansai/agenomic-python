"""Model call admission client: decisions, permit, resume, guard and the typed 409."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agenomic import Client
from agenomic.tools import (
    ModelCallRequest,
    ModelUsage,
    ToolApprovalPending,
    ToolCallDenied,
    ToolCallUnresolved,
    ToolExecutionError,
    usage_from_response,
)

BASE = "https://api.test"
RUN = "11111111-2222-4333-8444-555555555555"
APPROVAL = "aaaaaaaa-0000-4000-8000-000000000001"
AUTHORIZE = f"{BASE}/v1/tool-execution/runs/{RUN}/model-calls/authorize"
REPORT = f"{BASE}/v1/tool-execution/runs/{RUN}/model-calls/report"
APPROVAL_URL = f"{BASE}/v1/protect/approvals/{APPROVAL}"
TOOL = "model:openai/gpt-5"
PERMIT = {
    "document": {"schema_version": "agenomic.protect.permit/v1", "record_id": "rec_1"},
    "signature": {"algorithm": "ed25519", "value": "sig", "public_key_pem": "pem"},
}
STAMP = {
    "decision_id": "d1",
    "outcome": "allow",
    "effective_mode": "enforce",
    "reason_codes": [],
    "policy_snapshot_digest": "blake3:snap",
    "evaluated_at": "2026-10-15T10:00:00Z",
}


def _request(**extra: Any) -> ModelCallRequest:
    return ModelCallRequest(
        provider="openai",
        model="gpt-5",
        max_output_tokens=512,
        estimated_input_tokens=300,
        logical_call_id="call-1",
        **extra,
    )


def _allow() -> dict[str, Any]:
    return {
        "decision": "allow",
        "tool": TOOL,
        "record_id": "rec_1",
        "permit": PERMIT,
        "protect": STAMP,
        "phase": "dispatching",
    }


def _pending() -> dict[str, Any]:
    return {
        "decision": "pending",
        "tool": TOOL,
        "record_id": "rec_1",
        "approval_id": APPROVAL,
        "protect": {**STAMP, "outcome": "require_approval", "approval_id": APPROVAL},
        "phase": "approval_pending",
    }


def _denied() -> dict[str, Any]:
    return {
        "decision": "denied",
        "tool": TOOL,
        "record_id": "rec_1",
        "protect": {**STAMP, "outcome": "deny", "reason_codes": ["no_policy_bound"]},
        "phase": "denied",
    }


def _cloud() -> Client:
    return Client(api_key="key", base_url=BASE)


def _bodies(httpx_mock: Any, url: str) -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in httpx_mock.get_requests(url=url)]


def test_request_body_carries_only_declared_fields() -> None:
    minimal = _request().to_body()
    assert minimal == {
        "repetition": 1,
        "logical_call_id": "call-1",
        "attempt": 1,
        "provider": "openai",
        "model": "gpt-5",
        "max_output_tokens": 512,
        "estimated_input_tokens": 300,
    }
    full = _request(
        endpoint_host="api.openai.com",
        estimated_cost={"amount_minor": 12, "currency": "EUR"},
        data_classification=["internal"],
        tool_choice="tool",
        tool_choice_name="search",
        tools=["search"],
        stream=True,
        parent_call_id="parent",
        deadline_ms=5000,
    ).to_body()
    assert full["estimated_cost"] == {"amount_minor": 12, "currency": "EUR"}
    assert full["tool_choice"] == "tool"
    assert all(value is not None for value in full.values())
    assert "prompt" not in full


def test_request_refuses_unknown_fields() -> None:
    with pytest.raises(ValueError):
        ModelCallRequest(
            provider="p",
            model="m",
            max_output_tokens=1,
            estimated_input_tokens=1,
            prompt="secret",  # type: ignore[call-arg]
        )


def test_authorize_allow_returns_permit(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    decision = _cloud().tools.model_calls.authorize(RUN, _request())
    assert decision.allowed
    assert decision.permit is not None
    assert decision.permit.document["record_id"] == "rec_1"
    assert decision.phase == "dispatching"
    assert decision.protect is not None
    assert decision.protect.outcome == "allow"


def test_authorize_allow_without_permit_is_refused(httpx_mock: Any) -> None:
    body = {**_allow(), "permit": None}
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=body)
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.model_calls.authorize(RUN, _request())
    assert exc.value.code == "invalid_response"


def test_authorize_pending_and_denied_are_typed(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=403, json=_denied())
    model_calls = _cloud().tools.model_calls
    pending = model_calls.authorize(RUN, _request())
    assert pending.requires_approval
    assert pending.approval_id == APPROVAL
    assert pending.permit is None
    denied = model_calls.authorize(RUN, _request())
    assert denied.denied
    assert denied.protect is not None
    assert denied.protect.reason_codes == ["no_policy_bound"]


def test_authorize_is_cloud_only() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        Client().tools.model_calls.authorize(RUN, _request())
    assert exc.value.code == "cloud_required"


def test_admit_raises_typed_errors(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=403, json=_denied())
    model_calls = _cloud().tools.model_calls
    with pytest.raises(ToolApprovalPending) as pending:
        model_calls.admit(RUN, _request())
    assert pending.value.approval_id == APPROVAL
    assert pending.value.tool == TOOL
    with pytest.raises(ToolCallDenied) as denied:
        model_calls.admit(RUN, _request())
    assert denied.value.code == "policy_denied"
    assert denied.value.decision is not None
    assert denied.value.decision.reason_codes == ["no_policy_bound"]


def test_wait_resumes_with_the_identical_body(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "pending"})
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "approved"})
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    decision = _cloud().tools.model_calls.admit(
        RUN, _request(tools=["search"]), wait=True, poll_interval=0
    )
    assert decision.allowed
    first, second = _bodies(httpx_mock, AUTHORIZE)
    assert first == second


def test_wait_rejected_approval_is_denied(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "rejected"})
    with pytest.raises(ToolCallDenied) as exc:
        _cloud().tools.model_calls.admit(RUN, _request(), wait=True, poll_interval=0)
    assert exc.value.code == "rejected"
    assert len(httpx_mock.get_requests(url=AUTHORIZE)) == 1


def test_wait_times_out_while_pending(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "pending"})
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.model_calls.admit(RUN, _request(), wait=True, poll_interval=0, timeout=0)
    assert exc.value.code == "approval_timeout"


def test_consumed_approval_replay_conflict_is_not_swallowed(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "consumed"})
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=409,
        json={"error": {"code": "attempt_in_flight", "message": "m"}},
    )
    with pytest.raises(ToolCallUnresolved):
        _cloud().tools.model_calls.admit(RUN, _request(), wait=True, poll_interval=0)


@pytest.mark.parametrize(
    ("error", "reason", "other_attempt", "reconcile"),
    [
        ({"code": "attempt_in_flight", "message": "m"}, "attempt_in_flight", True, False),
        ({"code": "ambiguous_outcome", "message": "m"}, "ambiguous_outcome", True, True),
        ({"code": "conflict", "reason": "in_flight", "message": "m"}, "in_flight", False, False),
        (
            {"code": "conflict", "reason": "ambiguous_outcome", "message": "m"},
            "ambiguous_outcome",
            False,
            True,
        ),
    ],
)
def test_authorize_409_both_forms_raise_unresolved_once(
    httpx_mock: Any, error: dict[str, str], reason: str, other_attempt: bool, reconcile: bool
) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=409, json={"error": error})
    with pytest.raises(ToolCallUnresolved) as exc:
        _cloud().tools.model_calls.authorize(RUN, _request())
    assert exc.value.reason == reason
    assert exc.value.other_attempt is other_attempt
    assert exc.value.requires_reconciliation is reconcile
    assert exc.value.status == 409
    assert exc.value.tool == TOOL
    assert len(httpx_mock.get_requests()) == 1


def test_other_409_stays_a_plain_tool_execution_error(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=409,
        json={"error": {"code": "idempotency_key_reuse", "message": "m"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.model_calls.authorize(RUN, _request())
    assert not isinstance(exc.value, ToolCallUnresolved)
    assert exc.value.code == "idempotency_key_reuse"


def test_report_forwards_the_permit_and_usage(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    model_calls = _cloud().tools.model_calls
    request = _request()
    decision = model_calls.authorize(RUN, request)
    result = model_calls.report(
        RUN,
        request,
        decision,
        usage=ModelUsage(input_tokens=10, output_tokens=4),
        duration_ms=42,
    )
    assert result.record_id == "rec_1"
    (body,) = _bodies(httpx_mock, REPORT)
    assert body["permit"] == PERMIT
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert body["is_error"] is False
    assert body["duration_ms"] == 42
    assert set(body) <= {
        "repetition",
        "logical_call_id",
        "attempt",
        "provider",
        "model",
        "usage",
        "is_error",
        "duration_ms",
        "permit",
    }


def test_report_refusal_carries_the_server_code(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(
        method="POST",
        url=REPORT,
        status_code=403,
        json={"error": {"code": "permit_invalid", "message": "no valid permit"}},
    )
    model_calls = _cloud().tools.model_calls
    request = _request()
    decision = model_calls.authorize(RUN, request)
    with pytest.raises(ToolExecutionError) as exc:
        model_calls.report(RUN, request, decision)
    assert (exc.value.code, exc.value.status) == ("permit_invalid", 403)


def test_guard_reports_usage_after_the_provider_call(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    calls: list[str] = []
    with _cloud().tools.model_calls.guard(RUN, _request()) as call:
        calls.append("provider")
        call.set_usage_from({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert calls == ["provider"]
    assert call.report is not None
    assert call.report.record_id == "rec_1"
    (body,) = _bodies(httpx_mock, REPORT)
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert body["is_error"] is False


def test_guard_never_runs_the_body_when_not_allowed(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=403, json=_denied())
    ran: list[int] = []
    with pytest.raises(ToolCallDenied), _cloud().tools.model_calls.guard(RUN, _request()):
        ran.append(1)
    assert ran == []
    assert httpx_mock.get_requests(url=REPORT) == []


def test_guard_reports_an_error_and_reraises_the_original(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    with (
        pytest.raises(RuntimeError, match="provider down"),
        _cloud().tools.model_calls.guard(RUN, _request()),
    ):
        raise RuntimeError("provider down")
    (body,) = _bodies(httpx_mock, REPORT)
    assert body["is_error"] is True
    assert "usage" not in body


def test_guard_report_failure_does_not_hide_the_provider_error(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(
        method="POST",
        url=REPORT,
        status_code=409,
        json={"error": {"code": "conflict", "message": "already reported"}},
    )
    with (
        pytest.raises(RuntimeError, match="provider down"),
        _cloud().tools.model_calls.guard(RUN, _request()),
    ):
        raise RuntimeError("provider down")


def test_guard_report_failure_surfaces_after_a_successful_call(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(
        method="POST",
        url=REPORT,
        status_code=409,
        json={"error": {"code": "conflict", "message": "already reported"}},
    )
    with (
        pytest.raises(ToolExecutionError) as exc,
        _cloud().tools.model_calls.guard(RUN, _request()),
    ):
        pass
    assert exc.value.code == "conflict"


async def test_async_guard_end_to_end(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=AUTHORIZE, status_code=202, json=_pending())
    httpx_mock.add_response(method="GET", url=APPROVAL_URL, json={"status": "approved"})
    httpx_mock.add_response(method="POST", url=AUTHORIZE, json=_allow())
    httpx_mock.add_response(method="POST", url=REPORT, json={"record_id": "rec_1"})
    async with _cloud().tools.model_calls.aguard(
        RUN, _request(), wait=True, poll_interval=0
    ) as call:
        call.set_usage(input_tokens=1, output_tokens=2)
    (body,) = _bodies(httpx_mock, REPORT)
    assert body["usage"] == {"input_tokens": 1, "output_tokens": 2}


async def test_async_authorize_409_other_attempt(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=AUTHORIZE,
        status_code=409,
        json={"error": {"code": "ambiguous_outcome", "message": "m"}},
    )
    with pytest.raises(ToolCallUnresolved) as exc:
        await _cloud().tools.model_calls.aauthorize(RUN, _request())
    assert exc.value.requires_reconciliation


def test_usage_from_response_shapes() -> None:
    class Usage:
        input_tokens = 5
        output_tokens = 6

    class Response:
        usage = Usage()

    assert usage_from_response(Response()) is not None
    assert usage_from_response({"usage": {"input_tokens": 1}}) is None
    assert usage_from_response(object()) is None
