"""The runtime plane: ``client.tools.execute`` and ``client.vault.runtime``."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from agenomic import Client
from agenomic.client.retry import RetryPolicy
from agenomic.tools import ToolExecutionError
from agenomic.vault import (
    Sensitive,
    VaultApprovalRequired,
    VaultAuthenticationError,
    VaultConflict,
    VaultExecutionFailed,
    VaultExecutionInProgress,
    VaultGrantUnusable,
    VaultNotConfigured,
    VaultNotEntitled,
    VaultNotFound,
    VaultOutcomeUnknown,
    VaultPermissionDenied,
    VaultPolicyDenied,
    VaultRateLimited,
    VaultRefused,
    VaultRevoked,
    VaultServerError,
    VaultTransportError,
    VaultValidationError,
)

BASE = "https://api.test"
EXEC = f"{BASE}/v1/vault/runtime/executions"
ACTION = "0a1b2c3d-0000-4000-8000-000000000001"
RUNTIME_TOKEN = "vrt_runtime_token_0001"
API_KEY = "agm_admin_key_0001"
FAST = RetryPolicy(max_retries=2, base_delay=0.0)


def _client(**overrides: Any) -> Client:
    options: dict[str, Any] = {
        "base_url": BASE,
        "runtime_token": RUNTIME_TOKEN,
        "vault_retry": FAST,
    }
    options.update(overrides)
    return Client(**options)


def _finished(state: str = "succeeded", **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"status": "finished", "action_id": ACTION, "state": state}
    body.update(fields)
    return body


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **extra}}


def _bodies(httpx_mock: HTTPXMock) -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in httpx_mock.get_requests()]


def test_execute_returns_the_business_result_and_the_receipt(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        json=_finished(result={"name": "Ada"}, receipt_id="rcpt-1", status_code=200),
    )
    out = _client().tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
    )
    assert out.result == {"name": "Ada"}
    assert out.receipt_id == "rcpt-1"
    assert (out.status, out.state, out.status_code, out.replayed) == (
        "succeeded",
        "succeeded",
        200,
        False,
    )
    assert out.action_id == ACTION
    (request,) = httpx_mock.get_requests()
    assert request.headers["Authorization"] == f"Bearer {RUNTIME_TOKEN}"
    assert json.loads(request.content) == {
        "tool": "crm.get_customer",
        "binding": "crm-read",
        "arguments": {"id": "c_1"},
        "action_id": ACTION,
    }


def test_an_action_id_is_generated_and_exposed(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(result=1))
    out = _client().tools.execute(tool="t", binding="b")
    sent = _bodies(httpx_mock)[0]
    assert uuid.UUID(sent["action_id"]).version == 4
    assert out.action_id == sent["action_id"]
    assert sent["arguments"] == {}


def test_a_given_action_id_is_normalised_and_an_invalid_one_never_reaches_the_network() -> None:
    client = _client()
    with pytest.raises(VaultValidationError) as excinfo:
        client.tools.execute(tool="t", binding="b", action_id="not-a-uuid")
    assert excinfo.value.code == "invalid_action_id"


def test_invalid_requests_are_refused_locally() -> None:
    client = _client()
    with pytest.raises(VaultValidationError):
        client.tools.execute(tool="", binding="b")
    with pytest.raises(VaultValidationError):
        client.tools.execute(tool="t", binding="b", deadline_ms=0)


def test_deadline_is_sent_when_given(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished())
    _client().tools.execute(tool="t", binding="b", action_id=ACTION, deadline_ms=5000)
    assert _bodies(httpx_mock)[0]["deadline_ms"] == 5000


def test_a_technical_retry_reuses_the_same_action_id_and_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, status_code=503)
    httpx_mock.add_response(method="POST", url=EXEC, status_code=502)
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(result={"ok": True}))
    out = _client().tools.execute(tool="t", binding="b", arguments={"k": 1})
    bodies = _bodies(httpx_mock)
    assert len(bodies) == 3
    assert bodies[0] == bodies[1] == bodies[2]
    assert out.action_id == bodies[0]["action_id"]


def test_a_network_failure_is_retried_with_the_same_action_id(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"), method="POST", url=EXEC)
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(result=2))
    out = _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert [b["action_id"] for b in _bodies(httpx_mock)] == [ACTION, ACTION]
    assert out.result == 2


def test_exhausted_gateway_errors_raise_a_server_error_carrying_the_action_id(
    httpx_mock: HTTPXMock,
) -> None:
    for _ in range(3):
        httpx_mock.add_response(method="POST", url=EXEC, status_code=503)
    with pytest.raises(VaultServerError) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.action_id == ACTION
    assert len(httpx_mock.get_requests()) == 3


def test_exhausted_network_failures_raise_a_transport_error_without_a_chained_request(
    httpx_mock: HTTPXMock,
) -> None:
    for _ in range(3):
        httpx_mock.add_exception(httpx.ReadTimeout("slow"), method="POST", url=EXEC)
    with pytest.raises(VaultTransportError) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    error = excinfo.value
    assert (error.code, error.status, error.action_id) == ("transport_error", 0, ACTION)
    assert "same action_id" in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_a_rate_limit_is_retried_after_the_server_hint(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("agenomic.vault.transport._sleep", slept.append)
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=429,
        headers={"Retry-After": "7"},
        json=_error("too_many_requests", "rate limited; retry after 7s"),
    )
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished())
    _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert slept == [7.0]


def test_an_exhausted_rate_limit_raises_with_the_retry_after(httpx_mock: HTTPXMock) -> None:
    for _ in range(3):
        httpx_mock.add_response(
            method="POST",
            url=EXEC,
            status_code=429,
            headers={"Retry-After": "0"},
            json=_error("too_many_requests", "rate limited; retry after 0s"),
        )
    with pytest.raises(VaultRateLimited) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.retry_after == 0.0
    assert excinfo.value.status == 429


def test_outcome_unknown_is_raised_with_the_action_id_and_never_retried(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        json=_finished(
            "outcome_unknown",
            status_code=504,
            error_class="destination_error",
            receipt_id="rcpt-9",
            limitations=["will not be repeated automatically"],
        ),
    )
    with pytest.raises(VaultOutcomeUnknown) as excinfo:
        _client().tools.execute(tool="payments.refund", binding="pay", action_id=ACTION)
    error = excinfo.value
    assert (error.action_id, error.status_code, error.receipt_id) == (ACTION, 504, "rcpt-9")
    assert error.error_class == "destination_error"
    assert error.limitations == ["will not be repeated automatically"]
    assert "resolve" in str(error)
    assert len(httpx_mock.get_requests()) == 1


def test_outcome_unknown_after_a_technical_retry_stops_the_retries(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, status_code=503)
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished("outcome_unknown"))
    with pytest.raises(VaultOutcomeUnknown):
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert len(httpx_mock.get_requests()) == 2


def test_the_outcome_unknown_error_code_is_not_retried_either(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=409,
        json=_error("vault_outcome_unknown", "action outcome is unknown"),
    )
    with pytest.raises(VaultOutcomeUnknown) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.action_id == ACTION
    assert len(httpx_mock.get_requests()) == 1


def test_approval_required_is_pending_with_the_approval_id_then_resumes_with_the_same_action_id(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=202,
        json={"status": "approval_required", "action_id": ACTION, "approval_id": "apr-1"},
    )
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(result={"refunded": True}))
    client = _client()
    with pytest.raises(VaultApprovalRequired) as excinfo:
        client.tools.execute(tool="t", binding="b", arguments={"amount": 5}, action_id=ACTION)
    pending = excinfo.value
    assert (pending.approval_id, pending.action_id, pending.status) == ("apr-1", ACTION, 202)
    out = client.tools.execute(
        tool="t", binding="b", arguments={"amount": 5}, action_id=pending.action_id
    )
    assert out.result == {"refunded": True}
    first, second = _bodies(httpx_mock)
    assert first == second


def test_policy_denied_carries_the_reason_codes(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=403,
        json={
            "status": "denied",
            "action_id": ACTION,
            "reason_codes": ["no_policy_bound"],
            "explanation": "no policy is bound",
        },
    )
    with pytest.raises(VaultPolicyDenied) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.reason_codes == ["no_policy_bound"]
    assert excinfo.value.explanation == "no policy is bound"
    assert excinfo.value.action_id == ACTION
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("code", "reason", "hint"),
    [
        ("capability_not_entitled", "not_entitled", True),
        ("capability_not_in_edition", "not_in_edition", True),
        ("capability_disabled", "disabled", False),
    ],
)
def test_a_locked_add_on_is_distinguishable_and_never_retried(
    httpx_mock: HTTPXMock, code: str, reason: str, hint: bool
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=403,
        json=_error(
            code,
            "agents_vault is not included in the current plan",
            capability="agents_vault",
            reason=reason,
            required_plan="enterprise",
            request_id="req-1",
        ),
    )
    with pytest.raises(VaultNotEntitled) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    error = excinfo.value
    assert error.locked is True
    assert (error.capability, error.reason, error.required_plan) == (
        "agents_vault",
        reason,
        "enterprise",
    )
    assert error.upgrade_hint is hint
    assert (error.code, error.status, error.request_id) == (code, 403, "req-1")
    assert isinstance(error, ToolExecutionError)
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("reason", "exhausted", "missing"),
    [("exhausted", True, False), ("not_found", False, True), ("expired", False, False)],
)
def test_a_grant_problem_is_typed(
    httpx_mock: HTTPXMock, reason: str, exhausted: bool, missing: bool
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=409,
        json=_error("vault_grant_unusable", f"grant is not usable: {reason}"),
    )
    with pytest.raises(VaultGrantUnusable) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert (excinfo.value.reason, excinfo.value.exhausted, excinfo.value.missing) == (
        reason,
        exhausted,
        missing,
    )


def test_a_revoked_or_suspended_binding_is_typed(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=409,
        json=_error("vault_revoked", "credential or authority is revoked"),
    )
    with pytest.raises(VaultRevoked):
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)


def test_a_refused_envelope_is_a_typed_refusal(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=409,
        json={
            "status": "refused",
            "action_id": ACTION,
            "code": "live_call_denied",
            "message": "no",
        },
    )
    with pytest.raises(VaultRefused) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert (excinfo.value.code, excinfo.value.action_id) == ("live_call_denied", ACTION)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, _error("validation_error", "bad"), VaultValidationError),
        (401, _error("unauthorized", "no"), VaultAuthenticationError),
        (403, _error("forbidden", "no"), VaultPermissionDenied),
        (403, _error("vault_permission_denied", "credential.use"), VaultPermissionDenied),
        (404, _error("not_found", "binding not found"), VaultNotFound),
        (409, _error("conflict", "already being executed"), VaultConflict),
    ],
)
def test_the_server_codes_map_to_typed_errors_and_are_not_retried(
    httpx_mock: HTTPXMock, status: int, body: dict[str, Any], expected: type[Exception]
) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, status_code=status, json=body)
    with pytest.raises(expected) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.status == status  # type: ignore[attr-defined]
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("state", "extra", "expected"),
    [
        (
            "failed",
            {"error_class": "destination_error", "status_code": 404, "result": {"e": 1}},
            VaultExecutionFailed,
        ),
        ("refused", {"error_class": "no_policy_bound,write_blocked"}, VaultPolicyDenied),
        ("sent", {}, VaultExecutionInProgress),
        ("reserved", {}, VaultExecutionInProgress),
    ],
)
def test_every_non_success_state_has_its_own_error(
    httpx_mock: HTTPXMock, state: str, extra: dict[str, Any], expected: type[Exception]
) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(state, **extra))
    with pytest.raises(expected) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    error = excinfo.value
    assert error.action_id == ACTION  # type: ignore[attr-defined]
    if isinstance(error, VaultExecutionFailed):
        assert (error.status_code, error.result) == (404, {"e": 1})
    if isinstance(error, VaultPolicyDenied):
        assert error.reason_codes == ["no_policy_bound", "write_blocked"]


def test_a_malformed_success_body_is_an_invalid_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, content=b"<html>not json</html>")
    with pytest.raises(ToolExecutionError) as excinfo:
        _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.code == "invalid_response"


def test_get_execution_reads_any_state_without_raising(httpx_mock: HTTPXMock) -> None:
    url = f"{BASE}/v1/vault/runtime/executions/{ACTION}"
    httpx_mock.add_response(
        method="GET", url=url, json=_finished("outcome_unknown", receipt_id="r")
    )
    status = _client().tools.get_execution(ACTION)
    assert (status.state, status.receipt_id, status.terminal) == ("outcome_unknown", "r", True)


def test_runtime_calls_carry_only_the_runtime_token_and_admin_calls_only_the_api_key(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished())
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/vault/status", json={"installed": True})
    client = _client(api_key=API_KEY)
    client.tools.execute(tool="t", binding="b", action_id=ACTION)
    client.vault.status()
    runtime, admin = httpx_mock.get_requests()
    assert runtime.headers["Authorization"] == f"Bearer {RUNTIME_TOKEN}"
    assert admin.headers["Authorization"] == f"Bearer {API_KEY}"
    assert API_KEY.encode() not in runtime.content + runtime.url.raw_path
    assert RUNTIME_TOKEN.encode() not in admin.content + admin.url.raw_path


def test_a_missing_credential_fails_before_any_request(httpx_mock: HTTPXMock) -> None:
    with pytest.raises(VaultNotConfigured) as excinfo:
        Client(base_url=BASE, api_key=API_KEY).tools.execute(tool="t", binding="b")
    assert excinfo.value.code == "runtime_token_required"
    with pytest.raises(VaultNotConfigured) as excinfo:
        Client(base_url=BASE, runtime_token=RUNTIME_TOKEN).vault.status()
    assert excinfo.value.code == "api_key_required"
    with pytest.raises(VaultNotConfigured) as excinfo:
        Client(runtime_token=RUNTIME_TOKEN).tools.execute(tool="t", binding="b")
    assert excinfo.value.code == "cloud_required"
    assert httpx_mock.get_requests() == []


def test_an_empty_runtime_token_counts_as_missing() -> None:
    with pytest.raises(VaultNotConfigured):
        Client(base_url=BASE, runtime_token="").tools.execute(tool="t", binding="b")


def test_a_sensitive_value_cannot_ride_in_the_arguments(httpx_mock: HTTPXMock) -> None:
    with pytest.raises(VaultValidationError) as excinfo:
        _client().tools.execute(tool="t", binding="b", arguments={"k": Sensitive("a-secret-value")})  # type: ignore[dict-item]
    assert excinfo.value.code == "sensitive_not_allowed"
    assert httpx_mock.get_requests() == []


def test_unserialisable_arguments_are_refused_without_echo(httpx_mock: HTTPXMock) -> None:
    with pytest.raises(VaultValidationError) as excinfo:
        _client().tools.execute(tool="t", binding="b", arguments={"k": {1, 2}})  # type: ignore[dict-item]
    assert excinfo.value.code == "invalid_arguments"
    assert httpx_mock.get_requests() == []


def test_the_runtime_grants_routes(httpx_mock: HTTPXMock) -> None:
    grant = {"id": "g-1", "binding_id": "b-1", "state": "requested", "max_uses": 3}
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/runtime/grants?binding_id=b-1&state=approved",
        json=[grant],
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/vault/runtime/grants", status_code=201, json=grant
    )
    delegated = {**grant, "id": "g-2", "parent_grant_id": "g-1", "depth": 1}
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/runtime/grants/g-1/delegations",
        status_code=201,
        json=delegated,
    )
    runtime = _client().vault.runtime
    assert [g.id for g in runtime.list_grants(binding_id="b-1", state="approved")] == ["g-1"]
    created = runtime.request_grant(
        binding_id="b-1", max_uses=3, ttl_seconds=600, reason="refund batch"
    )
    assert created.state == "requested"
    child = runtime.delegate_grant(
        "g-1", delegate_agent_id="agent://acme/other", max_uses=1, ttl_seconds=60, reason="slice"
    )
    assert (child.parent_grant_id, child.depth) == ("g-1", 1)
    bodies = [json.loads(r.content) for r in httpx_mock.get_requests() if r.method == "POST"]
    assert bodies[0] == {
        "binding_id": "b-1",
        "max_uses": 3,
        "ttl_seconds": 600,
        "reason": "refund batch",
    }
    assert bodies[1]["delegate_agent_id"] == "agent://acme/other"


async def test_async_execute_matches_the_sync_path(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, status_code=503)
    httpx_mock.add_response(
        method="POST", url=EXEC, json=_finished(result={"ok": 1}, receipt_id="r")
    )
    out = await _client().tools.aexecute(tool="t", binding="b", arguments={"a": 1})
    bodies = _bodies(httpx_mock)
    assert bodies[0] == bodies[1]
    assert (out.result, out.receipt_id, out.action_id) == ({"ok": 1}, "r", bodies[0]["action_id"])


async def test_async_outcome_unknown_is_not_retried(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished("outcome_unknown"))
    with pytest.raises(VaultOutcomeUnknown) as excinfo:
        await _client().tools.aexecute(tool="t", binding="b", action_id=ACTION)
    assert excinfo.value.action_id == ACTION
    assert len(httpx_mock.get_requests()) == 1


async def test_async_status_read_and_errors(httpx_mock: HTTPXMock) -> None:
    url = f"{BASE}/v1/vault/runtime/executions/{ACTION}"
    httpx_mock.add_response(method="GET", url=url, json=_finished("sent"))
    status = await _client().tools.aget_execution(ACTION)
    assert status.state == "sent"
    assert status.terminal is False
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=202,
        json={"status": "approval_required", "action_id": ACTION, "approval_id": "a"},
    )
    with pytest.raises(VaultApprovalRequired):
        await _client().tools.aexecute(tool="t", binding="b", action_id=ACTION)


async def test_async_network_failures_are_retried_with_the_same_action_id(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"), method="POST", url=EXEC)
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished(result=3))
    out = await _client().tools.aexecute(tool="t", binding="b", action_id=ACTION)
    assert out.result == 3
    assert [b["action_id"] for b in _bodies(httpx_mock)] == [ACTION, ACTION]


async def test_async_exhausted_network_failures_raise_a_transport_error(
    httpx_mock: HTTPXMock,
) -> None:
    for _ in range(3):
        httpx_mock.add_exception(httpx.ReadTimeout("slow"), method="POST", url=EXEC)
    with pytest.raises(VaultTransportError) as excinfo:
        await _client().tools.aexecute(tool="t", binding="b", action_id=ACTION)
    assert (excinfo.value.action_id, excinfo.value.__context__) == (ACTION, None)


def test_an_unparsable_retry_after_falls_back_to_the_policy_delay(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("agenomic.vault.transport._sleep", slept.append)
    httpx_mock.add_response(
        method="POST", url=EXEC, status_code=429, headers={"Retry-After": "soon"}
    )
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished())
    Client(
        base_url=BASE,
        runtime_token=RUNTIME_TOKEN,
        vault_retry=RetryPolicy(max_retries=1, base_delay=0.5),
    ).tools.execute(tool="t", binding="b", action_id=ACTION)
    assert slept == [0.5]


def test_the_retry_after_hint_is_capped(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("agenomic.vault.transport._sleep", slept.append)
    httpx_mock.add_response(
        method="POST", url=EXEC, status_code=503, headers={"Retry-After": "86400"}
    )
    httpx_mock.add_response(method="POST", url=EXEC, json=_finished())
    _client().tools.execute(tool="t", binding="b", action_id=ACTION)
    assert slept == [60.0]


def test_a_call_that_declares_no_retry_is_never_resent(httpx_mock: HTTPXMock) -> None:
    from agenomic.vault.transport import Call, send

    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/vault/x", status_code=503)
    with pytest.raises(VaultServerError):
        send(_client(api_key=API_KEY), Call("POST", "/v1/vault/x", "admin", {"k": 1}))
    assert len(httpx_mock.get_requests()) == 1


def test_a_secret_bearing_body_with_an_unserialisable_object_is_refused_without_echo(
    httpx_mock: HTTPXMock,
) -> None:
    from agenomic.vault.transport import Call, send

    with pytest.raises(VaultValidationError) as excinfo:
        send(
            _client(api_key=API_KEY),
            Call("POST", "/v1/vault/x", "admin", {"v": object()}, secret_bearing=True),
        )
    assert excinfo.value.code == "invalid_arguments"
    assert httpx_mock.get_requests() == []
