"""Tests for the P0 enforcement wire types: unresolved 409, protection, settings, exemptions."""

from __future__ import annotations

from typing import Any

import pytest

from agenomic import Client
from agenomic.protect import EffectiveProtection
from agenomic.tools import InvocationStatus, ToolCallUnresolved, ToolExecutionError

BASE = "https://api.test"
RUN = "11111111-2222-4333-8444-555555555555"
RECORD = "bbbbbbbb-0000-4000-8000-000000000002"
INVOKE = f"{BASE}/v1/tool-execution/runs/{RUN}/invoke"
AUTHORIZE = f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize"
INVOCATION = f"{BASE}/v1/tool-execution/runs/{RUN}/invocations/{RECORD}"
RUNS = f"{BASE}/v1/tool-execution/runs"


def _status(phase: str, **extra: Any) -> dict[str, Any]:
    return {
        "record_id": RECORD,
        "run_id": RUN,
        "phase": phase,
        "status": "error",
        "external_state": "indeterminate",
        "tool": "crm.update",
        "logical_call_id": "c1",
        "repetition": 1,
        "attempt": 1,
        "requires_reconciliation": phase == "outcome_unknown",
        "started_at": "2026-10-15T00:00:00Z",
        **extra,
    }


def _conflict(reason: str, phase: str) -> dict[str, Any]:
    return {
        "error": {"code": "conflict", "reason": reason, "message": "m", "request_id": "req_1"},
        "agenomic": _status(phase),
    }


def _cloud() -> Client:
    return Client(api_key="key", base_url=BASE)


@pytest.mark.parametrize(
    ("reason", "phase", "reconcile"),
    [("in_flight", "dispatching", False), ("ambiguous_outcome", "outcome_unknown", True)],
)
def test_invoke_409_raises_unresolved_and_is_not_retried(
    httpx_mock: Any, reason: str, phase: str, reconcile: bool
) -> None:
    httpx_mock.add_response(
        method="POST", url=INVOKE, status_code=409, json=_conflict(reason, phase)
    )
    router = _cloud().tools.router(RUN)
    with pytest.raises(ToolCallUnresolved) as exc:
        router.call("crm.update", {"id": 1}, logical_call_id="c1")
    assert exc.value.reason == reason
    assert exc.value.status == 409
    assert exc.value.code == "conflict"
    assert exc.value.record_id == RECORD
    assert exc.value.requires_reconciliation is reconcile
    assert exc.value.invocation is not None
    assert exc.value.invocation.phase == phase
    assert isinstance(exc.value, ToolExecutionError)
    assert len(httpx_mock.get_requests()) == 1


def test_local_authorize_409_raises_unresolved_before_any_local_function(
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        method="POST", url=AUTHORIZE, status_code=409, json=_conflict("in_flight", "authorized")
    )
    ran: list[int] = []
    router = _cloud().tools.router(RUN, local_functions={"crm.update": lambda **_: ran.append(1)})
    with pytest.raises(ToolCallUnresolved) as exc:
        router.call("crm.update", {"id": 1}, logical_call_id="c1")
    assert exc.value.reason == "in_flight"
    assert ran == []
    assert len(httpx_mock.get_requests()) == 1


async def test_async_invoke_409_raises_unresolved(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=409,
        json=_conflict("ambiguous_outcome", "outcome_unknown"),
    )
    with pytest.raises(ToolCallUnresolved) as exc:
        await _cloud().tools.arouter(RUN).call("crm.update", {"id": 1}, logical_call_id="c1")
    assert exc.value.requires_reconciliation is True


def test_already_recorded_409_stays_a_plain_conflict(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST", url=INVOKE, status_code=409, json=_conflict("already_recorded", "succeeded")
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.invoke(RUN, "crm.update", {"id": 1}, logical_call_id="c1")
    assert not isinstance(exc.value, ToolCallUnresolved)
    assert exc.value.code == "conflict"
    assert exc.value.status == 409


def test_conflict_without_reason_is_a_plain_error(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST", url=INVOKE, status_code=409, json={"error": {"code": "conflict"}}
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.invoke(RUN, "crm.update", {}, logical_call_id="c1")
    assert not isinstance(exc.value, ToolCallUnresolved)


def test_unresolved_with_a_malformed_status_still_carries_the_reason(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=409,
        json={"error": {"code": "conflict", "reason": "in_flight", "message": "m"}, "agenomic": {}},
    )
    with pytest.raises(ToolCallUnresolved) as exc:
        _cloud().tools.invoke(RUN, "crm.update", {}, logical_call_id="c1")
    assert exc.value.reason == "in_flight"
    assert exc.value.invocation is None
    assert exc.value.record_id is None


def test_get_and_resolve_invocation(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=INVOCATION, json=_status("outcome_unknown"))
    httpx_mock.add_response(
        method="POST",
        url=f"{INVOCATION}/resolve",
        json=_status("succeeded", status="success", external_state="confirmed"),
    )
    tools = _cloud().tools
    status = tools.get_invocation(RUN, RECORD)
    assert isinstance(status, InvocationStatus)
    assert status.requires_reconciliation is True
    resolved = tools.resolve_invocation(RUN, RECORD, resolution="succeeded", evidence="ticket 7")
    assert resolved.phase == "succeeded"
    assert resolved.requires_reconciliation is False
    body = httpx_mock.get_requests()[1].content
    assert b'"resolution":"succeeded"' in body.replace(b" ", b"")


async def test_async_get_and_resolve_invocation(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=INVOCATION, json=_status("dispatching"))
    httpx_mock.add_response(
        method="POST", url=f"{INVOCATION}/resolve", json=_status("failed", status="error")
    )
    tools = _cloud().tools
    assert (await tools.aget_invocation(RUN, RECORD)).phase == "dispatching"
    resolved = await tools.aresolve_invocation(
        RUN, RECORD, resolution="failed", evidence="provider log"
    )
    assert resolved.phase == "failed"


def test_invocation_reads_need_the_cloud() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        Client().tools.get_invocation(RUN, RECORD)
    assert exc.value.code == "cloud_required"


def test_invoke_envelope_carries_the_phase(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        json={
            "result": {"ok": True},
            "agenomic": {
                "record_id": RECORD,
                "status": "success",
                "provenance": {"source": "live", "binding_mode": "live"},
                "external_state": "confirmed",
                "phase": "succeeded",
            },
        },
    )
    assert _cloud().tools.invoke(RUN, "crm.update", {}, logical_call_id="c1").phase == "succeeded"


def test_run_responses_expose_the_effective_protection(httpx_mock: Any) -> None:
    protection = {
        "entitlement": "not_entitled",
        "org_mandatory": True,
        "run_mode": "live",
        "protect_block": False,
        "state": "none",
    }
    httpx_mock.add_response(
        method="GET", url=f"{RUNS}/{RUN}", json={"run": {"id": RUN}, "protection": protection}
    )
    run = _cloud().tools.get_run(RUN)
    parsed = EffectiveProtection.model_validate(run["protection"])
    assert parsed.state == "none"
    assert parsed.is_protected is False
    assert run["id"] == RUN


def test_effective_protection_enforced_and_denied_shapes() -> None:
    enforced = EffectiveProtection.model_validate(
        {
            "entitlement": "entitled",
            "org_mandatory": True,
            "requirement": "mandatory_org",
            "run_mode": "hybrid",
            "protect_block": True,
            "protect_mode": "enforce",
            "state": "enforced",
        }
    )
    assert enforced.is_protected is True
    blocked = EffectiveProtection.model_validate(
        {
            "entitlement": "unknown",
            "org_mandatory": True,
            "run_mode": "live",
            "protect_block": False,
            "denial": "protect_state_unavailable",
            "state": "unknown",
        }
    )
    assert blocked.denial == "protect_state_unavailable"
    assert blocked.is_protected is False


def test_settings_roundtrip(httpx_mock: Any) -> None:
    body = {
        "protect_mandatory": True,
        "mandatory_in_force": False,
        "entitlement": "not_entitled",
        "revision": 3,
    }
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/protect/settings", json=body)
    httpx_mock.add_response(
        method="PUT", url=f"{BASE}/v1/protect/settings", json={**body, "revision": 4}
    )
    protect = _cloud().protect
    settings = protect.settings()
    assert settings.mandatory_in_force is False
    updated = protect.set_mandatory(True, expected_revision=settings.revision)
    assert updated.revision == 4
    sent = httpx_mock.get_requests()[1].content.replace(b" ", b"")
    assert b'"expected_revision":3' in sent


async def test_async_settings(httpx_mock: Any) -> None:
    body = {
        "protect_mandatory": False,
        "mandatory_in_force": False,
        "entitlement": "entitled",
        "revision": 1,
    }
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/protect/settings", json=body)
    httpx_mock.add_response(method="PUT", url=f"{BASE}/v1/protect/settings", json=body)
    protect = _cloud().protect
    assert (await protect.asettings()).entitlement == "entitled"
    assert (await protect.aset_mandatory(False)).protect_mandatory is False
    assert b"expected_revision" not in httpx_mock.get_requests()[1].content


EXEMPTION = {
    "id": "cccccccc-0000-4000-8000-000000000003",
    "scope_kind": "tool",
    "scope_ref": "crm.update",
    "reason": "migration window",
    "status": "active",
    "created_at": "2026-10-15T00:00:00Z",
    "expires_at": "2026-10-16T00:00:00Z",
}


def test_exemptions_list_create_lift(httpx_mock: Any) -> None:
    url = f"{BASE}/v1/protect/exemptions"
    httpx_mock.add_response(
        method="GET", url=f"{url}?status=active", json={"exemptions": [EXEMPTION]}
    )
    httpx_mock.add_response(method="POST", url=url, status_code=201, json=EXEMPTION)
    httpx_mock.add_response(
        method="POST", url=f"{url}/{EXEMPTION['id']}/lift", json={**EXEMPTION, "status": "lifted"}
    )
    exemptions = _cloud().protect.exemptions
    assert [e.scope_ref for e in exemptions.list(status="active")] == ["crm.update"]
    created = exemptions.create(
        scope_kind="tool",
        scope_ref="crm.update",
        reason="migration window",
        expires_at="2026-10-16T00:00:00Z",
    )
    assert created.status == "active"
    assert exemptions.lift(created.id).status == "lifted"


async def test_async_exemptions(httpx_mock: Any) -> None:
    url = f"{BASE}/v1/protect/exemptions"
    httpx_mock.add_response(method="GET", url=url, json={"exemptions": [EXEMPTION]})
    httpx_mock.add_response(method="POST", url=url, status_code=201, json=EXEMPTION)
    httpx_mock.add_response(
        method="POST", url=f"{url}/{EXEMPTION['id']}/lift", json={**EXEMPTION, "status": "lifted"}
    )
    exemptions = _cloud().protect.exemptions
    assert len(await exemptions.alist()) == 1
    created = await exemptions.acreate(
        scope_kind="org", reason="drill", expires_at="2026-10-16T00:00:00Z"
    )
    assert (await exemptions.alift(created.id)).status == "lifted"
    assert b"scope_ref" not in httpx_mock.get_requests()[1].content


def test_exemptions_need_the_cloud() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        Client().protect.exemptions.list()
    assert exc.value.code == "cloud_required"
