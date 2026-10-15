"""P1 mirror: second 409 form on invoke and local/authorize, delegations, family simulate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from agenomic import Client
from agenomic.protect import DelegationScope, ProtectRunConfig, SignedDelegation
from agenomic.tools import ToolCallUnresolved, ToolExecutionError

BASE = "https://api.test"
RUN = "11111111-2222-4333-8444-555555555555"
INVOKE = f"{BASE}/v1/tool-execution/runs/{RUN}/invoke"
AUTHORIZE = f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize"
DELEGATIONS = f"{BASE}/v1/protect/delegations"
SIMULATE = f"{BASE}/v1/protect/simulate"
LINK_ID = "dddddddd-0000-4000-8000-000000000001"


def _cloud() -> Client:
    return Client(api_key="key", base_url=BASE)


def _other_attempt(code: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": "another attempt", "request_id": "req_1"}}


@pytest.mark.parametrize("url", [INVOKE, AUTHORIZE])
@pytest.mark.parametrize(
    ("code", "reconcile"), [("attempt_in_flight", False), ("ambiguous_outcome", True)]
)
def test_new_attempt_409_without_reason_or_status_block(
    httpx_mock: Any, url: str, code: str, reconcile: bool
) -> None:
    httpx_mock.add_response(method="POST", url=url, status_code=409, json=_other_attempt(code))
    ran: list[int] = []
    local = {"crm.update": lambda **_: ran.append(1)} if url == AUTHORIZE else None
    router = _cloud().tools.router(RUN, local_functions=local)
    with pytest.raises(ToolCallUnresolved) as exc:
        router.call("crm.update", {"id": 1}, logical_call_id="c1", attempt=2)
    assert ran == []
    assert exc.value.code == code
    assert exc.value.reason == code
    assert exc.value.status == 409
    assert exc.value.other_attempt is True
    assert exc.value.invocation is None
    assert exc.value.record_id is None
    assert exc.value.requires_reconciliation is reconcile
    assert len(httpx_mock.get_requests()) == 1


def test_plain_conflict_without_reason_is_not_unresolved(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=INVOKE,
        status_code=409,
        json={"error": {"code": "conflict", "message": "m"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().tools.router(RUN).call("crm.update", {}, logical_call_id="c1")
    assert not isinstance(exc.value, ToolCallUnresolved)


def test_from_body_rejects_foreign_shapes() -> None:
    assert ToolCallUnresolved.from_body("t", {}) is None
    assert ToolCallUnresolved.from_body("t", {"error": "x"}) is None
    assert ToolCallUnresolved.from_body("t", {"error": {"code": "other"}}) is None
    assert ToolCallUnresolved.from_body("t", {"error": {"code": "conflict", "reason": "x"}}) is None


def _link(parent: str | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "agenomic.protect.delegation/v1",
        "id": LINK_ID,
        "org_id": "o",
        "issued_by": {"kind": "user", "id": "u"},
        "delegator": {"kind": "user", "id": "u"},
        "delegate": {"kind": "agent", "id": "agent://support"},
        "scope": {
            "tools": ["crm.*"],
            "families": ["financial"],
            "effect_ceiling": "reversible_write",
        },
        "budget": {"max_actions": 20},
        "issued_at": "2026-10-15T10:00:00Z",
        "expires_at": "2026-10-16T10:00:00Z",
        "issuer": {"key_id": "k", "algorithm": "ed25519"},
    }
    if parent:
        document["parent_id"] = parent
        document["parent_ref"] = "blake3:parent"
    return {
        "document": document,
        "signature": {"algorithm": "ed25519", "value": "sig", "public_key_pem": "pem"},
    }


def test_issue_delegation_sends_exactly_the_wire_fields(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=DELEGATIONS, status_code=201, json=_link())
    signed = _cloud().protect.delegations.issue(
        "agent://support",
        scope=DelegationScope(
            tools=["crm.*"], families=["financial"], effect_ceiling="reversible_write"
        ),
        budget={"max_actions": 20},
        expires_at=datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc),
    )
    assert isinstance(signed, SignedDelegation)
    assert signed.document.scope.families == ["financial"]
    assert signed.document.delegate.kind == "agent"
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {
        "delegate_agent_id": "agent://support",
        "scope": {
            "tools": ["crm.*"],
            "families": ["financial"],
            "effect_ceiling": "reversible_write",
        },
        "budget": {"max_actions": 20},
        "expires_at": "2026-10-16T10:00:00+00:00",
    }


def test_issue_child_delegation_names_its_parent(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=DELEGATIONS, status_code=201, json=_link("p1"))
    signed = _cloud().protect.delegations.issue(
        "agent://sub",
        parent_id="p1",
        scope={"tools": ["crm.get"], "effect_ceiling": "read"},
        budget={"max_actions": 1},
        expires_at="2026-10-16T00:00:00Z",
    )
    assert signed.document.parent_id == "p1"
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content)["parent_id"] == "p1"


def test_widening_is_refused_with_the_server_code(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=DELEGATIONS,
        status_code=409,
        json={"error": {"code": "conflict", "message": "delegation_widened: tools"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().protect.delegations.issue(
            "a",
            scope={"tools": ["*"], "effect_ceiling": "read"},
            budget={"max_actions": 1},
            expires_at="2026-10-16T00:00:00Z",
        )
    assert "delegation_widened" in str(exc.value)


def test_get_delegation(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{DELEGATIONS}/{LINK_ID}", json=_link())
    assert _cloud().protect.delegations.get(LINK_ID).document.id == LINK_ID


async def test_async_delegation_round_trip(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="POST", url=DELEGATIONS, status_code=201, json=_link())
    httpx_mock.add_response(method="GET", url=f"{DELEGATIONS}/{LINK_ID}", json=_link())
    delegations = _cloud().protect.delegations
    issued = await delegations.aissue(
        "agent://support",
        scope={"tools": ["crm.*"], "effect_ceiling": "read"},
        budget={"max_actions": 20},
        expires_at="2026-10-16T00:00:00Z",
    )
    assert (await delegations.aget(issued.document.id)).document.id == LINK_ID


def test_scope_rejects_unknown_family_and_ceiling() -> None:
    with pytest.raises(ValueError):
        DelegationScope(tools=["a"], families=["telepathy"], effect_ceiling="read")  # type: ignore[list-item]
    with pytest.raises(ValueError):
        DelegationScope(tools=["a"], effect_ceiling="write")  # type: ignore[arg-type]


def test_simulate_forwards_family_fields_untouched(httpx_mock: Any) -> None:
    decision = {
        "outcome": "require_approval",
        "reason_codes": ["family_default"],
    }
    httpx_mock.add_response(
        method="POST",
        url=SIMULATE,
        json={"decisions": [decision], "policy_snapshot_digest": "blake3:s", "warnings": []},
    )
    intent = {
        "tool_id": "payments.refund",
        "families": ["financial"],
        "family_tiers": {"financial": "high"},
        "delegation_chain": ["d1:blake3:x"],
        "arguments": {"amount_minor": 90000, "currency": "EUR"},
    }
    response = _cloud().protect.simulate([intent], policy_refs=["crm@1.0.0"])
    assert response["decisions"][0]["reason_codes"] == ["family_default"]
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content)["intents"] == [intent]


def test_protect_run_config_wire_form() -> None:
    config = ProtectRunConfig(
        agent_id="bot",
        environment="prod",
        mode="shadow",
        policy_refs=["pol@1"],
        policy_refs_mode="shadow",
        delegation_id=LINK_ID,
    )
    assert config.to_wire() == {
        "agent_id": "bot",
        "environment": "prod",
        "mode": "shadow",
        "policy_refs": ["pol@1"],
        "policy_refs_mode": "shadow",
        "delegation_id": LINK_ID,
    }
    assert ProtectRunConfig(agent_id="bot", environment="prod").to_wire() == {
        "agent_id": "bot",
        "environment": "prod",
        "mode": "enforce",
    }


@pytest.mark.parametrize(
    "fields",
    [{"delegation_id": "not-a-uuid"}, {"mode": "live"}, {"delegation": LINK_ID}],
)
def test_protect_run_config_rejects_what_the_gateway_rejects(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProtectRunConfig(agent_id="bot", environment="prod", **fields)


@pytest.mark.parametrize("nested", [False, True])
def test_create_run_sends_typed_protect_section(httpx_mock: Any, nested: bool) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs",
        json={"run": {"id": RUN, "status": "approved"}},
    )
    config: dict[str, Any] = {
        "schema_version": "agenomic.tool_execution/v1",
        "mode": "live",
        "protect": ProtectRunConfig(agent_id="bot", environment="prod", delegation_id=LINK_ID),
    }
    _cloud().tools.create_run(config={"tool_execution": config} if nested else config)
    sent = json.loads(httpx_mock.get_requests()[0].content)["config"]
    sent = sent["tool_execution"] if nested else sent
    assert sent["protect"] == {
        "agent_id": "bot",
        "environment": "prod",
        "mode": "enforce",
        "delegation_id": LINK_ID,
    }
    assert isinstance(config["protect"], ProtectRunConfig)
