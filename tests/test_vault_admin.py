"""The administrative plane ``client.vault`` and its coverage of the OpenAPI contract."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from agenomic import Client
from agenomic.client.retry import RetryPolicy
from agenomic.vault import (
    BearerAuth,
    BindingContent,
    Destination,
    PathParam,
    ProviderDescriptor,
    RequestTemplate,
    Sensitive,
    TokenAuth,
    VaultError,
    VaultPermissionDenied,
    VaultTransportError,
)

BASE = "https://api.test"
API_KEY = "agm_admin_key_0001"
ACTION = "0a1b2c3d-0000-4000-8000-000000000001"
FAST = RetryPolicy(max_retries=2, base_delay=0.0)

# Snapshot of docs/vault/openapi.yaml of agenomic-cloud: 45 operations on 39 paths.
OPENAPI_OPERATIONS = [
    ("GET", "/v1/vault/status"),
    ("GET", "/v1/vault/providers"),
    ("POST", "/v1/vault/providers"),
    ("GET", "/v1/vault/providers/{id}"),
    ("POST", "/v1/vault/providers/{id}/health"),
    ("POST", "/v1/vault/providers/{id}/state"),
    ("GET", "/v1/vault/secrets"),
    ("POST", "/v1/vault/secrets"),
    ("POST", "/v1/vault/secrets/references"),
    ("GET", "/v1/vault/secrets/{id}"),
    ("POST", "/v1/vault/secrets/{id}/versions"),
    ("POST", "/v1/vault/secrets/{id}/revoke"),
    ("GET", "/v1/vault/bindings"),
    ("POST", "/v1/vault/bindings"),
    ("GET", "/v1/vault/bindings/{id}"),
    ("POST", "/v1/vault/bindings/{id}/versions"),
    ("POST", "/v1/vault/bindings/{id}/versions/{version}/submit"),
    ("POST", "/v1/vault/bindings/{id}/versions/{version}/decide"),
    ("POST", "/v1/vault/bindings/{id}/versions/{version}/activate"),
    ("POST", "/v1/vault/bindings/{id}/revoke"),
    ("GET", "/v1/vault/runtime-identities"),
    ("POST", "/v1/vault/runtime-identities"),
    ("POST", "/v1/vault/runtime-identities/{id}/revoke"),
    ("GET", "/v1/vault/grants"),
    ("POST", "/v1/vault/grants"),
    ("POST", "/v1/vault/grants/{id}/decide"),
    ("POST", "/v1/vault/grants/{id}/revoke"),
    ("GET", "/v1/vault/revocations"),
    ("POST", "/v1/vault/kill-switch"),
    ("GET", "/v1/vault/executions"),
    ("GET", "/v1/vault/executions/{action_id}"),
    ("GET", "/v1/vault/receipts"),
    ("POST", "/v1/vault/runtime/executions"),
    ("GET", "/v1/vault/runtime/executions/{action_id}"),
    ("GET", "/v1/vault/runtime/grants"),
    ("POST", "/v1/vault/runtime/grants"),
    ("POST", "/v1/vault/secrets/{id}/rotations"),
    ("GET", "/v1/vault/rotations"),
    ("GET", "/v1/vault/rotations/{id}"),
    ("POST", "/v1/vault/rotations/{id}/activate"),
    ("POST", "/v1/vault/rotations/{id}/rollback"),
    ("POST", "/v1/vault/revocations/{id}/retry"),
    ("POST", "/v1/vault/revocations/{id}/lift"),
    ("POST", "/v1/vault/executions/{action_id}/resolve"),
    ("POST", "/v1/vault/runtime/grants/{id}/delegations"),
]

LIST_PATHS = {
    "/v1/vault/providers",
    "/v1/vault/secrets",
    "/v1/vault/bindings",
    "/v1/vault/runtime-identities",
    "/v1/vault/grants",
    "/v1/vault/revocations",
    "/v1/vault/executions",
    "/v1/vault/receipts",
    "/v1/vault/rotations",
    "/v1/vault/runtime/grants",
}

SUPERSET = {
    "id": "id-1",
    "action_id": ACTION,
    "version": 1,
    "secret": {"id": "s-1"},
    "binding": {"id": "b-1"},
    "identity": {"id": "i-1"},
    "token": "vrt_example_token",
    "status": "finished",
    "state": "succeeded",
}


def _content() -> BindingContent:
    return BindingContent(
        secret_id="s-1",
        upstream_identity="crm-service",
        tool_contract_ref="crm.get_customer@1",
        destination=Destination(host="crm.example.test"),
        auth=BearerAuth(),
        request=RequestTemplate(
            method="GET",
            path="/customers/{id}",
            path_params=[PathParam(name="id", pattern="^c_[0-9]+$")],
        ),
        effect="read",
    )


def _admin(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> Client:
    return Client(
        api_key=API_KEY,
        base_url=BASE,
        runtime_token="vrt_runtime_0001",
        transport=httpx.MockTransport(handler),
        vault_retry=FAST,
        **kw,
    )


def _recording(seen: list[tuple[str, str]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path in LIST_PATHS:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=SUPERSET)

    return handler


Spec = tuple[Callable[[Client], Any], str, tuple[Any, ...], dict[str, Any]]


def _specs() -> list[Spec]:
    secret = Sensitive("a value that never leaves in clear text")
    descriptor = ProviderDescriptor(
        kind="openbao", address="https://bao.test", mount="kv", auth=TokenAuth(env_var="BAO_TOKEN")
    )
    grant = {"binding_id": "b1", "max_uses": 1, "ttl_seconds": 60, "reason": "r"}
    new_secret = {"environment": "e", "name": "n", "secret_type": "api_key", "provider_id": "p1"}
    return [
        (lambda c: c.vault, "status", (), {}),
        (lambda c: c.vault.providers, "list", (), {}),
        (
            lambda c: c.vault.providers,
            "create",
            (),
            {"name": "p", "mode": "managed", "descriptor": descriptor},
        ),
        (lambda c: c.vault.providers, "get", ("p1",), {}),
        (lambda c: c.vault.providers, "health", ("p1",), {}),
        (lambda c: c.vault.providers, "set_state", ("p1",), {"state": "disabled", "reason": "r"}),
        (lambda c: c.vault.secrets, "list", (), {"environment": "e", "state": "active"}),
        (lambda c: c.vault.secrets, "create", (), {**new_secret, "value": secret}),
        (
            lambda c: c.vault.secrets,
            "register_reference",
            (),
            {**new_secret, "provider_ref": "kv/x", "provider_version": "3"},
        ),
        (lambda c: c.vault.secrets, "get", ("s1",), {}),
        (lambda c: c.vault.secrets, "add_version", ("s1",), {"value": secret}),
        (lambda c: c.vault.secrets, "rotate", ("s1",), {"value": secret, "overlap_seconds": 600}),
        (lambda c: c.vault.secrets, "revoke", ("s1",), {"reason": "r"}),
        (lambda c: c.vault.rotations, "start", ("s1",), {"value": secret}),
        (lambda c: c.vault.rotations, "list", (), {"secret_id": "s1", "state": "prepared"}),
        (lambda c: c.vault.rotations, "get", ("r1",), {}),
        (lambda c: c.vault.rotations, "activate", ("r1",), {}),
        (lambda c: c.vault.rotations, "rollback", ("r1",), {"reason": "r"}),
        (lambda c: c.vault.bindings, "list", (), {"agent_id": "agent://a/b"}),
        (
            lambda c: c.vault.bindings,
            "create",
            (),
            {
                "environment": "e",
                "name": "n",
                "agent_id": "agent://a/b",
                "tool_name": "t",
                "content": _content(),
            },
        ),
        (lambda c: c.vault.bindings, "get", ("b1",), {}),
        (lambda c: c.vault.bindings, "propose_version", ("b1",), {"content": _content()}),
        (lambda c: c.vault.bindings, "submit", ("b1", 2), {}),
        (lambda c: c.vault.bindings, "approve", ("b1", 2), {}),
        (lambda c: c.vault.bindings, "reject", ("b1", 2), {}),
        (lambda c: c.vault.bindings, "activate", ("b1", 2), {}),
        (lambda c: c.vault.bindings, "revoke", ("b1",), {"reason": "r"}),
        (lambda c: c.vault.grants, "list", (), {"binding_id": "b1"}),
        (lambda c: c.vault.grants, "request", (), {**grant, "version": 3}),
        (lambda c: c.vault.grants, "approve", ("g1",), {}),
        (lambda c: c.vault.grants, "deny", ("g1",), {}),
        (lambda c: c.vault.grants, "revoke", ("g1",), {"reason": "r"}),
        (lambda c: c.vault.runtime_identities, "list", (), {}),
        (
            lambda c: c.vault.runtime_identities,
            "issue",
            (),
            {"environment": "e", "agent_id": "agent://a/b", "label": "l", "ttl_seconds": 60},
        ),
        (lambda c: c.vault.runtime_identities, "revoke", ("i1",), {"reason": "r"}),
        (lambda c: c.vault.executions, "list", (), {"state": "outcome_unknown", "limit": 5}),
        (lambda c: c.vault.executions, "get", (ACTION,), {}),
        (
            lambda c: c.vault.executions,
            "resolve",
            (ACTION,),
            {"resolution": "applied", "note": "checked"},
        ),
        (lambda c: c.vault.receipts, "list", (), {"action_id": ACTION}),
        (lambda c: c.vault.revocations, "list", (), {}),
        (lambda c: c.vault.revocations, "retry", ("v1",), {}),
        (lambda c: c.vault.revocations, "lift", ("v1",), {"reason": "r"}),
        (
            lambda c: c.vault,
            "kill_switch",
            (),
            {"target_kind": "agent", "target_id": "a", "reason": "r"},
        ),
        (lambda c: c.vault.runtime, "get_execution", (ACTION,), {}),
        (lambda c: c.vault.runtime, "list_grants", (), {"binding_id": "b1"}),
        (lambda c: c.vault.runtime, "request_grant", (), grant),
        (
            lambda c: c.vault.runtime,
            "delegate_grant",
            ("g1",),
            {"delegate_agent_id": "agent://a/c", "max_uses": 1, "ttl_seconds": 60, "reason": "r"},
        ),
        (
            lambda c: c.vault.runtime,
            "execute",
            (),
            {"tool": "t", "binding": "b", "action_id": ACTION},
        ),
        (lambda c: c.tools, "execute", (), {"tool": "t", "binding": "b", "action_id": ACTION}),
        (lambda c: c.tools, "get_execution", (ACTION,), {}),
    ]


def _every_operation(client: Client) -> list[Callable[[], object]]:
    return [
        (lambda owner=owner(client), name=name, a=args, kw=kwargs: getattr(owner, name)(*a, **kw))
        for owner, name, args, kwargs in _specs()
    ]


def _matches(template: str, path: str) -> bool:
    pattern = re.escape(template)
    for name in ("{id}", "{version}", "{action_id}"):
        pattern = pattern.replace(re.escape(name), "[^/]+")
    return re.fullmatch(pattern, path) is not None


def test_the_sdk_reaches_every_operation_of_the_contract_and_nothing_else() -> None:
    seen: list[tuple[str, str]] = []
    client = _admin(_recording(seen))
    for operation in _every_operation(client):
        operation()
    for method, template in OPENAPI_OPERATIONS:
        assert any(m == method and _matches(template, p) for m, p in seen), (method, template)
    for method, path in seen:
        assert any(m == method and _matches(t, path) for m, t in OPENAPI_OPERATIONS), (method, path)


def test_every_sync_method_has_an_async_twin() -> None:
    client = Client()
    namespaces = [
        client.vault,
        client.vault.providers,
        client.vault.secrets,
        client.vault.rotations,
        client.vault.bindings,
        client.vault.grants,
        client.vault.runtime_identities,
        client.vault.executions,
        client.vault.receipts,
        client.vault.revocations,
        client.vault.runtime,
        client.tools,
    ]
    missing: list[str] = []
    for namespace in namespaces:
        for name in dir(namespace):
            member = getattr(namespace, name)
            if (
                name.startswith("_")
                or not callable(member)
                or not _is_vault_method(namespace, name)
            ):
                continue
            if name.startswith("a") and hasattr(namespace, name[1:]):
                continue
            if not hasattr(namespace, f"a{name}"):
                missing.append(f"{type(namespace).__name__}.{name}")
    assert missing == []


def _is_vault_method(namespace: object, name: str) -> bool:
    if type(namespace).__name__ != "ToolsResource":
        return True
    return name in {"execute", "get_execution"}


def _recorded_requests(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    inner = _recording([])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return inner(request)

    return handler


async def test_every_async_method_sends_exactly_what_its_sync_twin_sends() -> None:
    seen: list[httpx.Request] = []
    client = _admin(_recorded_requests(seen))
    for owner, name, args, kwargs in _specs():
        seen.clear()
        getattr(owner(client), name)(*args, **kwargs)
        (sync_request,) = seen
        seen.clear()
        await getattr(owner(client), f"a{name}")(*args, **kwargs)
        (async_request,) = seen
        label = f"{type(owner(client)).__name__}.{name}"
        assert (async_request.method, async_request.url) == (
            sync_request.method,
            sync_request.url,
        ), label
        assert async_request.content == sync_request.content, label
        assert async_request.headers["authorization"] == sync_request.headers["authorization"], (
            label
        )


def test_status_is_readable_when_locked_and_reports_the_server_evaluation(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/status",
        json={
            "installed": True,
            "entitled": False,
            "capability": {
                "id": "agents_vault",
                "enabled": False,
                "reason": "not_entitled",
                "required_plan": "plan-x",
            },
            "permissions": ["metadata.read"],
            "usage": {"providers": 1, "secrets": 2, "bindings": 3, "runtime_identities": 4},
            "limitations": ["A secret value is write-only: no endpoint returns it."],
        },
    )
    status = Client(api_key=API_KEY, base_url=BASE).vault.status()
    assert (status.installed, status.entitled, status.locked, status.upgrade_hint) == (
        True,
        False,
        True,
        True,
    )
    assert status.capability.required_plan == "plan-x"
    assert status.usage.runtime_identities == 4


def test_an_entitled_workspace_is_not_locked(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/v1/vault/status", json={"installed": True, "entitled": True}
    )
    status = Client(api_key=API_KEY, base_url=BASE).vault.status()
    assert (status.locked, status.upgrade_hint) == (False, False)


def test_secret_create_sends_the_value_once_and_returns_metadata_only(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/secrets",
        status_code=201,
        json={
            "secret": {"id": "s-1", "name": "crm", "state": "active", "revision": 1},
            "versions": [{"id": "v-1", "state": "active"}],
        },
    )
    detail = Client(api_key=API_KEY, base_url=BASE).vault.secrets.create(
        environment="prod",
        name="crm",
        secret_type="api_key",
        provider_id="p-1",
        value=Sensitive("super-secret-value-1234"),
        classification="restricted",
    )
    assert (detail.secret.id, detail.versions[0].state) == ("s-1", "active")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {
        "environment": "prod",
        "name": "crm",
        "secret_type": "api_key",
        "provider_id": "p-1",
        "value": "super-secret-value-1234",
        "classification": "restricted",
    }
    assert request.headers["content-type"] == "application/json"


def test_a_plain_string_value_is_refused_before_any_request(httpx_mock: HTTPXMock) -> None:
    with pytest.raises(TypeError):
        Client(api_key=API_KEY, base_url=BASE).vault.secrets.create(
            environment="e",
            name="n",
            secret_type="api_key",
            provider_id="p",
            value="plain",  # type: ignore[arg-type]
        )
    assert httpx_mock.get_requests() == []


def test_rotation_start_and_the_secret_shortcut_hit_the_same_route(httpx_mock: HTTPXMock) -> None:
    job = {
        "id": "r-1",
        "secret_id": "s-1",
        "state": "prepared",
        "direction": "forward",
        "overlap_seconds": 3600,
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/secrets/s-1/rotations",
        status_code=201,
        json=job,
        is_reusable=True,
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    first = vault.rotations.start("s-1", value=Sensitive("new-value-0001"), overlap_seconds=600)
    second = vault.secrets.rotate("s-1", value=Sensitive("new-value-0002"))
    assert (first.state, second.id) == ("prepared", "r-1")
    bodies = [json.loads(r.content) for r in httpx_mock.get_requests()]
    assert bodies == [
        {"value": "new-value-0001", "overlap_seconds": 600},
        {"value": "new-value-0002"},
    ]


def test_binding_content_is_validated_client_side_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValueError):
        BindingContent.model_validate(
            {**_content().model_dump(), "usage_rule": {"read_only": True}}
        )
    dumped = _content().model_dump(mode="json", exclude_none=True)
    assert dumped["request"]["body"] == {"mode": "none"}
    assert dumped["destination"] == {"scheme": "https", "host": "crm.example.test", "port": 443}
    assert dumped["auth"] == {"kind": "bearer"}


def test_binding_lifecycle_calls(httpx_mock: HTTPXMock) -> None:
    version = {"version": 2, "state": "in_review", "digest": "d"}
    base = f"{BASE}/v1/vault/bindings/b-1"
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/bindings",
        status_code=201,
        json={"binding": {"id": "b-1"}, "versions": [version]},
    )
    httpx_mock.add_response(method="POST", url=f"{base}/versions/2/submit", json=version)
    httpx_mock.add_response(
        method="POST", url=f"{base}/versions/2/decide", json={**version, "state": "approved"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{base}/versions/2/activate",
        json={"binding": {"id": "b-1", "state": "active", "active_version": 2}, "versions": []},
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    created = vault.bindings.create(
        environment="e", name="n", agent_id="agent://a/b", tool_name="t", content=_content()
    )
    assert created.binding.id == "b-1"
    assert vault.bindings.submit("b-1", 2).state == "in_review"
    assert vault.bindings.approve("b-1", 2).state == "approved"
    assert vault.bindings.activate("b-1", 2).binding.active_version == 2
    decide = [r for r in httpx_mock.get_requests() if r.url.path.endswith("/decide")][0]
    assert json.loads(decide.content) == {"approve": True}
    create = httpx_mock.get_requests()[0]
    assert json.loads(create.content)["content"]["request"]["path_params"] == [
        {"name": "id", "pattern": "^c_[0-9]+$"}
    ]


def test_approving_with_an_api_key_is_refused_by_the_server_and_typed(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/grants/g-1/decide",
        status_code=403,
        json={
            "error": {
                "code": "vault_permission_denied",
                "message": "permission grant.approve is required",
            }
        },
    )
    with pytest.raises(VaultPermissionDenied) as excinfo:
        Client(api_key=API_KEY, base_url=BASE).vault.grants.approve("g-1")
    assert "grant.approve" in str(excinfo.value)


def test_grant_filters_and_delegated_grant_fields(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/grants?binding_id=b-1&agent_id=agent%3A%2F%2Fa%2Fb&state=approved",
        json=[{"id": "g-2", "parent_grant_id": "g-1", "depth": 1, "state": "approved"}],
    )
    (grant,) = Client(api_key=API_KEY, base_url=BASE).vault.grants.list(
        binding_id="b-1", agent_id="agent://a/b", state="approved"
    )
    assert (grant.parent_grant_id, grant.depth) == ("g-1", 1)


def test_identity_issue_hides_the_token_and_hands_it_over_once(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/runtime-identities",
        status_code=201,
        json={
            "identity": {
                "id": "i-1",
                "agent_id": "agent://a/b",
                "assurance": "enrollment_token_declared_release",
            },
            "token": "vrt_one_time_token_value",
        },
    )
    issued = Client(api_key=API_KEY, base_url=BASE).vault.runtime_identities.issue(
        environment="prod",
        agent_id="agent://a/b",
        label="support",
        ttl_seconds=3600,
        declared_release="1.2.3",
    )
    assert "vrt_one_time_token_value" not in repr(issued) + str(issued) + issued.model_dump_json()
    assert issued.identity.id == "i-1"
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body == {
        "environment": "prod",
        "agent_id": "agent://a/b",
        "label": "support",
        "declared_release": "1.2.3",
        "ttl_seconds": 3600,
    }
    assert issued.token.consume() == "vrt_one_time_token_value"


def test_runtime_client_built_from_an_issued_identity_executes_with_that_token(
    httpx_mock: HTTPXMock,
) -> None:
    from agenomic.vault import IssuedIdentity, RuntimeIdentity

    issued = IssuedIdentity(identity=RuntimeIdentity(id="i-1"), token="vrt_handed_over")  # type: ignore[arg-type]
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/runtime/executions",
        json={"status": "finished", "action_id": ACTION, "state": "succeeded", "result": 1},
    )
    runtime = issued.runtime_client(BASE)
    assert runtime.tools.execute(tool="t", binding="b", action_id=ACTION).result == 1
    assert httpx_mock.get_requests()[0].headers["Authorization"] == "Bearer vrt_handed_over"
    assert issued.token.consume() == "vrt_handed_over"


def test_execution_resolution_sends_what_the_operator_established(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/executions/{ACTION}/resolve",
        json={"status": "finished", "action_id": ACTION, "state": "succeeded", "receipt_id": "r-2"},
    )
    status = Client(api_key=API_KEY, base_url=BASE).vault.executions.resolve(
        ACTION, resolution="applied", note="refund visible in the PSP dashboard"
    )
    assert (status.state, status.receipt_id) == ("succeeded", "r-2")
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "resolution": "applied",
        "note": "refund visible in the PSP dashboard",
    }


def test_execution_and_receipt_reads_use_the_documented_filters(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/executions?state=outcome_unknown&limit=5",
        json=[{"action_id": ACTION, "state": "outcome_unknown", "tool": "t"}],
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/receipts?action_id={ACTION}",
        json=[{"id": "r-1", "action_id": ACTION, "phase": "authorized", "body": {"k": 1}}],
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    (summary,) = vault.executions.list(state="outcome_unknown", limit=5)
    (receipt,) = vault.receipts.list(action_id=ACTION)
    assert (summary.state, receipt.phase, receipt.body) == (
        "outcome_unknown",
        "authorized",
        {"k": 1},
    )


def test_revocation_fields_and_retry_and_lift(httpx_mock: HTTPXMock) -> None:
    revocation = {
        "id": "v-1",
        "target_kind": "agent",
        "target_id": "agent://a/b",
        "agenomic_state": "lifted",
        "provider_state": "not_applicable",
        "attempts": 2,
        "next_attempt_at": "2026-10-05T10:00:00Z",
        "last_error": "provider_unreachable",
        "lifted_at": "2026-10-05T11:00:00Z",
        "lifted_by": "u-1",
        "lift_reason": "false alarm",
    }
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/vault/revocations/v-1/retry", json=revocation
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/vault/revocations/v-1/lift", json=revocation
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    retried = vault.revocations.retry("v-1")
    lifted = vault.revocations.lift("v-1", reason="false alarm")
    assert (retried.last_error, lifted.agenomic_state, lifted.lift_reason) == (
        "provider_unreachable",
        "lifted",
        "false alarm",
    )
    assert lifted.lifted_at is not None


def test_ids_are_percent_encoded_so_they_cannot_reach_another_route(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/v1/vault/secrets/a%2Fb%3Fx", json={"secret": {"id": "x"}}
    )
    Client(api_key=API_KEY, base_url=BASE).vault.secrets.get("a/b?x")


def test_reads_are_retried_but_a_failed_write_is_not(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/vault/providers", status_code=503)
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/vault/providers", json=[])
    client = Client(api_key=API_KEY, base_url=BASE, vault_retry=FAST)
    assert client.vault.providers.list() == []
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/vault/secrets/s-1/revoke", status_code=503
    )
    with pytest.raises(VaultError) as excinfo:
        client.vault.secrets.revoke("s-1", reason="r")
    assert excinfo.value.status == 503
    assert len([r for r in httpx_mock.get_requests() if r.method == "POST"]) == 1


def test_a_write_is_retried_only_when_the_server_throttled_it(httpx_mock: HTTPXMock) -> None:
    url = f"{BASE}/v1/vault/kill-switch"
    httpx_mock.add_response(
        method="POST",
        url=url,
        status_code=429,
        headers={"Retry-After": "0"},
        json={"error": {"code": "too_many_requests", "message": "slow"}},
    )
    httpx_mock.add_response(method="POST", url=url, json={"id": "v-1"})
    out = Client(api_key=API_KEY, base_url=BASE, vault_retry=FAST).vault.kill_switch(
        target_kind="agent", target_id="a", reason="r"
    )
    assert out.id == "v-1"
    assert len(httpx_mock.get_requests()) == 2


def test_a_write_that_cannot_be_sent_is_not_retried_and_says_so(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("refused"), method="POST", url=f"{BASE}/v1/vault/secrets/s-1/revoke"
    )
    with pytest.raises(VaultTransportError):
        Client(api_key=API_KEY, base_url=BASE, vault_retry=FAST).vault.secrets.revoke(
            "s-1", reason="r"
        )
    assert len(httpx_mock.get_requests()) == 1


def test_a_response_can_never_carry_a_value_into_a_model(httpx_mock: HTTPXMock) -> None:
    canary = "leaked-by-a-broken-server-9d2c"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/vault/secrets/s-1",
        json={
            "secret": {"id": "s-1", "value": canary, "secret_value": canary},
            "versions": [],
            "value": canary,
        },
    )
    detail = Client(api_key=API_KEY, base_url=BASE).vault.secrets.get("s-1")
    assert canary not in repr(detail) + detail.model_dump_json()
    assert not hasattr(detail.secret, "value")


def test_a_malformed_list_or_object_is_an_invalid_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/vault/providers", json={"not": "a list"})
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/v1/vault/providers/p-1", json={"name": "no id"}
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    with pytest.raises(VaultError) as first:
        vault.providers.list()
    with pytest.raises(VaultError) as second:
        vault.providers.get("p-1")
    assert (first.value.code, second.value.code) == ("invalid_response", "invalid_response")
    assert uuid.UUID(ACTION)


def test_a_provider_descriptor_and_a_binding_content_can_be_passed_as_plain_mappings(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/vault/providers", status_code=201, json={"id": "p-1"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/bindings",
        status_code=201,
        json={"binding": {"id": "b-1"}},
    )
    vault = Client(api_key=API_KEY, base_url=BASE).vault
    descriptor = {
        "kind": "openbao",
        "address": "https://bao.test",
        "mount": "kv",
        "auth": {"method": "token", "env_var": "T"},
    }
    content = {"secret_id": "s-1", "effect": "read"}
    vault.providers.create(name="p", mode="byov", descriptor=descriptor)
    vault.bindings.create(environment="e", name="n", agent_id="a", tool_name="t", content=content)
    first, second = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert first["descriptor"] == descriptor
    assert second["content"] == content
