"""Tests for the replay tool execution SDK surface (``client.tools``)."""

from __future__ import annotations

import json

import pytest

from agenomic import Client
from agenomic.tools import ToolCallError, ToolExecutionError, ToolRouter

BASE = "https://api.test"
RUN = "11111111-2222-4333-8444-555555555555"
CANARY = "tok_canary_sdk_0badf00d"


def _invoke_response(result, source="static", status="success"):
    return {
        "result": result,
        "agenomic": {
            "record_id": "rec_1",
            "status": status,
            "provenance": {
                "source": source,
                "binding_mode": "mock" if source != "live" else "live",
            },
            "external_state": "none" if source != "live" else "confirmed",
            "effects": [],
            "duration_ms": 3,
            "expected_error": False,
        },
    }


def test_local_client_has_no_fallback() -> None:
    client = Client()
    with pytest.raises(ToolExecutionError) as excinfo:
        client.tools.validate(
            config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"}
        )
    assert excinfo.value.code == "cloud_required"


def test_set_variable_is_write_only_and_status_never_carries_values(httpx_mock) -> None:
    httpx_mock.add_response(
        method="PUT",
        url=f"{BASE}/v1/tool-execution/profiles/p1/variables/CRM_API_TOKEN",
        json={"name": "CRM_API_TOKEN", "version": 1},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/tool-execution/profiles/p1",
        json={
            "profile": {"id": "p1", "name": "replay-staging", "allowed_env": ["CRM_API_TOKEN"]},
            "variables": [
                {"name": "CRM_API_TOKEN", "source": "stored", "version": 1, "available": True}
            ],
        },
    )
    client = Client(api_key="key", base_url=BASE)
    out = client.tools.set_variable("p1", "CRM_API_TOKEN", CANARY)
    assert out["version"] == 1
    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent == {"value": CANARY}
    status = client.tools.variable_status("p1")
    assert status[0]["available"] is True
    assert CANARY not in json.dumps(status)


def test_invoke_sends_normalized_invocation_with_stable_idempotency_key(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"customer": {"id": "c_1"}}, source="live"),
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"customer": {"id": "c_1"}}, source="live"),
    )
    client = Client(api_key="key", base_url=BASE)
    first = client.tools.invoke(RUN, "crm.get_customer", {"id": "c_1"}, logical_call_id="c1")
    second = client.tools.invoke(
        RUN, "crm.get_customer", {"id": "c_1"}, logical_call_id="c1", attempt=2
    )
    assert first.is_real
    assert first.ok
    assert first.result["customer"]["id"] == "c_1"
    requests = httpx_mock.get_requests()
    body = json.loads(requests[0].content)
    assert body == {
        "repetition": 1,
        "logical_call_id": "c1",
        "attempt": 1,
        "tool": "crm.get_customer",
        "arguments": {"id": "c_1"},
    }
    assert requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]
    assert requests[0].headers["Authorization"] == "Bearer key"
    assert second.record_id == "rec_1"


def test_router_routes_remote_and_reports_local_functions(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"delivered": True}),
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/report-local",
        json={"record_id": "rec_local"},
    )
    client = Client(api_key="key", base_url=BASE)
    router: ToolRouter = client.tools.router(
        RUN, local_functions={"math.add": lambda a, b: {"sum": a + b}}
    )
    assert router.call("email.send", {"to": "a@example.test"}) == {"delivered": True}
    send = router.wrap("email.send")
    assert send.__name__ == "email_send"
    assert router.call("math.add", {"a": 2, "b": 3}) == {"sum": 5}
    reported = json.loads(httpx_mock.get_requests()[1].content)
    assert reported["tool"] == "math.add"
    assert reported["result"] == {"sum": 5}
    assert reported["logical_call_id"] == "math.add#2"
    assert router.summary() == {
        "calls": 2,
        "by_source": {"static": 1, "runtime_local": 1},
        "has_real_calls": True,
    }


def test_router_raises_typed_error_on_error_outcome(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response(
            {"error": {"code": "not_found", "message": "no such ticket"}}, status="error"
        ),
    )
    client = Client(api_key="key", base_url=BASE)
    router = client.tools.router(RUN)
    with pytest.raises(ToolCallError) as excinfo:
        router.call("tickets.get", {"ticket_id": "ticket_9999"})
    assert excinfo.value.code == "not_found"
    assert excinfo.value.envelope.status == "error"
    lenient = client.tools.router(RUN, raise_on_error=False)
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"error": {"code": "not_found"}}, status="error"),
    )
    assert lenient.call("tickets.get", {"ticket_id": "x"})["error"]["code"] == "not_found"


def test_server_errors_surface_the_error_code(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        status_code=400,
        json={"error": {"code": "mock_unmatched", "message": "tool x: no mock matched"}},
    )
    client = Client(api_key="key", base_url=BASE)
    with pytest.raises(ToolExecutionError) as excinfo:
        client.tools.invoke(RUN, "x", {})
    assert excinfo.value.code == "mock_unmatched"
    assert excinfo.value.status == 400


def test_preflight_and_approval_flow(httpx_mock) -> None:
    config = {"schema_version": "agenomic.tool_execution/v1", "mode": "mock", "bindings": {}}
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/preflight",
        json={"plan": {"has_live": False}, "plan_hash": "blake3:abc", "runnable": True},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs",
        json={"run": {"id": RUN, "status": "approved", "plan_hash": "blake3:abc"}},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/start",
        json={"run": {"id": RUN, "status": "running"}},
    )
    client = Client(api_key="key", base_url=BASE)
    plan = client.tools.preflight(config=config, repetitions=3)
    assert plan["runnable"] is True
    run = client.tools.create_run(name="demo", config=config, repetitions=3)
    assert run["status"] == "approved"
    assert client.tools.start_run(RUN)["status"] == "running"
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body == {"repetitions": 3, "config": config}
    with pytest.raises(ValueError):
        client.tools.preflight(config=config, config_text="{}")
