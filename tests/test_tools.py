"""Tests for the replay tool execution SDK surface (``client.tools``)."""

from __future__ import annotations

import json
from typing import Any

import httpx
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


def test_cloud_client_never_falls_back_to_the_local_engine() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = Client(api_key="key", base_url=BASE, transport=httpx.MockTransport(unreachable))
    with pytest.raises(ToolExecutionError) as excinfo:
        client.tools.validate(
            config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"}
        )
    assert excinfo.value.code == "transport_error"
    with pytest.raises(ToolExecutionError) as excinfo:
        _ = client.tools.local
    assert excinfo.value.code == "cloud_required"


MOCK_CONFIG = {
    "schema_version": "agenomic.tool_execution/v1",
    "mode": "mock",
    "bindings": {
        "email.send": {
            "mode": "mock",
            "strategy": "static",
            "response": {
                "kind": "structured",
                "data": {"delivered": True, "message_id": "msg_0001"},
            },
        },
        "weather.now": {
            "mode": "mock",
            "strategy": "rules",
            "rules": [
                {
                    "id": "paris",
                    "priority": 5,
                    "when": {"args_match": {"city": "Paris"}},
                    "then": {"kind": "structured", "data": {"temp_c": 18}},
                },
                {"id": "elsewhere", "then": {"kind": "structured", "data": {"temp_c": 10}}},
            ],
        },
        "documents.extract": {"mode": "mock", "strategy": "recorded", "fixture_set_ref": "docs@1"},
    },
}


def _local_tools_with_fixtures() -> Any:
    tools = Client().tools
    created = tools.create_fixture_set(
        name="docs",
        version=1,
        fixtures=[
            {
                "fixture_id": "fx-1",
                "request": {"tool": "documents.extract", "arguments": {"doc": "a"}},
                "outcome": {"kind": "structured", "data": {"text": "hello"}},
                "fidelity": "recorded_response",
                "origin": "authored",
            }
        ],
    )
    tools.approve_fixture_set(created["fixture_set"]["id"])
    return tools


def test_local_mode_runs_static_rules_and_recorded_mocks_without_network() -> None:
    tools = _local_tools_with_fixtures()
    assert tools.validate(config=MOCK_CONFIG)["warnings"] == [
        "no contract for email.send",
        "no contract for weather.now",
        "no contract for documents.extract",
    ]
    plan = tools.preflight(config=MOCK_CONFIG, repetitions=2)
    assert plan["runnable"]
    assert plan["plan"]["mock_tools"] == ["email.send", "weather.now", "documents.extract"]
    run = tools.create_run(name="offline", config=MOCK_CONFIG, repetitions=2)
    assert run["status"] == "approved"
    tools.start_run(run["id"])
    router = tools.router(run["id"], repetition=2)
    assert router.call("email.send", {"to": "a@example.test"}) == {
        "delivered": True,
        "message_id": "msg_0001",
    }
    assert router.call("weather.now", {"city": "Paris"}) == {"temp_c": 18}
    assert router.call("weather.now", {"city": "Lyon"}) == {"temp_c": 10}
    assert router.call("documents.extract", {"doc": "a"}) == {"text": "hello"}
    assert router.calls[1].provenance.rule_id == "paris"
    assert router.calls[3].source == "recorded"
    assert router.calls[3].provenance.fixture_id == "fx-1"
    with pytest.raises(ToolExecutionError) as exc:
        router.call("documents.extract", {"doc": "unknown"})
    assert exc.value.code == "mock_unmatched"
    with pytest.raises(ToolExecutionError) as exc:
        router.call("tickets.get", {"ticket_id": "x"})
    assert exc.value.code == "tool_unknown"
    tools.complete_run(run["id"])
    report = tools.report(run["id"])["report"]
    assert report["calls_by_source"] == {"static": 3, "recorded": 1, "unrouted": 2}
    assert report["has_real_calls"] is False
    assert len(report["uncovered_calls"]) == 2
    assert tools.export(run["id"])["export_version"] == "agenomic.tool_run_export/v1"
    assert router.summary() == {
        "calls": 4,
        "by_source": {"static": 3, "recorded": 1},
        "has_real_calls": False,
        "unreported": 0,
    }


def test_local_mode_refuses_what_needs_the_cloud_at_preflight() -> None:
    tools = Client().tools
    tools.create_profile(name="staging", allowed_env=["CRM_MCP_URL"])
    config = {
        "schema_version": "agenomic.tool_execution/v1",
        "mode": "hybrid",
        "environment_profile": "staging",
        "bindings": {
            "crm.get_customer": {
                "mode": "live",
                "adapter": "mcp",
                "endpoint": "${env:CRM_MCP_URL}",
                "effect": "read",
            },
            "tickets.create": {"mode": "mock", "strategy": "scenario", "scenario_ref": "tickets@1"},
        },
    }
    plan = tools.preflight(config=config)
    assert plan["runnable"] is False
    assert plan["plan"]["missing_capabilities"] == [
        "adapter mcp requires Agenomic Cloud",
        "strategy scenario requires Agenomic Cloud",
    ]
    with pytest.raises(ToolExecutionError) as exc:
        tools.create_run(config=config)
    assert exc.value.code == "tool_execution_config_invalid"
    with pytest.raises(ToolExecutionError) as exc:
        tools.validate(
            config={
                "schema_version": "agenomic.tool_execution/v1",
                "mode": "mock",
                "safety": {"allow_implicit_fallback": True},
            }
        )
    assert exc.value.code == "tool_execution_config_invalid"
    with pytest.raises(ToolExecutionError) as exc:
        tools.validate(
            config={
                "schema_version": "agenomic.tool_execution/v1",
                "mode": "mock",
                "bindings": {
                    "x": {
                        "mode": "mock",
                        "strategy": "static",
                        "response": {"kind": "structured", "data": "${env:SECRET}"},
                    }
                },
            }
        )
    assert exc.value.code == "env_reference_forbidden_location"
    with pytest.raises(ToolExecutionError) as exc:
        tools.test_connection(
            profile="staging", tool="crm.get_customer", binding={"mode": "live", "adapter": "http"}
        )
    assert exc.value.code == "cloud_required"


LOCAL_FUNCTION_CONFIG = {
    "schema_version": "agenomic.tool_execution/v1",
    "mode": "hybrid",
    "environment_profile": "local-only",
    "limits": {"max_live_calls": 1},
    "bindings": {
        "math.add": {"mode": "live", "adapter": "local", "function": "math.add", "effect": "read"},
        "email.send": {
            "mode": "mock",
            "strategy": "static",
            "response": {"kind": "structured", "data": {"delivered": True}},
        },
    },
}


def test_local_mode_local_functions_follow_the_two_phase_protocol() -> None:
    tools = Client().tools
    tools.create_profile(name="local-only")
    run = tools.create_run(config=LOCAL_FUNCTION_CONFIG)
    assert run["status"] == "planned"
    with pytest.raises(ToolExecutionError) as exc:
        tools.start_run(run["id"])
    assert exc.value.code == "plan_approval_required"
    with pytest.raises(ToolExecutionError) as exc:
        tools.approve_run(run["id"], plan_hash="sha256:other")
    assert exc.value.code == "plan_approval_required"
    tools.approve_run(run["id"], plan_hash=run["plan_hash"])
    tools.start_run(run["id"])
    effects: list[str] = []

    def add(a: int, b: int) -> dict[str, int]:
        effects.append("ran")
        return {"sum": a + b}

    router = tools.router(
        run["id"],
        local_functions={"math.add": add, "email.send": lambda **_: effects.append("never")},
    )
    assert router.call("email.send", {"to": "a@example.test"}) == {"delivered": True}
    assert effects == []
    assert router.call("math.add", {"a": 2, "b": 3}) == {"sum": 5}
    assert effects == ["ran"]
    with pytest.raises(ToolExecutionError) as exc:
        router.call("math.add", {"a": 1, "b": 1})
    assert exc.value.code == "live_budget_exhausted"
    assert effects == ["ran"]
    records = tools.report(run["id"])["invocations"]
    assert [(r["tool"], r["source"], r["status"]) for r in records] == [
        ("email.send", "static", "success"),
        ("math.add", "runtime_local", "success"),
    ]
    assert router.summary()["has_real_calls"] is True
    with pytest.raises(ToolExecutionError) as exc:
        tools.report_local(run["id"], "math.add", {}, {"sum": 0}, logical_call_id="ghost")
    assert exc.value.code == "live_call_denied"


async def test_async_router_awaits_local_coroutines_and_engine_calls() -> None:
    tools = Client().tools
    tools.create_profile(name="local-only")
    run = tools.create_run(config=LOCAL_FUNCTION_CONFIG)
    tools.approve_run(run["id"], plan_hash=run["plan_hash"])
    tools.start_run(run["id"])

    async def add(a: int, b: int) -> dict[str, int]:
        return {"sum": a + b}

    router = tools.arouter(run["id"], local_functions={"math.add": add})
    assert await router.call("email.send", {"to": "a@example.test"}) == {"delivered": True}
    assert await router.wrap("math.add")(a=2, b=3) == {"sum": 5}
    assert router.summary()["by_source"] == {"static": 1, "runtime_local": 1}


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
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize",
        json={"decision": "local", "record_id": "rec_local"},
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
    requests = httpx_mock.get_requests()
    assert [r.url.path.rsplit("/", 1)[-1] for r in requests] == [
        "invoke",
        "authorize",
        "report-local",
    ]
    authorized = json.loads(requests[1].content)
    assert authorized["tool"] == "math.add"
    assert authorized["logical_call_id"] == "math.add#2"
    reported = json.loads(requests[2].content)
    assert reported["tool"] == "math.add"
    assert reported["result"] == {"sum": 5}
    assert reported["logical_call_id"] == "math.add#2"
    assert router.calls[1].record_id == "rec_local"
    assert router.summary() == {
        "calls": 2,
        "by_source": {"static": 1, "runtime_local": 1},
        "has_real_calls": True,
        "unreported": 0,
    }


def test_offline_router_refuses_before_executing_local_function() -> None:
    effects: list[str] = []
    router = Client().tools.router(
        RUN, local_functions={"email.send": lambda: effects.append("sent")}
    )
    with pytest.raises(ToolExecutionError) as exc:
        router.call("email.send")
    assert exc.value.code == "not_found"
    assert effects == []
    assert router.summary()["calls"] == 0


def test_router_refuses_to_execute_without_a_record_id(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize",
        json={"decision": "local"},
    )
    effects: list[str] = []
    router = Client(api_key="key", base_url=BASE).tools.router(
        RUN, local_functions={"email.send": lambda: effects.append("sent")}
    )
    with pytest.raises(ToolExecutionError) as exc:
        router.call("email.send")
    assert exc.value.code == "invalid_response"
    assert effects == []


def test_malformed_envelope_is_a_typed_error(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json={
            "result": 1,
            "agenomic": {
                "record_id": "r",
                "status": "weird",
                "provenance": {"source": "static"},
                "external_state": "none",
            },
        },
    )
    with pytest.raises(ToolExecutionError) as exc:
        Client(api_key="key", base_url=BASE).tools.invoke(RUN, "email.send")
    assert exc.value.code == "invalid_response"


async def test_async_invoke_uses_the_async_transport(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"delivered": True}),
    )
    out = await Client(api_key="key", base_url=BASE).tools.ainvoke(
        RUN, "email.send", {"to": "a@example.test"}
    )
    assert out.result == {"delivered": True}
    assert out.provenance.source == "static"
    assert httpx_mock.get_requests()[0].headers["Idempotency-Key"]


@pytest.mark.parametrize("denial", ["live_call_denied", "run_not_active", "live_budget_exhausted"])
def test_gateway_refusal_prevents_local_side_effect(httpx_mock, denial: str) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize",
        status_code=400,
        json={"error": {"code": denial, "message": "rejected by gateway"}},
    )
    effects: list[str] = []
    router = Client(api_key="key", base_url=BASE).tools.router(
        RUN, local_functions={"email.send": lambda: effects.append("sent")}
    )
    with pytest.raises(ToolExecutionError) as exc:
        router.call("email.send")
    assert exc.value.code == denial
    assert effects == []
    assert [r.url.path.rsplit("/", 1)[-1] for r in httpx_mock.get_requests()] == ["authorize"]


def test_mock_bound_tool_is_routed_to_the_gateway_instead_of_the_local_function(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize",
        json={"decision": "gateway"},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/invoke",
        json=_invoke_response({"delivered": True, "mocked": True}),
    )
    effects: list[str] = []
    router = Client(api_key="key", base_url=BASE).tools.router(
        RUN, local_functions={"email.send": lambda **_: effects.append("sent")}
    )
    assert router.call("email.send", {"to": "a@example.test"}) == {
        "delivered": True,
        "mocked": True,
    }
    assert effects == []
    assert router.summary()["has_real_calls"] is False


def test_failed_report_keeps_local_evidence_as_unreported(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/local/authorize",
        json={"decision": "local", "record_id": "rec_pending"},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/tool-execution/runs/{RUN}/report-local",
        status_code=502,
        text="<html>Bad Gateway</html>",
    )
    effects: list[str] = []
    router = Client(api_key="key", base_url=BASE).tools.router(
        RUN, local_functions={"email.send": lambda **_: effects.append("sent") or {"ok": True}}
    )
    with pytest.raises(ToolExecutionError) as exc:
        router.call("email.send", {"to": "a@example.test"})
    assert exc.value.status == 502
    assert effects == ["sent"]
    assert len(router.calls) == 1
    assert router.calls[0].reported is False
    assert router.calls[0].external_state == "indeterminate"
    assert router.calls[0].record_id == "rec_pending"
    assert router.summary() == {
        "calls": 1,
        "by_source": {"runtime_local": 1},
        "has_real_calls": True,
        "unreported": 1,
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
