"""Coverage of the remaining benchmark resource, bridge wrapper and CLI paths."""

from __future__ import annotations

import asyncio
import json

import pytest

from agenomic import Client
from agenomic.benchmarks import (
    AgentTargetBridge,
    BridgeServer,
    FixtureBridge,
    TurnReply,
    TurnRequest,
    aserve_bridge,
    serve_bridge,
)
from agenomic.cli.__main__ import main

BASE = "https://api.test"


def _client() -> Client:
    return Client(api_key="key_123", base_url=BASE)


def test_catalog_card_plans_runs_and_policies_endpoints(httpx_mock) -> None:
    client = _client()
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/catalog/agentdojo",
        json={"card": {"id": "agentdojo"}},
    )
    assert client.benchmarks.card("agentdojo")["card"]["id"] == "agentdojo"

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/sessions/rmp_1/benchmarks/plans",
        json={"plans": [{"plan_id": "bplan_1"}]},
    )
    assert client.benchmarks.list_plans("rmp_1") == [{"plan_id": "bplan_1"}]

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/plans/bplan_1",
        json={"view": {"plan": {"plan_id": "bplan_1"}}},
    )
    assert client.benchmarks.get_plan("bplan_1")["plan"]["plan_id"] == "bplan_1"

    httpx_mock.add_response(
        method="PUT",
        url=f"{BASE}/v1/rmp/benchmarks/plans/bplan_1",
        json={"plan": {"plan_id": "bplan_1", "revision": 2}},
    )
    updated = client.benchmarks.update_plan(
        "bplan_1",
        [{"benchmark_id": "agentdojo"}],
        budget={"max_credits": 5},
        target="customer_agent",
    )
    assert updated["revision"] == 2
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "selections": [{"benchmark_id": "agentdojo"}],
        "budget": {"max_credits": 5},
        "target": "customer_agent",
    }

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/rmp/benchmarks/plans/bplan_1/cancel",
        json={"view": {"plan": {"status": "cancelled"}}},
    )
    assert client.benchmarks.cancel_plan("bplan_1")["plan"]["status"] == "cancelled"

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/plans/bplan_2/compare?baseline=bplan_1",
        json={"comparison": {"paired_tasks": 3}},
    )
    assert client.benchmarks.compare("bplan_2", "bplan_1") == {"paired_tasks": 3}

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/runs?session_id=rmp_1&limit=10",
        json={"runs": [{"run_id": "brun_1"}]},
    )
    assert client.benchmarks.list_runs("rmp_1", limit=10) == [{"run_id": "brun_1"}]

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/rmp/benchmarks/runs/brun_1/cancel",
        json={"run": {"status": "cancelling"}},
    )
    assert client.benchmarks.cancel_run("brun_1")["status"] == "cancelling"

    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/policies?agent_id=agent%3A%2F%2Facme%2Fsupport&plan_id=bplan_1",
        json={"policies": [{"proposal_id": "bpol_1"}]},
    )
    assert client.benchmarks.list_policies(agent="agent://acme/support", plan_id="bplan_1") == [
        {"proposal_id": "bpol_1"}
    ]

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/rmp/benchmarks/policies",
        json={"policy": {"proposal_id": "bpol_2"}},
    )
    assert (
        client.benchmarks.propose_policy({"plan_id": "bplan_1", "title": "deny"})["proposal_id"]
        == "bpol_2"
    )

    httpx_mock.add_response(
        method="GET", url=f"{BASE}/v1/rmp/benchmarks/policies", json={"unexpected": True}
    )
    assert client.benchmarks.list_policies() == []


def _idle_gateway(httpx_mock, bridge_id: str) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/rmp/benchmarks/bridge/register",
        json={"bridge_id": bridge_id},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/bridge/turns/next?agent_id=a&bridge_id={bridge_id}&wait=20",
        json={"turn": None},
    )


def test_sync_wrappers_delegate_to_the_async_server(httpx_mock) -> None:
    _idle_gateway(httpx_mock, "b")
    server = BridgeServer(_client(), FixtureBridge(), agent="a", bridge_id="b", release_id=None)
    assert (server.agent, server.release_id, server.bridge_id, server.wait_seconds) == (
        "a",
        None,
        "b",
        20,
    )
    assert server.register()["bridge_id"] == "b"
    assert server.poll_once() is False
    server.stop()
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/rmp/benchmarks/bridge/register", json={"bridge_id": "b"}
    )
    assert server.serve() == 0
    assert server.turns_answered == 0


def test_serve_helpers_stop_when_idle(httpx_mock) -> None:
    _idle_gateway(httpx_mock, "b")
    assert serve_bridge(_client(), FixtureBridge(), agent="a", bridge_id="b", idle_timeout=0) == 0
    _idle_gateway(httpx_mock, "c")
    assert (
        asyncio.run(
            aserve_bridge(_client(), FixtureBridge(), agent="a", bridge_id="c", idle_timeout=0)
        )
        == 0
    )


class EchoBridge(AgentTargetBridge):
    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        return TurnReply(content="echo", stop=True)


def test_cli_benchmark_serve_with_fixture_and_module_bridges(
    httpx_mock, capsys, monkeypatch
) -> None:
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/rmp/benchmarks/bridge/register", json={})
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/bridge/turns/next?agent_id=a&bridge_id=bridge_cli&wait=20",
        json={"turn": None},
    )
    monkeypatch.setattr("ulid.new", lambda: type("U", (), {"str": "cli"})())
    rc = main(
        [
            "benchmark",
            "serve",
            "--agent",
            "a",
            "--bridge",
            "fixture",
            "--base-url",
            BASE,
            "--api-key",
            "key_123",
            "--idle-timeout",
            "0",
        ]
    )
    assert rc == 0
    assert "bridge stopped after 0 turn(s)" in capsys.readouterr().out

    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/rmp/benchmarks/bridge/register", json={})
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/v1/rmp/benchmarks/bridge/turns/next?agent_id=a&bridge_id=bridge_cli&wait=20",
        json={"turn": None},
    )
    rc = main(
        [
            "benchmark",
            "serve",
            "--agent",
            "a",
            "--bridge",
            f"{__name__}:EchoBridge",
            "--base-url",
            BASE,
            "--api-key",
            "key_123",
            "--idle-timeout",
            "0",
        ]
    )
    assert rc == 0


@pytest.mark.parametrize("bridge", ["no_attr_here", "json:dumps"])
def test_cli_benchmark_serve_rejects_invalid_bridges(bridge: str, capsys) -> None:
    rc = main(["benchmark", "serve", "--agent", "a", "--bridge", bridge, "--base-url", BASE])
    assert rc == 2
    assert "error: --bridge" in capsys.readouterr().err
