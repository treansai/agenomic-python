import json

import pytest

from agenomic import Client
from agenomic.benchmarks import (
    AgentTargetBridge,
    BridgeCapability,
    BridgeServer,
    CallableBridge,
    FixtureBridge,
    ToolCall,
    TurnReply,
    TurnRequest,
)
from agenomic.exceptions import CloudError


def test_local_client_refuses_benchmarks() -> None:
    client = Client()
    with pytest.raises(CloudError):
        client.benchmarks.catalog()
    with pytest.raises(CloudError):
        BridgeServer(client, FixtureBridge(), agent="agent://acme/support")


def test_rmp_start_still_does_not_launch_benchmarks(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/sessions",
        json={"session": {"session_id": "rmp_1", "agent_id": "agent://acme/support"}},
    )
    client = Client(api_key="key_123", base_url="https://api.test")
    client.rmp.start(agent="agent://acme/support")
    assert [r.url.path for r in httpx_mock.get_requests()] == ["/v1/rmp/sessions"]


def test_plan_preflight_launch_calls(httpx_mock) -> None:
    client = Client(api_key="key_123", base_url="https://api.test")
    httpx_mock.add_response(
        method="GET",
        url="https://api.test/v1/rmp/benchmarks/catalog?agent_id=agent%3A%2F%2Facme%2Fsupport",
        json={"benchmarks": [{"card": {"id": "agentdojo"}}]},
    )
    assert client.benchmarks.catalog(agent="agent://acme/support")[0]["card"]["id"] == "agentdojo"

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/sessions/rmp_1/benchmarks/plans",
        json={"plan": {"plan_id": "bplan_1", "status": "draft"}},
    )
    selections = [
        {
            "benchmark_id": "agentdojo",
            "benchmark_version": "v0.1.35",
            "scope": {"domains": ["workspace"]},
            "profile": "smoke",
        }
    ]
    plan = client.benchmarks.create_plan("rmp_1", selections, budget={"max_total_trials": 10})
    assert plan["plan_id"] == "bplan_1"
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {
        "target": "customer_agent",
        "selections": selections,
        "budget": {"max_total_trials": 10},
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/benchmarks/plans/bplan_1/preflight",
        json={
            "plan": {
                "plan_id": "bplan_1",
                "status": "preflight_passed",
                "preflight": {"passed": True},
            }
        },
    )
    assert client.benchmarks.preflight("bplan_1")["status"] == "preflight_passed"

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/benchmarks/plans/bplan_1/launch",
        json={
            "plan": {"plan_id": "bplan_1"},
            "runs": [{"run_id": "brun_1"}],
            "already_launched": False,
        },
    )
    launched = client.benchmarks.launch("bplan_1")
    assert launched["runs"][0]["run_id"] == "brun_1"
    assert launched["already_launched"] is False

    httpx_mock.add_response(
        method="GET",
        url="https://api.test/v1/rmp/benchmarks/runs/brun_1?after=0&limit=200",
        json={
            "run": {"run_id": "brun_1", "status": "running"},
            "trials": [],
            "events": [],
            "next_event_cursor": 0,
        },
    )
    assert client.benchmarks.get_run("brun_1")["run"]["status"] == "running"

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/benchmarks/policies/bpol_1/decide",
        json={"policy": {"proposal_id": "bpol_1", "status": "approved"}},
    )
    decided = client.benchmarks.decide_policy("bpol_1", "approved", manifest_hash="blake3:abc")
    assert decided["status"] == "approved"
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "to": "approved",
        "manifest_hash": "blake3:abc",
    }


def _turn(turn_id: str, task_id: str = "user_task_0", turn_index: int = 0) -> dict:
    return {
        "turn_id": turn_id,
        "deadline_at": "2026-09-13T00:00:00Z",
        "request": {
            "messages": [{"role": "user", "content": "do the task"}],
            "tools": [
                {"name": "send_email", "description": "send", "parameters": {"type": "object"}}
            ],
            "context": {
                "benchmark_id": "agentdojo",
                "run_id": "brun_1",
                "trial_id": "btrial_1",
                "task_id": task_id,
                "trial_index": 0,
                "turn_index": turn_index,
                "max_turns": 5,
                "instructions": "be helpful",
                "tracking_session_id": None,
                "target": "customer_agent",
            },
        },
    }


def test_bridge_registers_polls_and_replies_with_the_customer_agent(httpx_mock) -> None:
    client = Client(api_key="key_123", base_url="https://api.test")
    seen: list[TurnRequest] = []

    def agent(turn: TurnRequest) -> TurnReply:
        seen.append(turn)
        return TurnReply(
            tool_calls=[ToolCall(id="c1", name=turn.tools[0].name, arguments={"to": "x"})],
            usage={"input_tokens": 3, "output_tokens": 1},
        )

    bridge = CallableBridge(
        agent, capabilities=[BridgeCapability.MULTI_TURN, BridgeCapability.BENCHMARK_TOOLS]
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/benchmarks/bridge/register",
        json={"bridge_id": "b1"},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.test/v1/rmp/benchmarks/bridge/turns/next?agent_id=agent%3A%2F%2Facme%2Fsupport&bridge_id=b1&wait=20&release_id=rel_1",
        json={"turn": _turn("bturn_1")},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/benchmarks/bridge/turns/bturn_1/reply",
        json={"turn": {"turn_id": "bturn_1", "status": "answered"}},
    )

    server = BridgeServer(
        client, bridge, agent="agent://acme/support", release_id="rel_1", bridge_id="b1"
    )
    answered = server.serve(max_turns=1)
    assert answered == 1
    assert seen[0].task_id == "user_task_0"
    assert seen[0].instructions == "be helpful"
    requests = httpx_mock.get_requests()
    register = json.loads(requests[0].content)
    assert register["agent_id"] == "agent://acme/support"
    assert register["release_id"] == "rel_1"
    assert register["capabilities"] == ["multi_turn", "benchmark_tools"]
    assert requests[0].headers["Authorization"] == "Bearer key_123"
    reply = json.loads(requests[-1].content)
    assert reply == {
        "reply": {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "send_email", "arguments": {"to": "x"}}],
            },
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "stop": False,
        }
    }


def test_bridge_handler_failure_is_reported_not_hidden(httpx_mock) -> None:
    client = Client(api_key="key_123", base_url="https://api.test")

    class Broken(AgentTargetBridge):
        def handle_turn(self, turn: TurnRequest) -> TurnReply:
            raise RuntimeError("boom")

    httpx_mock.add_response(
        method="POST", url="https://api.test/v1/rmp/benchmarks/bridge/register", json={}
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.test/v1/rmp/benchmarks/bridge/turns/next?agent_id=a&bridge_id=b&wait=20",
        json={"turn": _turn("bturn_2")},
    )
    httpx_mock.add_response(
        method="POST", url="https://api.test/v1/rmp/benchmarks/bridge/turns/bturn_2/reply", json={}
    )
    BridgeServer(client, Broken(), agent="a", bridge_id="b").serve(max_turns=1)
    reply = json.loads(httpx_mock.get_requests()[-1].content)["reply"]
    assert reply["message"]["content"].startswith("[bridge error] RuntimeError")
    assert reply["stop"] is True
