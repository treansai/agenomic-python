from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from agenomic import Client
from agenomic.benchmarks import AgentTargetBridge, BridgeServer, ToolCall, TurnReply, TurnRequest
from agenomic.exceptions import CloudError


@dataclass
class Exchange:
    method: str
    path: str
    response: object = None
    status: int = 200
    body: object = None
    authorization: str | None = "Bearer integration-key"


@contextmanager
def http_gateway(exchanges: list[Exchange]) -> Iterator[str]:
    """Serve scripted gateway responses over real loopback HTTP without a cloud backend."""
    pending = list(exchanges)
    errors: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def dispatch(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                assert pending, f"Unexpected {self.command} {self.path}"
                expected = pending.pop(0)
                assert (self.command, self.path) == (expected.method, expected.path)
                assert self.headers.get("Authorization") == expected.authorization
                assert (json.loads(raw) if raw else None) == expected.body
                if raw:
                    assert self.headers.get_content_type() == "application/json"
                payload = b"" if expected.status == 204 else json.dumps(expected.response).encode()
                self.send_response(expected.status)
            except (AssertionError, ValueError) as exc:
                errors.append(str(exc))
                payload = b'{"error":"unexpected request"}'
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self.dispatch()

        def do_POST(self) -> None:
            self.dispatch()

        def do_PUT(self) -> None:
            self.dispatch()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors, errors
        assert not pending, pending


def wire_turn(turn_id: str, trial_id: str, turn_index: int) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "deadline_at": "2030-01-01T00:00:00Z",
        "request": {
            "messages": [{"role": "user", "content": "Vérifie le café ☕"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object"}}],
            "context": {
                "benchmark_id": "agentdojo",
                "run_id": "run_1",
                "trial_id": trial_id,
                "task_id": "task_1",
                "trial_index": 0,
                "turn_index": turn_index,
                "max_turns": 3,
                "instructions": "Utilise les outils du benchmark",
                "tracking_session_id": "tracking_1",
                "target": "customer_agent",
            },
        },
    }


def registration() -> Exchange:
    import platform

    from agenomic._version import __version__

    return Exchange(
        "POST",
        "/v1/rmp/benchmarks/bridge/register",
        {"bridge_id": "bridge_1"},
        body={
            "agent_id": "agent://acme/support",
            "release_id": "release_1",
            "bridge_id": "bridge_1",
            "capabilities": ["multi_turn", "benchmark_tools"],
            "sdk": f"agenomic-python/{__version__} ({platform.python_implementation()} {platform.python_version()})",
        },
    )


NEXT = "/v1/rmp/benchmarks/bridge/turns/next?agent_id=agent%3A%2F%2Facme%2Fsupport&bridge_id=bridge_1&wait=0&release_id=release_1"


class RecordingBridge(AgentTargetBridge):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def start_trial(self, turn: TurnRequest) -> None:
        self.events.append(("start", turn.trial_id))

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        self.events.append(("handle", turn.turn_id))
        assert turn.messages[0].content == "Vérifie le café ☕"
        assert turn.tracking_session_id == "tracking_1"
        if turn.trial_id == "trial_2":
            raise RuntimeError("agent failed")
        return TurnReply(
            tool_calls=[ToolCall(id="call_1", name=turn.tools[0].name, arguments={"q": "café"})],
            usage={"input_tokens": 3, "output_tokens": 2},
        )

    def end_trial(self, trial_id: str) -> None:
        self.events.append(("end", trial_id))


def bridge_server(base_url: str, bridge: AgentTargetBridge) -> BridgeServer:
    return BridgeServer(
        Client(api_key="integration-key", base_url=base_url, timeout=2),
        bridge,
        agent="agent://acme/support",
        release_id="release_1",
        bridge_id="bridge_1",
        wait_seconds=0,
    )


def test_bridge_http_multiturn_lifecycle_and_agent_failure() -> None:
    bridge = RecordingBridge()
    exchanges = [registration()]
    for turn_id, trial_id, turn_index in [
        ("turn_1", "trial_1", 0),
        ("turn_2", "trial_1", 1),
        ("turn_3", "trial_2", 0),
    ]:
        reply = (
            TurnReply(content="[bridge error] handler raised RuntimeError", stop=True)
            if trial_id == "trial_2"
            else TurnReply(
                tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "café"})],
                usage={"input_tokens": 3, "output_tokens": 2},
            )
        )
        exchanges.extend(
            [
                Exchange("GET", NEXT, {"turn": wire_turn(turn_id, trial_id, turn_index)}),
                Exchange(
                    "POST",
                    f"/v1/rmp/benchmarks/bridge/turns/{turn_id}/reply",
                    {"turn": {"status": "answered"}},
                    body={"reply": reply.to_wire()},
                ),
            ]
        )
    with http_gateway(exchanges) as base_url:
        server = bridge_server(base_url, bridge)
        assert server.serve(max_turns=3) == 3
    assert bridge.events == [
        ("start", "trial_1"),
        ("handle", "turn_1"),
        ("handle", "turn_2"),
        ("end", "trial_1"),
        ("start", "trial_2"),
        ("handle", "turn_3"),
        ("end", "trial_2"),
    ]


def test_bridge_http_empty_poll_204_is_idle() -> None:
    bridge = RecordingBridge()
    with http_gateway([registration(), Exchange("GET", NEXT, status=204)]) as base_url:
        server = bridge_server(base_url, bridge)
        assert server.serve(idle_timeout=0) == 0
    assert bridge.events == []


@pytest.mark.parametrize("status", [401, 403])
def test_bridge_http_authorization_failure_stops_before_polling(status: int) -> None:
    bridge = RecordingBridge()
    exchange = registration()
    exchange.status = status
    exchange.response = {"error": "write key required"}
    with http_gateway([exchange]) as base_url:
        server = bridge_server(base_url, bridge)
        with pytest.raises(CloudError, match=str(status)):
            server.serve(max_turns=1)
        assert server.turns_answered == 0
    assert bridge.events == []


def test_bridge_http_rejected_reply_is_not_counted_as_answered() -> None:
    bridge = RecordingBridge()
    reply = TurnReply(
        tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "café"})],
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    exchanges = [
        registration(),
        Exchange("GET", NEXT, {"turn": wire_turn("turn_1", "trial_1", 0)}),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/bridge/turns/turn_1/reply",
            {"error": "expired turn"},
            status=409,
            body={"reply": reply.to_wire()},
        ),
    ]
    with http_gateway(exchanges) as base_url:
        server = bridge_server(base_url, bridge)
        with pytest.raises(CloudError, match="409"):
            server.serve(max_turns=1)
        assert server.turns_answered == 0


def test_benchmarks_http_plan_update_launch_compare_and_cancel() -> None:
    selection = {"benchmark_id": "agentdojo", "benchmark_version": "v0.1.35", "profile": "smoke"}
    selections = [selection]
    plan = {"plan_id": "plan_1", "status": "draft"}
    run = {"run_id": "run_1", "status": "running"}
    exchanges = [
        Exchange(
            "POST",
            "/v1/rmp/sessions/session_1/benchmarks/plans",
            {"plan": plan},
            body={
                "target": "customer_agent",
                "selections": selections,
                "budget": {"max_total_trials": 2},
            },
        ),
        Exchange(
            "PUT",
            "/v1/rmp/benchmarks/plans/plan_1",
            {"plan": {**plan, "revision": 2}},
            body={"selections": selections, "budget": {}},
        ),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/plans/plan_1/preflight",
            {"plan": {**plan, "status": "preflight_passed"}},
            body={},
        ),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/plans/plan_1/launch",
            {"plan": plan, "runs": [run], "already_launched": False},
            body={},
        ),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/plans/plan_1/launch",
            {"plan": plan, "runs": [run], "already_launched": True},
            body={},
        ),
        Exchange(
            "GET",
            "/v1/rmp/benchmarks/plans/plan_1/compare?baseline=baseline%2F1%3F%26",
            {"comparison": {"paired_tasks": 1}},
        ),
        Exchange(
            "GET",
            "/v1/rmp/benchmarks/runs/run_1?after=7&limit=3",
            {"run": run, "trials": [], "events": [{"sequence": 8}], "next_event_cursor": 8},
        ),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/runs/run_1/cancel",
            {"run": {**run, "status": "cancelled"}},
            body={},
        ),
        Exchange(
            "POST",
            "/v1/rmp/benchmarks/plans/plan_1/cancel",
            {"view": {"plan": {**plan, "status": "cancelled"}}},
            body={},
        ),
    ]
    with http_gateway(exchanges) as base_url:
        benchmarks = Client(
            api_key="integration-key", base_url=base_url + "/", timeout=2
        ).benchmarks
        assert (
            benchmarks.create_plan("session_1", selections, budget={"max_total_trials": 2}) == plan
        )
        assert benchmarks.update_plan("plan_1", selections, budget={})["revision"] == 2
        assert benchmarks.preflight("plan_1")["status"] == "preflight_passed"
        assert benchmarks.launch("plan_1")["already_launched"] is False
        assert benchmarks.launch("plan_1")["already_launched"] is True
        assert benchmarks.compare("plan_1", "baseline/1?&") == {"paired_tasks": 1}
        assert benchmarks.get_run("run_1", after=7, limit=3)["next_event_cursor"] == 8
        assert benchmarks.cancel_run("run_1")["status"] == "cancelled"
        assert benchmarks.cancel_plan("plan_1")["plan"]["status"] == "cancelled"
