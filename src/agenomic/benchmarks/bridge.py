"""AgentTargetBridge: run *your* agent against Agenomic benchmark turns.

Agenomic never calls into a customer runtime. Benchmarks execute inside
isolated Agenomic runners; every agent turn is published as a pending turn
that this bridge pulls, hands to your agent, and answers. Your runtime,
prompts, model configuration and memory stay yours; the benchmark's tools
replace your production tools for the duration of a trial and are executed
by the benchmark environment, never by your integrations.

Implement :class:`AgentTargetBridge` (or wrap a callable with
:class:`CallableBridge`) and run :func:`serve_bridge`::

    from agenomic import Client
    from agenomic.benchmarks import AgentTargetBridge, BridgeCapability, TurnRequest, TurnReply, serve_bridge

    class MyBridge(AgentTargetBridge):
        capabilities = [BridgeCapability.MULTI_TURN, BridgeCapability.BENCHMARK_TOOLS]

        def handle_turn(self, turn: TurnRequest) -> TurnReply:
            reply = my_agent.chat(turn.messages, tools=turn.tools, system=turn.instructions)
            return TurnReply(content=reply.text, tool_calls=reply.tool_calls, usage=reply.usage)

    serve_bridge(Client(api_key=..., base_url=...), MyBridge(), agent="agent://acme/support", release_id="rel_1")
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from urllib.parse import quote

import ulid

from agenomic._version import __version__
from agenomic.exceptions import CloudError

log = logging.getLogger("agenomic.benchmarks.bridge")

BRIDGE_HEARTBEAT_SECONDS = 60.0
DEFAULT_WAIT_SECONDS = 20


class BridgeCapability(str, Enum):
    MULTI_TURN = "multi_turn"
    BENCHMARK_TOOLS = "benchmark_tools"
    CODE_EXECUTION = "code_execution"
    MCP_TOOLS = "mcp_tools"
    ENVIRONMENT_RESET = "environment_reset"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRequest:
    turn_id: str
    messages: list[Message]
    tools: list[ToolSpec]
    instructions: Optional[str]
    benchmark_id: str
    run_id: str
    trial_id: str
    task_id: str
    trial_index: int
    turn_index: int
    max_turns: int
    tracking_session_id: Optional[str]
    target: str
    deadline_at: str

    @classmethod
    def from_wire(cls, turn: dict[str, Any]) -> TurnRequest:
        req = turn["request"]
        ctx = req["context"]
        return cls(
            turn_id=turn["turn_id"],
            messages=[
                Message(
                    role=m["role"],
                    content=m.get("content"),
                    tool_calls=[
                        ToolCall(
                            id=c.get("id", ""), name=c["name"], arguments=c.get("arguments") or {}
                        )
                        for c in (m.get("tool_calls") or [])
                    ],
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
                for m in req.get("messages", [])
            ],
            tools=[
                ToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters") or {},
                )
                for t in req.get("tools", [])
            ],
            instructions=ctx.get("instructions"),
            benchmark_id=ctx["benchmark_id"],
            run_id=ctx["run_id"],
            trial_id=ctx["trial_id"],
            task_id=ctx["task_id"],
            trial_index=int(ctx.get("trial_index", 0)),
            turn_index=int(ctx.get("turn_index", 0)),
            max_turns=int(ctx.get("max_turns", 0)),
            tracking_session_id=ctx.get("tracking_session_id"),
            target=ctx.get("target", "customer_agent"),
            deadline_at=turn.get("deadline_at", ""),
        )


@dataclass
class TurnReply:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[dict[str, int]] = None
    stop: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "message": {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
                ],
            },
            "usage": self.usage,
            "stop": self.stop,
        }


class AgentTargetBridge:
    """Contract implemented by the customer's runtime.

    ``handle_turn`` receives the benchmark transcript and the benchmark's tool
    schemas and returns the agent's next message. Tool calls are executed by
    the benchmark environment; the outcome comes back on the next turn.
    """

    capabilities: list[BridgeCapability] = [
        BridgeCapability.MULTI_TURN,
        BridgeCapability.BENCHMARK_TOOLS,
    ]

    def start_trial(self, turn: TurnRequest) -> None:
        """Called before the first turn of a trial. Reset per-trial state here."""

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        raise NotImplementedError

    def end_trial(self, trial_id: str) -> None:
        """Called when a trial's turns stop arriving (best effort)."""


class CallableBridge(AgentTargetBridge):
    def __init__(
        self,
        fn: Callable[[TurnRequest], TurnReply],
        capabilities: Optional[list[BridgeCapability]] = None,
    ) -> None:
        self._fn = fn
        if capabilities is not None:
            self.capabilities = capabilities

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        return self._fn(turn)


class FixtureBridge(AgentTargetBridge):
    """Deterministic bridge for integration tests: never calls a tool."""

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        return TurnReply(
            content=f"[fixture] task {turn.task_id} turn {turn.turn_index}",
            usage={"input_tokens": 0, "output_tokens": 0},
            stop=True,
        )


class BridgeServer:
    """Registers the bridge, long-polls turns and answers them.

    One thread; ``stop()`` ends the loop after the in-flight turn. Replies are
    idempotent on the cloud side, so a retried delivery never double counts.
    """

    def __init__(
        self,
        client: Any,
        bridge: AgentTargetBridge,
        *,
        agent: str,
        release_id: Optional[str] = None,
        bridge_id: Optional[str] = None,
        wait_seconds: int = DEFAULT_WAIT_SECONDS,
    ) -> None:
        if not getattr(client, "is_cloud", False):
            raise CloudError("the benchmark bridge needs a cloud client (base_url)")
        self._client = client
        self._bridge = bridge
        self.agent = agent
        self.release_id = release_id
        self.bridge_id = bridge_id or f"bridge_{ulid.new().str}"
        self.wait_seconds = max(0, min(int(wait_seconds), 25))
        self._stop = threading.Event()
        self._last_heartbeat = 0.0
        self._current_trial: Optional[str] = None
        self.turns_answered = 0

    def register(self) -> dict[str, Any]:
        body = {
            "agent_id": self.agent,
            "release_id": self.release_id,
            "bridge_id": self.bridge_id,
            "capabilities": [c.value for c in self._bridge.capabilities],
            "sdk": f"agenomic-python/{__version__} ({platform.python_implementation()} {platform.python_version()})",
        }
        self._last_heartbeat = time.monotonic()
        registration: dict[str, Any] = self._client._post(
            "/v1/rmp/benchmarks/bridge/register", body
        )
        return registration

    def poll_once(self) -> bool:
        if time.monotonic() - self._last_heartbeat > BRIDGE_HEARTBEAT_SECONDS:
            self.register()
        query = f"agent_id={quote(self.agent, safe='')}&bridge_id={quote(self.bridge_id, safe='')}&wait={self.wait_seconds}"
        if self.release_id:
            query += f"&release_id={quote(self.release_id, safe='')}"
        response = self._client._get(f"/v1/rmp/benchmarks/bridge/turns/next?{query}")
        turn = response.get("turn") if isinstance(response, dict) else None
        if not turn:
            return False
        request = TurnRequest.from_wire(turn)
        if request.trial_id != self._current_trial:
            if self._current_trial is not None:
                self._bridge.end_trial(self._current_trial)
            self._current_trial = request.trial_id
            self._bridge.start_trial(request)
        try:
            reply = self._bridge.handle_turn(request)
        except Exception as exc:
            log.exception("bridge handler failed on turn %s", request.turn_id)
            reply = TurnReply(content=f"[bridge error] {type(exc).__name__}: {exc}", stop=True)
        self._client._post(
            f"/v1/rmp/benchmarks/bridge/turns/{quote(request.turn_id, safe='')}/reply",
            {"reply": reply.to_wire()},
        )
        self.turns_answered += 1
        return True

    def serve(
        self, *, max_turns: Optional[int] = None, idle_timeout: Optional[float] = None
    ) -> int:
        self.register()
        idle_since = time.monotonic()
        while not self._stop.is_set():
            handled = self.poll_once()
            now = time.monotonic()
            if handled:
                idle_since = now
                if max_turns is not None and self.turns_answered >= max_turns:
                    break
            elif idle_timeout is not None and now - idle_since >= idle_timeout:
                break
        if self._current_trial is not None:
            self._bridge.end_trial(self._current_trial)
        return self.turns_answered

    def stop(self) -> None:
        self._stop.set()


def serve_bridge(
    client: Any,
    bridge: AgentTargetBridge,
    *,
    agent: str,
    release_id: Optional[str] = None,
    bridge_id: Optional[str] = None,
    max_turns: Optional[int] = None,
    idle_timeout: Optional[float] = None,
) -> int:
    """Serve benchmark turns until stopped; returns the number of turns answered."""
    server = BridgeServer(client, bridge, agent=agent, release_id=release_id, bridge_id=bridge_id)
    return server.serve(max_turns=max_turns, idle_timeout=idle_timeout)
