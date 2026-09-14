"""AgentTargetBridge: run *your* agent against Agenomic benchmark turns.

Agenomic never calls into a customer runtime. Benchmarks execute inside
isolated Agenomic runners; every agent turn is published as a pending turn
that this bridge pulls, hands to your agent, and answers. Your runtime,
prompts, model configuration and memory stay yours; the benchmark's tools
replace your production tools for the duration of a trial and are executed
by the benchmark environment, never by your integrations.

Implement :class:`AgentTargetBridge` (or wrap a callable with
:class:`CallableBridge`) and run :func:`serve_bridge`, or
:func:`aserve_bridge` inside an asyncio runtime::

    from agenomic import Client
    from agenomic.benchmarks import AgentTargetBridge, BridgeCapability, TurnRequest, TurnReply, serve_bridge

    class MyBridge(AgentTargetBridge):
        capabilities = [BridgeCapability.MULTI_TURN, BridgeCapability.BENCHMARK_TOOLS]

        def handle_turn(self, turn: TurnRequest) -> TurnReply:
            reply = my_agent.chat(turn.messages, tools=turn.tools, system=turn.instructions)
            return TurnReply(content=reply.text, tool_calls=reply.tool_calls, usage=reply.usage)

    serve_bridge(Client(api_key=..., base_url=...), MyBridge(), agent="agent://acme/support", release_id="rel_1")

Every reply goes through a :class:`~agenomic.redaction.RedactionEngine`
before it is posted; the default rules mask credential-looking keys inside
tool arguments. A handler exception is reported to the benchmark as a generic
bridge error without the exception text.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import platform
import threading
import time
from collections.abc import Awaitable, Mapping
from enum import Enum
from typing import Any, Callable, Optional, Union
from urllib.parse import quote

import httpx
import ulid
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agenomic._version import __version__
from agenomic.exceptions import CloudError
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule

log = logging.getLogger("agenomic.benchmarks.bridge")

BRIDGE_HEARTBEAT_SECONDS = 60.0
DEFAULT_WAIT_SECONDS = 20

#: Credential-looking keys masked in tool arguments before a reply is exported.
DEFAULT_BRIDGE_REDACTION_RULES: list[RedactionRule] = [
    RedactionRule(path=f"message.tool_calls.*.arguments.**.{key}", mode=RedactionMode.MASK)
    for key in (
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "private_key",
    )
]


class BridgeCapability(str, Enum):
    """What the served agent supports; the catalogue derives compatibility from it.

    Example:
        >>> BridgeCapability.MULTI_TURN.value
        'multi_turn'
    """

    MULTI_TURN = "multi_turn"
    BENCHMARK_TOOLS = "benchmark_tools"
    CODE_EXECUTION = "code_execution"
    MCP_TOOLS = "mcp_tools"
    ENVIRONMENT_RESET = "environment_reset"


class ToolCall(BaseModel):
    """One tool invocation requested by the agent.

    Example:
        >>> ToolCall(id="c1", name="send_email", arguments={"to": "a@b.c"}).name
        'send_email'
    """

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class Message(BaseModel):
    """One transcript message as relayed by the benchmark.

    Example:
        >>> Message(role="user", content="hello").role
        'user'
    """

    model_config = ConfigDict(extra="ignore")

    role: str
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolSpec(BaseModel):
    """A benchmark tool schema offered to the agent for this turn.

    Example:
        >>> ToolSpec(name="send_email", parameters={"type": "object"}).name
        'send_email'
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class _WireContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    benchmark_id: str
    run_id: str
    trial_id: str
    task_id: str
    trial_index: int = 0
    turn_index: int = 0
    max_turns: int = 0
    instructions: Optional[str] = None
    tracking_session_id: Optional[str] = None
    target: str = "customer_agent"


class _WireRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    context: _WireContext


class _WireTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turn_id: str
    deadline_at: str = ""
    request: _WireRequest


class TurnRequest(BaseModel):
    """Everything the agent needs to produce its next message.

    Example:
        >>> turn = TurnRequest.from_wire({
        ...     "turn_id": "bturn_1",
        ...     "request": {
        ...         "messages": [{"role": "user", "content": "do the task"}],
        ...         "tools": [{"name": "send_email"}],
        ...         "context": {"benchmark_id": "agentdojo", "run_id": "brun_1",
        ...                     "trial_id": "btrial_1", "task_id": "user_task_0"},
        ...     },
        ... })
        >>> turn.task_id, turn.tools[0].name
        ('user_task_0', 'send_email')
    """

    model_config = ConfigDict(extra="forbid")

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
    def from_wire(cls, turn: Mapping[str, object]) -> TurnRequest:
        """Validate a relay payload; malformed turns raise ``pydantic.ValidationError``."""
        wire = _WireTurn.model_validate(turn)
        ctx = wire.request.context
        return cls(
            turn_id=wire.turn_id,
            messages=wire.request.messages,
            tools=wire.request.tools,
            instructions=ctx.instructions,
            benchmark_id=ctx.benchmark_id,
            run_id=ctx.run_id,
            trial_id=ctx.trial_id,
            task_id=ctx.task_id,
            trial_index=ctx.trial_index,
            turn_index=ctx.turn_index,
            max_turns=ctx.max_turns,
            tracking_session_id=ctx.tracking_session_id,
            target=ctx.target,
            deadline_at=wire.deadline_at,
        )


class TurnReply(BaseModel):
    """The agent's next message.

    Example:
        >>> TurnReply(content="done", stop=True).to_wire()["stop"]
        True
    """

    model_config = ConfigDict(extra="forbid")

    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Optional[dict[str, int]] = None
    stop: bool = False

    def to_wire(self) -> dict[str, JsonValue]:
        """The relay representation, before redaction."""
        return {
            "message": {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [c.model_dump(mode="json") for c in self.tool_calls],
            },
            "usage": dict(self.usage) if self.usage is not None else None,
            "stop": self.stop,
        }


MaybeAwaitable = Union[TurnReply, Awaitable[TurnReply]]


class AgentTargetBridge:
    """Contract implemented by the customer's runtime.

    ``handle_turn`` receives the benchmark transcript and the benchmark's tool
    schemas and returns the agent's next message, synchronously or as a
    coroutine. Tool calls are executed by the benchmark environment; the
    outcome comes back on the next turn.

    Example:
        >>> class Echo(AgentTargetBridge):
        ...     def handle_turn(self, turn: TurnRequest) -> TurnReply:
        ...         return TurnReply(content=turn.messages[-1].content, stop=True)
        >>> Echo().capabilities[0].value
        'multi_turn'
    """

    capabilities: list[BridgeCapability] = [
        BridgeCapability.MULTI_TURN,
        BridgeCapability.BENCHMARK_TOOLS,
    ]

    def start_trial(self, turn: TurnRequest) -> Union[None, Awaitable[None]]:
        """Called before the first turn of a trial. Reset per-trial state here."""
        return None

    def handle_turn(self, turn: TurnRequest) -> MaybeAwaitable:
        raise NotImplementedError

    def end_trial(self, trial_id: str) -> Union[None, Awaitable[None]]:
        """Called when a trial's turns stop arriving (best effort)."""
        return None


class CallableBridge(AgentTargetBridge):
    """Wrap a plain function or coroutine function as a bridge.

    Example:
        >>> bridge = CallableBridge(lambda turn: TurnReply(content="ok", stop=True))
        >>> bridge.capabilities[1].value
        'benchmark_tools'
    """

    def __init__(
        self,
        fn: Callable[[TurnRequest], MaybeAwaitable],
        capabilities: Optional[list[BridgeCapability]] = None,
    ) -> None:
        self._fn = fn
        if capabilities is not None:
            self.capabilities = capabilities

    def handle_turn(self, turn: TurnRequest) -> MaybeAwaitable:
        return self._fn(turn)


class FixtureBridge(AgentTargetBridge):
    """Deterministic bridge for integration tests: never calls a tool.

    Example:
        >>> FixtureBridge().capabilities[0].value
        'multi_turn'
    """

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        return TurnReply(
            content=f"[fixture] task {turn.task_id} turn {turn.turn_index}",
            usage={"input_tokens": 0, "output_tokens": 0},
            stop=True,
        )


async def _maybe_await(value: Union[None, Awaitable[None]]) -> None:
    if inspect.isawaitable(value):
        await value


class AsyncBridgeServer:
    """Registers the bridge, long-polls turns and answers them (asyncio).

    ``stop()`` ends the loop after the in-flight turn. Replies are idempotent
    on the cloud side, so a retried delivery never double counts. Every reply
    is redacted before it leaves the process.

    Example:
        >>> import asyncio
        >>> from agenomic import Client
        >>> server = AsyncBridgeServer(
        ...     Client(api_key="agm_...", base_url="https://cloud.agenomic.io"),
        ...     FixtureBridge(), agent="agent://acme/support")
        >>> asyncio.run(server.serve(max_turns=1, idle_timeout=0))  # doctest: +SKIP
        0
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
        redaction: Optional[RedactionEngine] = None,
    ) -> None:
        if not getattr(client, "is_cloud", False):
            raise CloudError("the benchmark bridge needs a cloud client (base_url)")
        self._client = client
        self._bridge = bridge
        self.agent = agent
        self.release_id = release_id
        self.bridge_id = bridge_id or f"bridge_{ulid.new().str}"
        self.wait_seconds = max(0, min(int(wait_seconds), 25))
        self.redaction = redaction or RedactionEngine(DEFAULT_BRIDGE_REDACTION_RULES)
        self._stopped = False
        self._last_heartbeat = 0.0
        self._current_trial: Optional[str] = None
        self.turns_answered = 0

    async def _request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        try:
            async with self._client._ahttp() as http:
                response = await http.request(method, path, json=body)
                response.raise_for_status()
                data: dict[str, Any] = response.json() if response.content else {}
                return data
        except httpx.HTTPError as exc:
            raise CloudError(f"{method} {path} failed: {exc}") from exc

    async def register(self) -> dict[str, Any]:
        """Announce the bridge and its capabilities; repeated as a heartbeat."""
        body = {
            "agent_id": self.agent,
            "release_id": self.release_id,
            "bridge_id": self.bridge_id,
            "capabilities": [c.value for c in self._bridge.capabilities],
            "sdk": f"agenomic-python/{__version__} ({platform.python_implementation()} {platform.python_version()})",
        }
        self._last_heartbeat = time.monotonic()
        return await self._request("POST", "/v1/rmp/benchmarks/bridge/register", body)

    async def poll_once(self) -> bool:
        """Fetch at most one pending turn, answer it, and report whether one was handled."""
        if time.monotonic() - self._last_heartbeat > BRIDGE_HEARTBEAT_SECONDS:
            await self.register()
        query = f"agent_id={quote(self.agent, safe='')}&bridge_id={quote(self.bridge_id, safe='')}&wait={self.wait_seconds}"
        if self.release_id:
            query += f"&release_id={quote(self.release_id, safe='')}"
        response = await self._request("GET", f"/v1/rmp/benchmarks/bridge/turns/next?{query}")
        turn = response.get("turn")
        if not isinstance(turn, Mapping):
            return False
        request = TurnRequest.from_wire(turn)
        if request.trial_id != self._current_trial:
            if self._current_trial is not None:
                await _maybe_await(self._bridge.end_trial(self._current_trial))
            self._current_trial = request.trial_id
            await _maybe_await(self._bridge.start_trial(request))
        try:
            produced = self._bridge.handle_turn(request)
            reply = await produced if inspect.isawaitable(produced) else produced
        except Exception as exc:
            log.exception("bridge handler failed on turn %s", request.turn_id)
            reply = TurnReply(
                content=f"[bridge error] handler raised {type(exc).__name__}", stop=True
            )
        wire = self.redaction.apply(reply.to_wire())
        await self._request(
            "POST",
            f"/v1/rmp/benchmarks/bridge/turns/{quote(request.turn_id, safe='')}/reply",
            {"reply": wire},
        )
        self.turns_answered += 1
        return True

    async def serve(
        self, *, max_turns: Optional[int] = None, idle_timeout: Optional[float] = None
    ) -> int:
        """Serve turns until stopped, ``max_turns`` answered or idle for ``idle_timeout`` seconds."""
        await self.register()
        idle_since = time.monotonic()
        try:
            while not self._stopped:
                handled = await self.poll_once()
                now = time.monotonic()
                if handled:
                    idle_since = now
                    if max_turns is not None and self.turns_answered >= max_turns:
                        break
                elif idle_timeout is not None and now - idle_since >= idle_timeout:
                    break
        finally:
            trial = self._current_trial
            self._current_trial = None
            if trial is not None:
                await _maybe_await(self._bridge.end_trial(trial))
        return self.turns_answered

    def stop(self) -> None:
        """End the loop after the in-flight turn."""
        self._stopped = True


class BridgeServer:
    """Synchronous entry point over :class:`AsyncBridgeServer`.

    Each call runs the async server to completion with ``asyncio.run``; use
    it from plain scripts and the CLI, and :class:`AsyncBridgeServer` from an
    asyncio runtime.

    Example:
        >>> from agenomic import Client
        >>> server = BridgeServer(
        ...     Client(api_key="agm_...", base_url="https://cloud.agenomic.io"),
        ...     FixtureBridge(), agent="agent://acme/support")
        >>> server.serve(max_turns=1, idle_timeout=0)  # doctest: +SKIP
        0
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
        redaction: Optional[RedactionEngine] = None,
    ) -> None:
        self._inner = AsyncBridgeServer(
            client,
            bridge,
            agent=agent,
            release_id=release_id,
            bridge_id=bridge_id,
            wait_seconds=wait_seconds,
            redaction=redaction,
        )
        self._stop = threading.Event()

    @property
    def agent(self) -> str:
        return self._inner.agent

    @property
    def release_id(self) -> Optional[str]:
        return self._inner.release_id

    @property
    def bridge_id(self) -> str:
        return self._inner.bridge_id

    @property
    def wait_seconds(self) -> int:
        return self._inner.wait_seconds

    @property
    def turns_answered(self) -> int:
        return self._inner.turns_answered

    def register(self) -> dict[str, Any]:
        """Announce the bridge and its capabilities."""
        return asyncio.run(self._inner.register())

    def poll_once(self) -> bool:
        """Fetch at most one pending turn and answer it."""
        return asyncio.run(self._inner.poll_once())

    def serve(
        self, *, max_turns: Optional[int] = None, idle_timeout: Optional[float] = None
    ) -> int:
        """Serve turns until stopped; returns the number of turns answered."""
        return asyncio.run(self._inner.serve(max_turns=max_turns, idle_timeout=idle_timeout))

    def stop(self) -> None:
        """End the loop after the in-flight turn."""
        self._stop.set()
        self._inner.stop()


async def aserve_bridge(
    client: Any,
    bridge: AgentTargetBridge,
    *,
    agent: str,
    release_id: Optional[str] = None,
    bridge_id: Optional[str] = None,
    max_turns: Optional[int] = None,
    idle_timeout: Optional[float] = None,
    redaction: Optional[RedactionEngine] = None,
) -> int:
    """Serve benchmark turns from an asyncio runtime; returns the number of turns answered.

    Example:
        >>> import asyncio
        >>> from agenomic import Client
        >>> client = Client(api_key="agm_...", base_url="https://cloud.agenomic.io")
        >>> asyncio.run(aserve_bridge(client, FixtureBridge(), agent="agent://acme/support",
        ...                           idle_timeout=0))  # doctest: +SKIP
        0
    """
    server = AsyncBridgeServer(
        client, bridge, agent=agent, release_id=release_id, bridge_id=bridge_id, redaction=redaction
    )
    return await server.serve(max_turns=max_turns, idle_timeout=idle_timeout)


def serve_bridge(
    client: Any,
    bridge: AgentTargetBridge,
    *,
    agent: str,
    release_id: Optional[str] = None,
    bridge_id: Optional[str] = None,
    max_turns: Optional[int] = None,
    idle_timeout: Optional[float] = None,
    redaction: Optional[RedactionEngine] = None,
) -> int:
    """Serve benchmark turns until stopped; returns the number of turns answered.

    Top-level synchronous wrapper around :func:`aserve_bridge`; do not call it
    from inside a running event loop.

    Example:
        >>> from agenomic import Client
        >>> client = Client(api_key="agm_...", base_url="https://cloud.agenomic.io")
        >>> serve_bridge(client, FixtureBridge(), agent="agent://acme/support",
        ...              release_id="rel_1", idle_timeout=0)  # doctest: +SKIP
        0
    """
    return asyncio.run(
        aserve_bridge(
            client,
            bridge,
            agent=agent,
            release_id=release_id,
            bridge_id=bridge_id,
            max_turns=max_turns,
            idle_timeout=idle_timeout,
            redaction=redaction,
        )
    )
