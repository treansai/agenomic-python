from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import metadata as package_metadata
from typing import Any, Literal, Optional, Union, cast

import ulid
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import ValidationError

from agenomic._transport import aapi_request, aclose_pool, segment
from agenomic._version import __version__
from agenomic.canonical.hashing import canonical_json
from agenomic.exceptions import ApiError
from agenomic.experiments.context import TrialContext, TrialState
from agenomic.experiments.errors import (
    ErrorClass,
    ErrorHook,
    LeaseLost,
    RecordedFixtureAmbiguous,
    RecordedFixtureMiss,
    RunnerConfigurationError,
    SecretResolutionError,
    TrialBudgetExceeded,
    classify,
    error_code,
)
from agenomic.experiments.isolation import (
    NamespacedStore,
    check_graph,
    isolation_record,
    jsonable,
    revive_state,
    seed_store,
    serialize_state,
)
from agenomic.experiments.models import (
    ASSIGNMENT_SCHEMA,
    CASE_SCHEMA,
    HELLO_SCHEMA,
    RESULT_SCHEMA,
    VIEW_SCHEMA,
    ExperimentCase,
    RunnerEvaluatorView,
    RunnerView,
    TrialAssignment,
    ViewLimits,
)
from agenomic.experiments.secrets import (
    LOG_REDACTOR,
    SecretResolver,
    SecretValues,
    redact_message,
    redact_outbound,
)
from agenomic.experiments.tools import HttpToolProxy, ToolProxy
from agenomic.integrations.langchain_prompts import to_langchain_messages
from agenomic.integrations.langgraph_binding import ManagedGraph, bind_langgraph
from agenomic.prompts.bundle import PromptBundle
from agenomic.prompts.local import LocalPromptEngine
from agenomic.prompts.models import ExecutionBinding, ManagedPromptVersion, PromptVersionRecord
from agenomic.prompts.resources import binding_bundle

__all__ = [
    "CallableEntryPoint",
    "ExperimentRunner",
    "GraphNodeEntryPoint",
    "GraphTarget",
    "RunnerEvaluator",
    "SnapshotRefusedError",
    "TrialRun",
    "local_assignment",
    "snapshot_case",
]

log = logging.getLogger("agenomic.experiments")
log.addFilter(LOG_REDACTOR)

SDK = f"agenomic-python/{__version__}"
HELLO_REFRESH_SECONDS = 60.0
CLAIM_WAIT_MAX_SECONDS = 25
REPORT_ROUNDS = (1.0, 2.0, 4.0)
TOKEN_ENV = "AGENOMIC_RUNNER_TOKEN"
ENDPOINT_ENV = "AGENOMIC_ENDPOINT"
_TOKEN = re.compile(r"agr_[0-9a-f]{64}", re.ASCII)
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.ASCII)
_REPORT_NAMESPACE = uuid.UUID("6c0f5b2e-3d9a-4f7e-8a41-2b7c9d0e1f35")
_NO_RETRY_REASONS = frozenset({"cancelled", "stopped", "shutdown"})

_asleep = asyncio.sleep


@dataclass(frozen=True)
class GraphNodeEntryPoint:
    node_path: str
    seed_as_node: Optional[str] = "__start__"

    @property
    def kind(self) -> Literal["graph_node"]:
        return "graph_node"


@dataclass(frozen=True)
class CallableEntryPoint:
    fn: Callable[[dict[str, Any], Any], Any]

    @property
    def kind(self) -> Literal["callable"]:
        return "callable"


EntryPoint = Union[GraphNodeEntryPoint, CallableEntryPoint]


@dataclass(frozen=True)
class RunnerEvaluator:
    fn: Callable[[ExperimentCase, Mapping[str, Any]], Any]
    version: int
    value_kind: Literal["binary", "continuous"] = "binary"
    code_digest: Optional[str] = None

    def digest(self) -> str:
        if self.code_digest is not None:
            return self.code_digest
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            source = getattr(self.fn, "__qualname__", repr(self.fn))
        return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphTarget:
    factory: Callable[[TrialContext], Any]
    runtime_digest: str
    input_adapter: Optional[Callable[[ExperimentCase], Any]] = None
    output_adapter: Optional[Callable[[Any], Any]] = None
    turn_adapter: Optional[Callable[[str], Any]] = None
    entry_points: Mapping[str, EntryPoint] = field(default_factory=dict)
    children: Mapping[str, str] = field(default_factory=dict)
    evaluators: Mapping[str, RunnerEvaluator] = field(default_factory=dict)
    live_tools: bool = False
    checkpointer_factory: Optional[Callable[[str], Any]] = None
    keep_checkpoints: bool = False
    store: Any = None
    on_trial_end: Optional[Callable[[TrialContext], Any]] = None

    def tool_modes(self) -> list[str]:
        modes = ["none", "mock", "recorded"]
        return [*modes, "live"] if self.live_tools else modes

    def levels(self) -> list[str]:
        return ["agent", "node"] if self.entry_points else ["agent"]


@dataclass(frozen=True)
class TrialRun:
    kind: Literal["result", "failure", "release", "lost"]
    result: Optional[dict[str, Any]] = None
    error_class: Optional[ErrorClass] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    reason: Optional[str] = None


class SnapshotRefusedError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.message = message


class CallableInterruptError(Exception):
    def __init__(self) -> None:
        super().__init__("interrupt() is not supported inside a callable entry point")


class _RunnerHttp:
    def __init__(self, base_url: str, token: str, transport: Any, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._transport = transport
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"_RunnerHttp(base_url={self.base_url!r})"

    def _http_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "headers": {"User-Agent": SDK, "Authorization": f"Bearer {self._token}"},
            "timeout": self._timeout,
        }


class _TrialCallbacks(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    def __init__(self, state: TrialState, limits: ViewLimits) -> None:
        self._state = state
        self._limits = limits
        self._open: dict[Any, dict[str, Any]] = {}

    def _admit(self, run_id: Any, kwargs: Mapping[str, Any]) -> None:
        state = self._state
        error: Optional[TrialBudgetExceeded] = None
        calls = self._limits.max_model_calls_per_trial
        tokens = self._limits.max_tokens_per_trial
        with state.lock:
            if calls is not None and state.model_call_count >= calls:
                error = TrialBudgetExceeded("model_calls", calls)
            elif tokens is not None and state.tokens_used >= tokens:
                error = TrialBudgetExceeded("tokens", tokens)
            else:
                state.model_call_count += 1
                self._open[run_id] = {
                    "started": time.monotonic(),
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "params": dict(kwargs.get("invocation_params") or {}),
                }
        if error is not None:
            state.mark_terminal(error)
            raise error

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        self._admit(run_id, kwargs)

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._admit(run_id, kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        self._close(run_id, _usage(response))

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._close(run_id, None)

    def _close(self, run_id: Any, usage: Optional[tuple[int, int]]) -> None:
        state = self._state
        with state.lock:
            entry = self._open.pop(run_id, None)
            if entry is None:
                return
            state.model_calls.append(_model_call(entry, usage))
            if usage is not None:
                state.tokens_used += usage[0] + usage[1]


def _usage(response: LLMResult) -> Optional[tuple[int, int]]:
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration):
                metadata = getattr(generation.message, "usage_metadata", None)
                if metadata:
                    return int(metadata.get("input_tokens", 0)), int(
                        metadata.get("output_tokens", 0)
                    )
    token_usage = (response.llm_output or {}).get("token_usage")
    if isinstance(token_usage, Mapping) and "prompt_tokens" in token_usage:
        return int(token_usage.get("prompt_tokens") or 0), int(
            token_usage.get("completion_tokens") or 0
        )
    return None


def _first(metadata: Mapping[str, Any], key: str) -> Optional[str]:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value.split(",")[0]


def _model_call(entry: Mapping[str, Any], usage: Optional[tuple[int, int]]) -> dict[str, Any]:
    metadata = entry["metadata"]
    params = entry["params"]
    model = metadata.get("ls_model_name") or params.get("model") or params.get("model_name")
    return {
        "slot_path": _first(metadata, "agenomic_prompt_slots"),
        "prompt_ref": _first(metadata, "agenomic_prompt_refs"),
        "content_digest": _first(metadata, "agenomic_prompt_content_digests"),
        "rendered_hash": metadata.get("agenomic_rendered_hash"),
        "protect_overlay_digest": None,
        "provider": metadata.get("ls_provider"),
        "model": model if isinstance(model, str) else None,
        "input_tokens": None if usage is None else usage[0],
        "output_tokens": None if usage is None else usage[1],
        "usage_source": "not_reported" if usage is None else "provider_reported",
        "latency_ms": int((time.monotonic() - entry["started"]) * 1000),
        "seed_applied": False,
    }


def _installed(name: str) -> Optional[str]:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def _seconds_until(stamp: str) -> Optional[float]:
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())


def _default_input(case: ExperimentCase) -> Any:
    if isinstance(case.input, Mapping):
        return {
            key: value
            for key, value in case.input.items()
            if key not in ("context", "store_seed", "provenance")
        }
    return case.input


def _default_final(values: Any) -> Any:
    if isinstance(values, Mapping):
        messages = values.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[-1], BaseMessage):
            return messages[-1].content
    return values


def _metric(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None or isinstance(value, (int, float)):
        return value
    raise TypeError("a runner evaluator returns a number, a boolean or None")


def _manifest_check(model_calls: Sequence[Mapping[str, Any]], bundle: PromptBundle) -> None:
    prompts = bundle.document["prompts"]
    for call in model_calls:
        ref = call.get("prompt_ref")
        digest = call.get("content_digest")
        if ref is None and digest is None:
            continue
        entry = prompts.get(ref) if isinstance(ref, str) else None
        if entry is None or entry.get("content_digest") != digest:
            raise RunnerConfigurationError(
                "prompt_outside_manifest",
                "a model call used a prompt outside the arm's manifest closure",
                prompt_ref=ref,
            )


@dataclass
class _Control:
    reason: Optional[str] = None


@dataclass
class _ServeState:
    max_trials: Optional[int]
    idle_timeout: Optional[float]
    claimed: int = 0
    completed: int = 0
    last_activity: float = field(default_factory=time.monotonic)

    def reserve(self) -> bool:
        if self.max_trials is not None and self.claimed >= self.max_trials:
            return False
        self.claimed += 1
        return True

    def idle(self) -> bool:
        return (
            self.idle_timeout is not None
            and time.monotonic() - self.last_activity >= self.idle_timeout
        )

    def wait_seconds(self) -> int:
        if self.idle_timeout is None:
            return CLAIM_WAIT_MAX_SECONDS
        remaining = self.idle_timeout - (time.monotonic() - self.last_activity)
        return max(0, min(CLAIM_WAIT_MAX_SECONDS, math.ceil(remaining)))


class ExperimentRunner:
    def __init__(
        self,
        *,
        targets: Mapping[str, GraphTarget],
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        secrets: Optional[SecretResolver] = None,
        max_concurrency: int = 4,
        classify_error: Optional[ErrorHook] = None,
        judge_model: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        workspace_id: Optional[str] = None,
        transport: Any = None,
        timeout: float = 30.0,
    ) -> None:
        if not targets:
            raise ValueError("targets maps at least one agent id to a GraphTarget")
        for agent_id, target in targets.items():
            if not isinstance(agent_id, str) or _UUID.fullmatch(agent_id) is None:
                raise ValueError("targets keys are lowercase agent uuids")
            if not isinstance(target, GraphTarget):
                raise TypeError("targets values are GraphTarget objects")
            if not isinstance(target.runtime_digest, str) or not target.runtime_digest:
                raise ValueError("GraphTarget.runtime_digest is the declared bundle hash")
        if token is not None and _TOKEN.fullmatch(token) is None:
            raise ValueError("the runner token is agr_ followed by 64 lowercase hex characters")
        if workspace_id is not None and _UUID.fullmatch(workspace_id) is None:
            raise ValueError("workspace_id must be a lowercase uuid")
        self.targets = dict(targets)
        self._token = token
        self.base_url = base_url
        self.secrets = secrets
        self.max_concurrency = self._concurrency(max_concurrency)
        self._classify_error = classify_error
        self._judge_model = judge_model
        self._workspace_id = workspace_id
        self._workspace_lock = threading.Lock()
        self._transport = transport
        self._timeout = timeout
        self._stopping = threading.Event()
        self._last_hello = 0.0
        self._outbox: dict[uuid.UUID, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"ExperimentRunner(agents={sorted(self.targets)!r}, base_url={self.base_url!r})"

    @staticmethod
    def _concurrency(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
            raise ValueError("max_concurrency must be between 1 and 64")
        return value

    def hello_document(self) -> dict[str, Any]:
        agents = []
        for agent_id in sorted(self.targets):
            target = self.targets[agent_id]
            agents.append(
                {
                    "agent_id": agent_id,
                    "runtime_digests": [target.runtime_digest],
                    "levels": target.levels(),
                    "tool_modes": target.tool_modes(),
                    "entry_points": [
                        {
                            "name": name,
                            "kind": entry.kind,
                            "node_path": entry.node_path
                            if isinstance(entry, GraphNodeEntryPoint)
                            else name,
                            "seed_as_node": entry.seed_as_node
                            if isinstance(entry, GraphNodeEntryPoint)
                            else None,
                        }
                        for name, entry in sorted(target.entry_points.items())
                    ],
                    "custom_evaluators": [
                        {
                            "name": name,
                            "version": evaluator.version,
                            "code_digest": evaluator.digest(),
                        }
                        for name, evaluator in sorted(target.evaluators.items())
                    ],
                    "live_tool_egress": "runner_managed",
                }
            )
        return {
            "schema": HELLO_SCHEMA,
            "sdk": SDK,
            "runtime": {
                "python": platform.python_version(),
                "langgraph": _installed("langgraph"),
                "langgraph_checkpoint": _installed("langgraph-checkpoint"),
                "langchain_core": _installed("langchain-core"),
            },
            "max_concurrency": self.max_concurrency,
            "agents": agents,
            "secret_ref_names": sorted(self.secrets.names()) if self.secrets is not None else [],
        }

    def stop(self) -> None:
        self._stopping.set()

    def _literals(self, secrets: Optional[SecretValues] = None) -> list[str]:
        literals = [] if secrets is None else secrets.literals()
        if self._token:
            literals.append(self._token)
        return literals

    def _connect(self) -> _RunnerHttp:
        token = self._token or os.environ.get(TOKEN_ENV)
        if not token or _TOKEN.fullmatch(token) is None:
            raise ValueError(f"set {TOKEN_ENV} to the agr_ runner token")
        base_url = self.base_url or os.environ.get(ENDPOINT_ENV)
        if not base_url:
            raise ValueError(f"pass base_url or set {ENDPOINT_ENV}")
        self._token = token
        timeout = max(self._timeout, CLAIM_WAIT_MAX_SECONDS + 10.0)
        return _RunnerHttp(base_url, token, self._transport, timeout)

    def run_trial(
        self,
        assignment: Union[TrialAssignment, Mapping[str, Any]],
        *,
        proxy: Optional[ToolProxy] = None,
    ) -> TrialRun:
        return asyncio.run(self.arun_trial(assignment, proxy=proxy))

    async def arun_trial(
        self,
        assignment: Union[TrialAssignment, Mapping[str, Any]],
        *,
        proxy: Optional[ToolProxy] = None,
    ) -> TrialRun:
        parsed = (
            assignment
            if isinstance(assignment, TrialAssignment)
            else TrialAssignment.model_validate(assignment)
        )
        state = TrialState(proxy=proxy, lease_token=parsed.lease_token, literals=self._literals())
        with LOG_REDACTOR.tracking(lambda: state.literals):
            return await self._aexecute(parsed, state)

    def _pin_workspace(self, workspace_id: str) -> None:
        with self._workspace_lock:
            if self._workspace_id is None:
                self._workspace_id = workspace_id
            elif self._workspace_id != workspace_id:
                raise RunnerConfigurationError(
                    "workspace_mismatch", "the assignment belongs to another workspace"
                )

    def _prepare(
        self, assignment: TrialAssignment
    ) -> tuple[str, GraphTarget, ExecutionBinding, PromptBundle]:
        view = assignment.view
        if (view.trial_id, view.attempt) != (assignment.trial_id, assignment.attempt):
            raise RunnerConfigurationError(
                "assignment_invalid", "the view names another trial or attempt"
            )
        try:
            binding = ExecutionBinding.model_validate(view.binding)
        except ValidationError as error:
            raise RunnerConfigurationError(
                "assignment_invalid", "the view carries no valid execution binding"
            ) from error
        target = self.targets.get(binding.agent_id)
        if target is None:
            raise RunnerConfigurationError(
                "agent_not_served", "this runner does not serve the agent of the trial"
            )
        if target.runtime_digest != view.arm.runtime_digest:
            raise RunnerConfigurationError(
                "runtime_digest_mismatch", "the arm needs another runtime than this runner serves"
            )
        if view.tools.mode not in target.tool_modes():
            raise RunnerConfigurationError(
                "tool_mode_unavailable",
                "this runner does not serve the trial's tool mode",
                tool_mode=view.tools.mode,
            )
        expected_key = f"exp:{view.experiment_id}:{view.trial_id}:a{view.attempt}"
        experiment = binding.experiment
        if (
            binding.thread_key != expected_key
            or binding.scope != "thread"
            or binding.release_id != view.arm.release_id
            or not isinstance(experiment, Mapping)
            or experiment.get("arm_key") != view.arm.arm_key
        ):
            raise RunnerConfigurationError(
                "binding_mismatch", "the trial binding does not match the assignment"
            )
        self._pin_workspace(binding.workspace_id)
        manifest = view.arm.prompt_manifest_digest
        if (
            binding.prompt_manifest_digest != manifest
            or view.prompts.get("prompt_manifest_digest") != manifest
        ):
            raise RunnerConfigurationError(
                "artifact_digest_mismatch", "the arm, binding and prompts name different manifests"
            )
        try:
            bundle = binding_bundle(binding, view.prompts)
        except ApiError as error:
            raise RunnerConfigurationError("artifact_digest_mismatch", error.message) from error
        return binding.agent_id, target, binding, bundle

    def _resolve_secrets(self, refs: Sequence[str]) -> SecretValues:
        if not refs:
            return SecretValues()
        if self.secrets is None:
            raise SecretResolutionError("the trial needs secrets and the runner has no resolver")
        return SecretValues({ref: self.secrets.resolve(ref) for ref in refs})

    def _context(
        self,
        assignment: TrialAssignment,
        agent_id: str,
        target: GraphTarget,
        secrets: SecretValues,
        state: TrialState,
    ) -> tuple[TrialContext, str, str]:
        view = assignment.view
        thread_key = str(view.binding["thread_key"])
        if target.checkpointer_factory is not None:
            checkpointer, saver_kind = target.checkpointer_factory(thread_key), "per_trial_factory"
        else:
            checkpointer, saver_kind = InMemorySaver(), "per_trial_in_memory"
        store = (
            NamespacedStore(target.store, view.isolation.store_namespace)
            if target.store is not None
            else InMemoryStore()
        )
        context = TrialContext(
            view=view,
            agent_id=agent_id,
            checkpointer=checkpointer,
            store=store,
            secrets=secrets,
            state=state,
            live_allowed="live" in target.tool_modes(),
        )
        if isinstance(view.case.input, Mapping):
            seed_store(store, context.store_namespace, view.case.input.get("store_seed"))
        return context, saver_kind, "per_trial_namespace"

    def _config(self, context: TrialContext, callbacks: _TrialCallbacks) -> dict[str, Any]:
        return {
            "configurable": {**context.case.context, "thread_id": context.thread_key},
            "callbacks": [callbacks],
        }

    async def _invoke(
        self,
        managed: ManagedGraph,
        payload: Any,
        config: Mapping[str, Any],
        turns: list[dict[str, Any]],
        index: int,
        **kwargs: Any,
    ) -> Any:
        started = time.monotonic()
        await managed.ainvoke(payload, cast(Any, config), **kwargs)
        snapshot = await managed.aget_state(
            cast(Any, {"configurable": dict(config["configurable"])})
        )
        turns.append(
            {
                "index": index,
                "interrupted": bool(snapshot.interrupts),
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return snapshot

    async def _run_agent(
        self,
        managed: ManagedGraph,
        context: TrialContext,
        target: GraphTarget,
        config: Mapping[str, Any],
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        case = context.case
        first = target.input_adapter(case) if target.input_adapter else _default_input(case)
        snapshot = await self._invoke(managed, first, config, turns, 0)
        for index, turn in enumerate(case.turns, start=1):
            if "resume" in turn:
                payload: Any = Command(resume=turn["resume"])
            elif target.turn_adapter is not None:
                payload = target.turn_adapter(str(turn.get("user", "")))
            else:
                payload = {"messages": [{"role": "user", "content": str(turn.get("user", ""))}]}
            snapshot = await self._invoke(managed, payload, config, turns, index)
        values = snapshot.values
        final = target.output_adapter(values) if target.output_adapter else _default_final(values)
        messages = values.get("messages") if isinstance(values, Mapping) else None
        return {
            "final": jsonable(final),
            "messages": jsonable(messages) if isinstance(messages, list) else [],
        }

    @staticmethod
    def _entry_point(view: RunnerView, target: GraphTarget) -> EntryPoint:
        entry = view.entry_point
        declared = target.entry_points.get(entry.name) if entry is not None else None
        if entry is None or declared is None or declared.kind != entry.kind:
            raise RunnerConfigurationError(
                "entry_point_unavailable", "this runner does not declare the trial's entry point"
            )
        if isinstance(declared, GraphNodeEntryPoint) and (
            declared.node_path != entry.node_path or declared.seed_as_node != entry.seed_as_node
        ):
            raise RunnerConfigurationError(
                "entry_point_unavailable", "the declared entry point differs from the trial's"
            )
        return declared

    async def _run_callable(
        self,
        declared: CallableEntryPoint,
        context: TrialContext,
        binding: ExecutionBinding,
        bundle: PromptBundle,
        target: GraphTarget,
        config: Mapping[str, Any],
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        case = context.case
        if case.initial_state is None or case.turns:
            raise RunnerConfigurationError(
                "initial_state_invalid", "a callable case has initial_state and no turns"
            )
        captured: dict[str, Any] = {}

        def entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            captured["update"] = declared.fn(dict(state), config)
            return {}

        builder: Any = StateGraph(cast(Any, dict))
        builder.add_node("entry", entry)
        builder.add_edge(START, "entry")
        managed = bind_langgraph(
            builder.compile(),
            agent_id=binding.agent_id,
            binding=binding,
            resolution=bundle,
            children=dict(target.children) or None,
        )
        started = time.monotonic()
        result = await managed.ainvoke(revive_state(case.initial_state), cast(Any, config))
        turns.append(
            {
                "index": 0,
                "interrupted": False,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        )
        if isinstance(result, Mapping) and "__interrupt__" in result:
            raise CallableInterruptError()
        return {"state_update": serialize_state(captured.get("update")), "extra_nodes_executed": []}

    async def _run_node(
        self,
        managed: ManagedGraph,
        context: TrialContext,
        entry_point: GraphNodeEntryPoint,
        config: Mapping[str, Any],
        turns: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        entry = entry_point
        case = context.case
        if case.initial_state is None:
            raise RunnerConfigurationError("initial_state_invalid", "a node case has initial_state")
        plain = {"configurable": dict(config["configurable"])}
        await managed.aupdate_state(
            cast(Any, config), revive_state(case.initial_state), as_node=entry.seed_as_node
        )
        seeded = await managed.aget_state(cast(Any, plain))
        fork = "none"
        if case.is_snapshot:
            fork = "state_values_only"
            if serialize_state(seeded.values) != case.initial_state:
                raise RunnerConfigurationError(
                    "fork_unsupported", "the seeded state differs from the snapshot values"
                )
        if tuple(seeded.next) != (entry.node_path,):
            raise RunnerConfigurationError(
                "entry_point_not_next",
                "after seeding, the entry point is not the only next node",
                next=list(seeded.next),
            )
        before = serialize_state(seeded.values)
        seeded_id = seeded.config["configurable"]["checkpoint_id"]
        snapshot = await self._invoke(
            managed, None, config, turns, 0, interrupt_after=[entry.node_path]
        )
        for index, turn in enumerate(case.turns, start=1):
            if "resume" not in turn:
                raise RunnerConfigurationError(
                    "node_turn_unsupported", "a node case only takes resume turns"
                )
            snapshot = await self._invoke(
                managed,
                Command(resume=turn["resume"]),
                config,
                turns,
                index,
                interrupt_after=[entry.node_path],
            )
        newer = []
        async for item in managed.aget_state_history(cast(Any, plain)):
            if item.config["configurable"]["checkpoint_id"] == seeded_id:
                break
            newer.append(item)
        extra = [task.name for item in newer[1:] for task in item.tasks]
        if extra:
            raise RunnerConfigurationError(
                "entry_point_not_next", "nodes other than the entry point ran", nodes=extra
            )
        after = serialize_state(snapshot.values)
        update = {key: value for key, value in after.items() if before.get(key, None) != value}
        return {"state_update": update, "extra_nodes_executed": []}, fork

    async def _evaluate(
        self,
        target: GraphTarget,
        evaluators: Sequence[RunnerEvaluatorView],
        case: ExperimentCase,
        output: Mapping[str, Any],
        workspace_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics: dict[str, Any] = {}
        judges: list[dict[str, Any]] = []
        for evaluator in evaluators:
            if evaluator.kind == "runner_custom":
                name = str(evaluator.config.get("name", evaluator.evaluator_id))
                declared = target.evaluators.get(name)
                if (
                    declared is None
                    or declared.version != evaluator.config.get("version", evaluator.version)
                    or declared.digest() != evaluator.config.get("code_digest")
                ):
                    raise RunnerConfigurationError(
                        "evaluator_unavailable",
                        "this runner does not declare the custom evaluator with that digest",
                        evaluator=name,
                    )
                try:
                    value = _metric(declared.fn(case, output))
                except Exception:
                    log.warning("runner evaluator %s failed", name, exc_info=True)
                    value = None
                metrics[name] = {
                    "value": value,
                    "evaluator_ref": f"runner_custom:{name}@{declared.version}",
                }
            elif evaluator.kind == "model_judge":
                judges.append(await self._judge(evaluator, case, output, workspace_id))
        return metrics, judges

    async def _judge(
        self,
        evaluator: RunnerEvaluatorView,
        case: ExperimentCase,
        output: Mapping[str, Any],
        workspace_id: str,
    ) -> dict[str, Any]:
        settings = evaluator.config.get("model")
        settings = dict(settings) if isinstance(settings, Mapping) else {}
        entry: dict[str, Any] = {
            "evaluator_id": evaluator.evaluator_id,
            "evaluator_version": evaluator.version,
            "score": None,
            "parsed": False,
            "judge_model": settings.get("model"),
            "judge_input_tokens": None,
            "judge_output_tokens": None,
        }
        if self._judge_model is None or evaluator.prompt is None:
            return entry
        try:
            record = PromptVersionRecord.model_validate(evaluator.prompt)
            version = ManagedPromptVersion.from_record(
                record, workspace_id=workspace_id, lookup=lambda prompt_id, number: None
            )
            values = {
                "input": case.input,
                "output": output.get("final"),
                "expected": case.expected,
                "rubric": evaluator.config.get("rubric"),
            }
            rendered = version.render(
                {key: values[key] for key in version.variables if key in values}
            )
            messages: list[BaseMessage] = (
                to_langchain_messages(rendered.messages or [])
                if rendered.kind == "chat"
                else [HumanMessage(content=str(rendered.text))]
            )
            reply = await self._judge_model(settings).ainvoke(messages)
            usage = getattr(reply, "usage_metadata", None) or {}
            parsed = json.loads(str(reply.content))
            score = parsed.get("score") if isinstance(parsed, Mapping) else None
            scale = (evaluator.config.get("rubric") or {}).get("scale") or {}
            low, high = scale.get("min"), scale.get("max")
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or (isinstance(low, int) and score < low)
                or (isinstance(high, int) and score > high)
            ):
                return entry
            entry.update(
                score=score,
                parsed=True,
                judge_input_tokens=usage.get("input_tokens"),
                judge_output_tokens=usage.get("output_tokens"),
            )
        except Exception:
            log.warning("judge %s failed", evaluator.evaluator_id, exc_info=True)
        return entry

    def _result(
        self,
        assignment: TrialAssignment,
        state: TrialState,
        *,
        outcome: str,
        started: float,
        turns: list[dict[str, Any]],
        isolation: dict[str, Any],
        output: Any = None,
        error: Optional[dict[str, Any]] = None,
        runner_metrics: Optional[dict[str, Any]] = None,
        judge_scores: Optional[list[dict[str, Any]]] = None,
        source: str = "default_rules/v1",
    ) -> TrialRun:
        with state.lock:
            model_calls = [dict(call) for call in state.model_calls]
        document = {
            "schema": RESULT_SCHEMA,
            "trial_id": assignment.trial_id,
            "attempt": assignment.attempt,
            "outcome": outcome,
            "error": error,
            "output": output,
            "turns": turns,
            "model_calls": model_calls,
            "measurements": {"wall_clock_ms": int((time.monotonic() - started) * 1000)},
            "runner_metrics": runner_metrics or {},
            "judge_scores": judge_scores or [],
            "artifacts": [],
            "runtime": {
                "runtime_digest": assignment.view.arm.runtime_digest,
                "runtime_digest_source": "runner_declared",
                "sdk": SDK,
            },
            "runner_view_digest": assignment.runner_view_digest,
            "isolation": isolation,
            "error_class_source": source,
        }
        return TrialRun("result", result=redact_outbound(document, state.literals))

    def _failure(
        self, error_class: ErrorClass, code: str, message: str, state: TrialState
    ) -> TrialRun:
        return TrialRun(
            "failure",
            error_class=error_class,
            error_code=code,
            message=redact_message(message, state.literals),
        )

    def _from_exception(
        self,
        raised: BaseException,
        assignment: TrialAssignment,
        state: TrialState,
        *,
        started: float,
        turns: list[dict[str, Any]],
        isolation: dict[str, Any],
    ) -> TrialRun:
        error = state.terminal or raised
        common: dict[str, Any] = {"started": started, "turns": turns, "isolation": isolation}
        if isinstance(error, LeaseLost):
            return TrialRun("lost")
        if isinstance(error, TrialBudgetExceeded):
            return self._result(
                assignment,
                state,
                outcome="budget_stopped",
                error={"code": "trial_budget_exceeded", "limit": error.limit},
                **common,
            )
        if isinstance(error, RecordedFixtureMiss):
            return self._result(
                assignment,
                state,
                outcome="recorded_fixture_miss",
                error={
                    "code": "recorded_fixture_miss",
                    "tool": error.tool,
                    "arguments_hash": error.arguments_hash,
                },
                **common,
            )
        if isinstance(error, RecordedFixtureAmbiguous):
            return self._result(
                assignment,
                state,
                outcome="recorded_fixture_ambiguous",
                error={"code": "recorded_fixture_ambiguous", "tool": error.tool},
                **common,
            )
        error_class, source = classify(error, self._classify_error)
        code = (
            "interrupt_unsupported_in_callable"
            if isinstance(error, CallableInterruptError)
            else error_code(error)
        )
        if error_class != "agent":
            return self._failure(error_class, code, str(error), state)
        return self._result(
            assignment,
            state,
            outcome="agent_error",
            error={
                "code": code,
                "type": type(error).__name__,
                "message": redact_message(str(error), state.literals),
            },
            source=source,
            **common,
        )

    async def _aexecute(self, assignment: TrialAssignment, state: TrialState) -> TrialRun:
        started = time.monotonic()
        turns: list[dict[str, Any]] = []
        view = assignment.view
        stage = "assignment_invalid"
        try:
            agent_id, target, binding, bundle = self._prepare(assignment)
            stage = "secret_unresolved"
            secrets = self._resolve_secrets(view.secret_refs)
        except RunnerConfigurationError as refused:
            return self._failure("runner_configuration", refused.code, str(refused), state)
        except Exception as error:
            return self._failure("runner_configuration", stage, str(error), state)
        state.literals = self._literals(secrets)
        context, saver_kind, store_kind = None, "per_trial_in_memory", "per_trial_namespace"
        fork = "state_values_only" if view.case.is_snapshot else "none"
        try:
            context, saver_kind, store_kind = self._context(
                assignment, agent_id, target, secrets, state
            )
            config = self._config(context, _TrialCallbacks(state, view.limits))
            entry_point = self._entry_point(view, target) if view.level == "node" else None
            if isinstance(entry_point, CallableEntryPoint):
                output = await self._run_callable(
                    entry_point, context, binding, bundle, target, config, turns
                )
            else:
                graph = target.factory(context)
                if inspect.isawaitable(graph):
                    graph = await graph
                check_graph(graph, context.checkpointer, context.store)
                managed = bind_langgraph(
                    graph,
                    agent_id=agent_id,
                    binding=binding,
                    resolution=bundle,
                    children=dict(target.children) or None,
                )
                if isinstance(entry_point, GraphNodeEntryPoint):
                    output, fork = await self._run_node(
                        managed, context, entry_point, config, turns
                    )
                else:
                    output = await self._run_agent(managed, context, target, config, turns)
            if state.terminal is not None:
                raise state.terminal
            with state.lock:
                calls = list(state.model_calls)
            _manifest_check(calls, bundle)
            metrics, judges = await self._evaluate(
                target, view.runner_evaluators, view.case, output, binding.workspace_id
            )
            isolation = isolation_record(
                checkpointer=saver_kind, store=store_kind, fork_fidelity=fork
            )
            return self._result(
                assignment,
                state,
                outcome="evaluated",
                started=started,
                turns=turns,
                isolation=isolation,
                output=output,
                runner_metrics=metrics,
                judge_scores=judges,
            )
        except Exception as raised:
            isolation = isolation_record(
                checkpointer=saver_kind, store=store_kind, fork_fidelity=fork
            )
            return self._from_exception(
                raised, assignment, state, started=started, turns=turns, isolation=isolation
            )
        finally:
            if context is not None:
                self._cleanup(context, target)

    def _cleanup(self, context: TrialContext, target: GraphTarget) -> None:
        if target.on_trial_end is not None:
            try:
                target.on_trial_end(context)
            except Exception:
                log.warning("on_trial_end failed for trial %s", context.trial_id, exc_info=True)
        if target.keep_checkpoints:
            return
        delete = getattr(context.checkpointer, "delete_thread", None)
        if callable(delete):
            try:
                delete(context.thread_key)
            except NotImplementedError:
                log.debug("the checkpointer cannot delete threads")

    def timeout_run(
        self, assignment: TrialAssignment, state: TrialState, started: float
    ) -> TrialRun:
        fork = "state_values_only" if assignment.view.case.is_snapshot else "none"
        target = self.targets.get(str(assignment.view.binding.get("agent_id")))
        saver_kind = (
            "per_trial_factory"
            if target is not None and target.checkpointer_factory is not None
            else "per_trial_in_memory"
        )
        return self._result(
            assignment,
            state,
            outcome="agent_timeout",
            started=started,
            turns=[],
            isolation=isolation_record(
                checkpointer=saver_kind, store="per_trial_namespace", fork_fidelity=fork
            ),
            error={"code": "deadline_exceeded"},
        )

    async def _hello(self, http: _RunnerHttp) -> dict[str, Any]:
        response = await aapi_request(
            http, "POST", "/v1/experiment-runner/hello", self.hello_document(), retry=True
        )
        self._last_hello = time.monotonic()
        return dict(response.body)

    async def _refresh_hello(self, http: _RunnerHttp) -> bool:
        try:
            await self._hello(http)
        except ApiError as error:
            if error.code != "registry_unavailable" and error.status < 500:
                raise
            log.warning("hello failed: %s", error.code)
            await _asleep(1.0)
            return False
        return True

    async def _heartbeat(
        self,
        http: _RunnerHttp,
        assignment: TrialAssignment,
        trial: asyncio.Task[TrialRun],
        control: _Control,
    ) -> None:
        interval = max(0.01, float(assignment.heartbeat_interval_seconds))
        path = f"/v1/experiment-runner/trials/{segment(assignment.trial_id)}/heartbeat"
        while not trial.done():
            await _asleep(interval)
            if trial.done():
                return
            try:
                response = await aapi_request(
                    http, "POST", path, {"lease_token": assignment.lease_token}, retry=True
                )
            except ApiError as error:
                if error.code == "experiment_lease_stale":
                    control.reason = "lost"
                    trial.cancel()
                    return
                log.warning("heartbeat for trial %s failed: %s", assignment.trial_id, error.code)
                continue
            body = response.body
            for flag, reason in (
                ("cancel_requested", "cancelled"),
                ("stop_requested", "stopped"),
                ("deadline_exceeded", "timeout"),
            ):
                if body.get(flag) is True:
                    control.reason = reason
                    trial.cancel()
                    return

    async def _post(
        self, http: _RunnerHttp, assignment: TrialAssignment, route: str, body: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        path = f"/v1/experiment-runner/trials/{segment(assignment.trial_id)}/{route}"
        try:
            response = await aapi_request(http, "POST", path, body, retry=True)
        except ApiError as error:
            if error.code in ("experiment_lease_stale", "experiment_release_after_effect"):
                log.info("trial %s %s refused: %s", assignment.trial_id, route, error.code)
                return None
            raise
        return dict(response.body)

    async def _report_result(
        self, http: _RunnerHttp, assignment: TrialAssignment, result: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        key = uuid.uuid5(
            _REPORT_NAMESPACE,
            f"{assignment.trial_id}:{assignment.attempt}:{assignment.lease_token}",
        )
        body = self._outbox.setdefault(
            key, {"lease_token": assignment.lease_token, "result": dict(result)}
        )
        for delay in (*REPORT_ROUNDS, None):
            try:
                acknowledged = await self._post(http, assignment, "result", body)
            except ApiError as error:
                if error.code == "experiment_result_secret_detected":
                    self._outbox.pop(key, None)
                    return await self._post(
                        http,
                        assignment,
                        "failure",
                        self._failure_body(
                            assignment,
                            "runner_configuration",
                            "secret_in_report",
                            "the result matched a secret pattern",
                        ),
                    )
                if error.code == "experiment_result_invalid":
                    self._outbox.pop(key, None)
                    return await self._post(
                        http,
                        assignment,
                        "failure",
                        self._failure_body(
                            assignment,
                            "runner_configuration",
                            error.reason or "result_invalid",
                            error.message,
                        ),
                    )
                if error.code == "experiment_result_conflict" or delay is None:
                    self._outbox.pop(key, None)
                    log.warning(
                        "result of trial %s not accepted: %s", assignment.trial_id, error.code
                    )
                    return None
                if error.code == "registry_unavailable" or error.status >= 500:
                    await _asleep(delay)
                    continue
                self._outbox.pop(key, None)
                raise
            self._outbox.pop(key, None)
            return acknowledged
        return None

    def _failure_body(
        self, assignment: TrialAssignment, error_class: str, code: str, message: str
    ) -> dict[str, Any]:
        return {
            "lease_token": assignment.lease_token,
            "error_class": error_class,
            "error_code": code,
            "message": redact_message(message, self._literals()),
        }

    async def _deliver(self, http: _RunnerHttp, assignment: TrialAssignment, run: TrialRun) -> None:
        if run.kind == "lost":
            return
        if run.kind == "release":
            await self._post(
                http,
                assignment,
                "release",
                {"lease_token": assignment.lease_token, "reason": run.reason or "shutdown"},
            )
            return
        if run.kind == "failure":
            await self._post(
                http,
                assignment,
                "failure",
                {
                    "lease_token": assignment.lease_token,
                    "error_class": run.error_class,
                    "error_code": run.error_code,
                    "message": run.message or "",
                },
            )
            return
        assert run.result is not None
        await self._report_result(http, assignment, run.result)

    async def _process(self, http: _RunnerHttp, raw: Mapping[str, Any]) -> TrialRun:
        try:
            assignment = TrialAssignment.model_validate(raw)
        except ValidationError:
            log.warning("an assignment could not be parsed; its lease will expire")
            return TrialRun("lost")
        state = TrialState(
            proxy=HttpToolProxy(http, assignment.trial_id),
            lease_token=assignment.lease_token,
            literals=self._literals(),
        )
        with LOG_REDACTOR.tracking(lambda: state.literals):
            return await self._execute_and_deliver(http, assignment, state)

    async def _execute_and_deliver(
        self, http: _RunnerHttp, assignment: TrialAssignment, state: TrialState
    ) -> TrialRun:
        started = time.monotonic()
        control = _Control()
        trial = asyncio.ensure_future(self._aexecute(assignment, state))
        beat = asyncio.ensure_future(self._heartbeat(http, assignment, trial, control))
        try:
            done, _ = await asyncio.wait({trial}, timeout=_seconds_until(assignment.deadline_at))
            if trial not in done:
                trial.cancel()
                control.reason = "timeout"
            try:
                run = await trial
            except asyncio.CancelledError:
                if control.reason is None:
                    raise
                run = self._interrupted(control.reason, assignment, state, started)
            except Exception:
                log.exception("the runner failed while executing trial %s", assignment.trial_id)
                run = TrialRun(
                    "failure",
                    error_class="infrastructure",
                    error_code="runner_internal_error",
                    message="the runner failed while executing the trial",
                )
        finally:
            beat.cancel()
        try:
            await self._deliver(http, assignment, run)
        except ApiError as error:
            log.warning("delivery for trial %s failed: %s", assignment.trial_id, error.code)
        except Exception:
            log.exception("delivery for trial %s failed", assignment.trial_id)
        return run

    def _interrupted(
        self, reason: Optional[str], assignment: TrialAssignment, state: TrialState, started: float
    ) -> TrialRun:
        if reason == "timeout":
            return self.timeout_run(assignment, state, started)
        if reason in _NO_RETRY_REASONS:
            return TrialRun("release", reason=reason)
        return TrialRun("lost")

    async def _worker(self, http: _RunnerHttp, serve: _ServeState) -> None:
        while not self._stopping.is_set():
            if not serve.reserve():
                return
            stale = time.monotonic() - self._last_hello >= HELLO_REFRESH_SECONDS
            if stale and not await self._refresh_hello(http):
                serve.claimed -= 1
                continue
            try:
                response = await aapi_request(
                    http,
                    "POST",
                    "/v1/experiment-runner/claims",
                    {"wait_seconds": serve.wait_seconds()},
                )
            except ApiError as error:
                serve.claimed -= 1
                if error.code == "experiment_runner_hello_required":
                    await self._refresh_hello(http)
                    continue
                if error.status in (401, 403):
                    raise
                log.warning("claim failed: %s", error.code)
                await _asleep(1.0)
                continue
            assignment = response.body.get("assignment")
            if response.status == 204 or not isinstance(assignment, Mapping):
                serve.claimed -= 1
                if serve.idle():
                    return
                continue
            serve.last_activity = time.monotonic()
            if self._stopping.is_set():
                await self._release_unstarted(http, assignment)
                return
            await self._process(http, assignment)
            serve.completed += 1
            serve.last_activity = time.monotonic()

    async def _release_unstarted(self, http: _RunnerHttp, raw: Mapping[str, Any]) -> None:
        trial_id, lease_token = raw.get("trial_id"), raw.get("lease_token")
        if not isinstance(trial_id, str) or not isinstance(lease_token, str):
            return
        path = f"/v1/experiment-runner/trials/{segment(trial_id)}/release"
        try:
            await aapi_request(
                http, "POST", path, {"lease_token": lease_token, "reason": "shutdown"}, retry=True
            )
        except ApiError as error:
            log.info("release of trial %s refused: %s", trial_id, error.code)

    async def aserve(
        self, *, max_trials: Optional[int] = None, idle_timeout: Optional[float] = None
    ) -> int:
        http = self._connect()
        self._stopping.clear()
        serve = _ServeState(max_trials, idle_timeout)
        try:
            await self._hello(http)
            workers = [
                asyncio.ensure_future(self._worker(http, serve))
                for _ in range(self.max_concurrency)
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                for worker in workers:
                    worker.cancel()
        finally:
            await aclose_pool(http)
        return serve.completed

    def serve(
        self, *, max_trials: Optional[int] = None, idle_timeout: Optional[float] = None
    ) -> int:
        return asyncio.run(self.aserve(max_trials=max_trials, idle_timeout=idle_timeout))


def _view_digest(view: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(view).encode("utf-8")).hexdigest()


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_assignment(
    engine: LocalPromptEngine,
    *,
    agent_id: str,
    release_id: str,
    case: Mapping[str, Any],
    experiment_id: str = "exp_local",
    trial_id: Optional[str] = None,
    attempt: int = 1,
    arm_key: Optional[str] = None,
    level: Literal["agent", "node"] = "agent",
    entry_point: Optional[Mapping[str, Any]] = None,
    seed: str = "0",
    tools: Optional[Mapping[str, Any]] = None,
    limits: Optional[Mapping[str, Any]] = None,
    secret_refs: Sequence[str] = (),
    runner_evaluators: Sequence[Mapping[str, Any]] = (),
    parent_binding_id: Optional[str] = None,
    heartbeat_interval_seconds: float = 30,
    deadline_seconds: int = 900,
) -> dict[str, Any]:
    artifacts = engine.resolve(agent_id, release_id=release_id)
    release = artifacts["release"]
    trial = trial_id or "extr_" + ulid.new().str.lower()
    key = arm_key or "arm_" + uuid.uuid4().hex[:8]
    provenance = (
        case.get("input", {}).get("provenance") if isinstance(case.get("input"), Mapping) else None
    )
    parent = parent_binding_id
    if parent is None and isinstance(provenance, Mapping):
        value = provenance.get("parent_binding_id")
        parent = value if isinstance(value, str) else None
    now = datetime.now(timezone.utc)
    binding = {
        "schema": "agenomic.execution_binding/v1",
        "binding_id": "bnd_" + ulid.new().str.lower(),
        "workspace_id": engine.workspace_id,
        "agent_id": agent_id,
        "thread_key": f"exp:{experiment_id}:{trial}:a{attempt}",
        "scope": "thread",
        "release_id": release_id,
        "release_name": release["release_name"],
        "genome_version": release["genome_version"],
        "prompt_manifest_digest": artifacts["prompt_manifest_digest"],
        "runtime": {"bundle_id": release["bundle_id"], "bundle_hash": release["bundle_hash"]},
        "resolved_from": {"release_id": release_id},
        "children": {
            child_id: {
                "release_id": child["release_id"],
                "genome_version": child["genome_version"],
                "prompt_manifest_digest": child["prompt_manifest_digest"],
                "source": "manifest",
                "channel": None,
                "generation": None,
            }
            for child_id, child in artifacts["children"].items()
        },
        "parent_binding_id": parent,
        "experiment": {
            "experiment_id": experiment_id,
            "trial_id": trial,
            "arm_key": key,
            "attempt": attempt,
        },
        "runtime_client": None,
        "created_at": _stamp(now),
        "created_by": {"user_id": None, "api_key_id": None},
    }
    view = {
        "schema": VIEW_SCHEMA,
        "experiment_id": experiment_id,
        "trial_id": trial,
        "attempt": attempt,
        "level": level,
        "case": {name: value for name, value in case.items() if name != "schema"},
        "arm": {
            "arm_key": key,
            "release_id": release_id,
            "genome_version": release["genome_version"],
            "prompt_manifest_digest": artifacts["prompt_manifest_digest"],
            "runtime_digest": release["bundle_hash"],
        },
        "binding": binding,
        "prompts": artifacts,
        "isolation": {
            "store_namespace": ["agenomic_exp", experiment_id, trial, f"a{attempt}"],
            "resource_suffix": hashlib.sha256(f"{trial}:{attempt}".encode()).hexdigest()[:8],
        },
        "seed": seed,
        "tools": {
            "mode": "none",
            "tool_repetition": attempt,
            "declared_tools": [],
            "on_fixture_miss": "stop_trial",
            **dict(tools or {}),
        },
        "protect_overlay": None,
        "limits": {
            "max_tokens_per_trial": None,
            "max_model_calls_per_trial": None,
            "max_output_bytes": None,
            **dict(limits or {}),
        },
        "runner_evaluators": [dict(item) for item in runner_evaluators],
        "secret_refs": list(secret_refs),
        "entry_point": None if entry_point is None else dict(entry_point),
        "node_scope": None,
    }
    return {
        "schema": ASSIGNMENT_SCHEMA,
        "trial_id": trial,
        "attempt": attempt,
        "lease_token": str(uuid.uuid4()),
        "lease_until": _stamp(now + timedelta(seconds=120)),
        "deadline_at": _stamp(now + timedelta(seconds=deadline_seconds)),
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "runner_view_digest": _view_digest(view),
        "view": view,
    }


def snapshot_case(
    graph: Any,
    thread_id: str,
    *,
    agent_id: Optional[str] = None,
    case_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    if isinstance(graph, ManagedGraph):
        raise ValueError("pass the compiled graph and its checkpointer, not the bound proxy")
    if getattr(graph, "checkpointer", None) in (None, False, True):
        raise ValueError("the graph has no checkpointer to read the thread from")
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    state = graph.get_state({"configurable": configurable})
    if not state.config.get("configurable", {}).get("checkpoint_id"):
        raise SnapshotRefusedError("thread_not_found", "the thread has no checkpoint")
    if state.interrupts:
        raise SnapshotRefusedError("pending_interrupt", "the thread waits on an interrupt")
    if any(task.state is not None for task in state.tasks):
        raise SnapshotRefusedError(
            "subgraph_in_flight", "a subgraph task of the thread is in flight"
        )
    parent = graph.get_state(state.parent_config) if state.parent_config else None
    writers = [task.name for task in parent.tasks] if parent is not None else []
    if len(writers) != 1:
        raise SnapshotRefusedError(
            "parallel_step",
            f"the last step had {len(writers)} writers, so its values cannot be attributed",
        )
    metadata = dict(state.metadata or {})
    stamped = metadata.get("agenomic_agent_id")
    if agent_id is not None and stamped is not None and stamped != agent_id:
        raise SnapshotRefusedError("agent_mismatch", "the thread belongs to another agent")
    try:
        values = serialize_state(state.values)
        scratch = graph.copy(update={"checkpointer": InMemorySaver(), "store": None})
        scratch_config: dict[str, Any] = {"configurable": {"thread_id": "agenomic-snapshot-check"}}
        scratch.update_state(scratch_config, revive_state(values), as_node=writers[0])
        seeded = scratch.get_state(scratch_config)
        seeded_values = serialize_state(seeded.values)
    except RunnerConfigurationError as error:
        raise SnapshotRefusedError("not_serializable", error.message) from error
    if seeded_values != values:
        raise SnapshotRefusedError(
            "non_identity_reducer",
            "re-applying the values through the channel reducers changes them",
        )
    if tuple(seeded.next) != tuple(state.next):
        raise SnapshotRefusedError("next_mismatch", "the seeded thread would run other nodes")
    checkpoint = str(state.config["configurable"]["checkpoint_id"])
    binding_id = metadata.get("agenomic_binding_id")
    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id or f"snapshot-{checkpoint}"[:128],
        "kind": "node_state",
        "input": {
            "provenance": {
                "source": "production_snapshot",
                "parent_binding_id": binding_id if isinstance(binding_id, str) else None,
                "checkpoint_id": checkpoint,
            }
        },
        "expected": None,
        "initial_state": values,
        "turns": [],
        "tags": list(tags),
    }
