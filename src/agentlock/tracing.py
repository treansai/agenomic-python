from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, cast
from uuid import uuid4

from agentlock.client import AgentLockClient
from agentlock.models import (
    AgentRun,
    HumanFeedback,
    MemoryAccess,
    ModelCall,
    PolicyCheck,
    RunCompleted,
    ToolCall,
    TraceEnvelope,
    TraceEventUnion,
)
from agentlock.redaction import Redactor
from agentlock.utils import hash_payload, merge_metadata, to_jsonable, utc_now

F = TypeVar("F", bound=Callable[..., Any])


class TraceRecorder:
    def __init__(
        self,
        agent_id: str,
        release: str = "dev",
        input: Any | None = None,
        labels: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        redactor: Redactor | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid4().hex
        self.run_id = run_id or uuid4().hex
        self.agent_id = agent_id
        self.release = release
        self.timestamp = utc_now()
        self.labels = list(labels or [])
        self.metadata = merge_metadata(metadata)
        self.redactor = redactor
        self._raw_input = input
        self._input_hash = hash_payload(input)
        self._input = self._sanitize(input)
        self._completed = False
        self._events: list[TraceEventUnion] = [
            AgentRun(
                trace_id=self.trace_id,
                run_id=self.run_id,
                agent_id=self.agent_id,
                release=self.release,
                input_hash=self._input_hash,
                input=self._input,
                labels=self.labels,
            )
        ]
        self.model_calls: list[ModelCall] = []
        self.tool_calls: list[ToolCall] = []
        self.memory_accesses: list[MemoryAccess] = []
        self.policy_checks: list[PolicyCheck] = []
        self.human_feedback: list[HumanFeedback] = []

    def add_model_call(
        self,
        name: str,
        provider: str | None = None,
        request: Any | None = None,
        response: Any | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        success: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelCall:
        model_call = ModelCall(
            name=name,
            provider=provider,
            request=self._sanitize(request),
            response=self._sanitize(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            metadata=merge_metadata(metadata),
        )
        self.model_calls.append(model_call)
        self._events.append(model_call)
        return model_call

    def add_tool_call(
        self,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        success: bool = True,
        latency_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolCall:
        tool_call = ToolCall(
            name=name,
            input=self._sanitize(input),
            output=self._sanitize(output),
            success=success,
            latency_ms=latency_ms,
            metadata=merge_metadata(metadata),
        )
        self.tool_calls.append(tool_call)
        self._events.append(tool_call)
        return tool_call

    def add_memory_access(
        self,
        store: str,
        operation: str,
        key: str | None = None,
        value: Any | None = None,
        success: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryAccess:
        memory_access = MemoryAccess(
            store=store,
            operation=operation,
            key=key,
            value=self._sanitize(value),
            success=success,
            metadata=merge_metadata(metadata),
        )
        self.memory_accesses.append(memory_access)
        self._events.append(memory_access)
        return memory_access

    def add_policy_check(
        self,
        name: str,
        passed: bool,
        details: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyCheck:
        policy_check = PolicyCheck(
            name=name,
            passed=passed,
            details=self._sanitize(details),
            metadata=merge_metadata(metadata),
        )
        self.policy_checks.append(policy_check)
        self._events.append(policy_check)
        return policy_check

    def add_human_feedback(
        self,
        rating: int | None = None,
        comment: str | None = None,
        reviewer: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HumanFeedback:
        feedback = HumanFeedback(
            rating=rating,
            comment=comment,
            reviewer=reviewer,
            metadata=merge_metadata(metadata),
        )
        self.human_feedback.append(feedback)
        self._events.append(feedback)
        return feedback

    def complete(
        self,
        final_output: Any | None = None,
        success: bool = True,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceEnvelope:
        if self._completed:
            msg = "TraceRecorder.complete() may only be called once per trace"
            raise RuntimeError(msg)

        self._completed = True
        output_hash = hash_payload(final_output) if final_output is not None else None
        run_completed = RunCompleted(
            success=success,
            final_output_hash=output_hash,
            error=error,
        )

        envelope_metadata = merge_metadata(
            self.metadata,
            metadata,
            {
                "input_hash": self._input_hash,
                "output_hash": output_hash,
                "success": success,
                "error_type": error,
            },
        )

        events = [*self._events, run_completed]

        return TraceEnvelope(
            trace_id=self.trace_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
            release=self.release,
            timestamp=self.timestamp,
            input=self._input,
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            final_output=self._sanitize(final_output),
            labels=self.labels,
            metadata=envelope_metadata,
            events=events,
            memory_accesses=self.memory_accesses,
            policy_checks=self.policy_checks,
            human_feedback=self.human_feedback,
            run_completed=run_completed,
        )

    def _sanitize(self, value: Any) -> Any:
        normalized = to_jsonable(value)
        if self.redactor is None:
            return normalized
        return self.redactor.redact(normalized)


def trace_agent_run(
    agent_id: str,
    release: str = "dev",
    client: AgentLockClient | None = None,
    redactor: Redactor | None = None,
    labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                payload = _build_call_payload(func, args, kwargs)
                recorder = TraceRecorder(
                    agent_id=agent_id,
                    release=release,
                    input=payload,
                    labels=labels,
                    metadata=metadata,
                    redactor=redactor,
                )
                call_kwargs = _inject_trace_argument(func, kwargs, recorder)
                try:
                    result = await cast(Callable[..., Awaitable[Any]], func)(*args, **call_kwargs)
                except Exception as exc:
                    trace = recorder.complete(
                        success=False,
                        error=type(exc).__name__,
                        metadata={"exception_type": type(exc).__name__},
                    )
                    await _safe_emit_async(client, trace)
                    raise

                trace = recorder.complete(final_output=result, success=True)
                await _safe_emit_async(client, trace)
                return result

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = _build_call_payload(func, args, kwargs)
            recorder = TraceRecorder(
                agent_id=agent_id,
                release=release,
                input=payload,
                labels=labels,
                metadata=metadata,
                redactor=redactor,
            )
            call_kwargs = _inject_trace_argument(func, kwargs, recorder)
            try:
                result = func(*args, **call_kwargs)
            except Exception as exc:
                trace = recorder.complete(
                    success=False,
                    error=type(exc).__name__,
                    metadata={"exception_type": type(exc).__name__},
                )
                _safe_emit_sync(client, trace)
                raise

            trace = recorder.complete(final_output=result, success=True)
            _safe_emit_sync(client, trace)
            return result

        return cast(F, sync_wrapper)

    return decorator


def _build_call_payload(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    payload = dict(bound.arguments)
    payload.pop("trace", None)
    payload.pop("agentlock_trace", None)

    if len(payload) == 1:
        return next(iter(payload.values()))
    return payload


def _inject_trace_argument(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    recorder: TraceRecorder,
) -> dict[str, Any]:
    signature = inspect.signature(func)
    updated_kwargs = dict(kwargs)
    for candidate in ("trace", "agentlock_trace"):
        if candidate in signature.parameters and candidate not in updated_kwargs:
            updated_kwargs[candidate] = recorder
            break
    return updated_kwargs


def _safe_emit_sync(client: AgentLockClient | None, trace: TraceEnvelope) -> None:
    if client is None:
        return
    try:
        client.emit_trace(trace)
    except Exception:
        return


async def _safe_emit_async(client: AgentLockClient | None, trace: TraceEnvelope) -> None:
    if client is None:
        return
    try:
        await client.emit_trace_async(trace)
    except Exception:
        return
