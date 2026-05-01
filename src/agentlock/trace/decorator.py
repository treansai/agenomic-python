"""@trace_agent_run decorator — sync and async."""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Optional, TypeVar, cast

import ulid
from typing_extensions import ParamSpec

from agentlock.exporters.base import Exporter
from agentlock.redaction.engine import RedactionEngine
from agentlock.trace.context import (
    reset_current_recorder,
    set_current_recorder,
)
from agentlock.trace.envelope_builder import build_envelope
from agentlock.trace.recorder import TraceRecorder

logger = logging.getLogger("agentlock.trace.decorator")

P = ParamSpec("P")
R = TypeVar("R")


def _capture_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"args": list(args), "kwargs": dict(kwargs)}


def _export_safe(exporter: Optional[Exporter], envelope: Any) -> None:
    if exporter is None:
        return
    try:
        result = exporter.export(envelope)
    except Exception:  # pragma: no cover - defensive
        logger.exception("exporter %r failed", exporter)
        return
    if inspect.isawaitable(result):
        coro = cast("Any", result)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.ensure_future(coro)
        else:
            asyncio.run(coro)


async def _export_safe_async(
    exporter: Optional[Exporter], envelope: Any
) -> None:
    if exporter is None:
        return
    try:
        result = exporter.export(envelope)
        if inspect.isawaitable(result):
            await result
    except Exception:  # pragma: no cover - defensive
        logger.exception("exporter %r failed", exporter)


def trace_agent_run(
    agent_id: str,
    *,
    release: Optional[str] = None,
    exporter: Optional[Exporter] = None,
    redaction: Optional[RedactionEngine] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a function so each invocation is traced as one agent run.

    Captures (after redaction):
        - input arguments
        - model calls + tool calls recorded via TraceRecorder
        - final output, or error if the function raised
        - duration in milliseconds

    Example:
        >>> from agentlock.exporters.jsonl import JsonlExporter  # doctest: +SKIP
        >>> @trace_agent_run("agent://acme/demo")  # doctest: +SKIP
        ... def handle(query: str) -> dict:
        ...     return {"answer": query.upper()}
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                recorder = TraceRecorder(
                    agent_id=agent_id,
                    run_id=ulid.new().str,
                    trace_id=ulid.new().str,
                )
                token = set_current_recorder(recorder)
                error: Optional[str] = None
                output: Any = None
                try:
                    output = await func(*args, **kwargs)
                    return cast("R", output)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    reset_current_recorder(token)
                    envelope = build_envelope(
                        recorder,
                        raw_input=_capture_args(args, kwargs),
                        raw_output=output,
                        error=error,
                        release=release,
                        capture_input=capture_input,
                        capture_output=capture_output,
                        redaction=redaction,
                    )
                    await _export_safe_async(exporter, envelope)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            recorder = TraceRecorder(
                agent_id=agent_id,
                run_id=ulid.new().str,
                trace_id=ulid.new().str,
            )
            token = set_current_recorder(recorder)
            error: Optional[str] = None
            output: Any = None
            try:
                output = func(*args, **kwargs)
                return cast("R", output)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                reset_current_recorder(token)
                envelope = build_envelope(
                    recorder,
                    raw_input=_capture_args(args, kwargs),
                    raw_output=output,
                    error=error,
                    release=release,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    redaction=redaction,
                )
                _export_safe(exporter, envelope)

        return sync_wrapper

    return decorator


__all__ = ["trace_agent_run"]
