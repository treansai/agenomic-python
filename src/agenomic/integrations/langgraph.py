"""LangGraph integration — lazy-imported wrapper for state-graph nodes."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.trace.context import current_recorder
from agenomic.types.trace import CallStatus, ToolCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agenomic.canonical.recorder import CanonicalRun


logger = logging.getLogger("agenomic.integrations.langgraph")


def _warn_if_nothing_wrapped(function: str, total: int, wrapped: int) -> None:
    if total and not wrapped:
        logger.warning(
            "%s wrapped 0 of %d nodes: a real LangGraph graph holds Runnable objects, "
            "not plain callables, so nothing will be recorded; "
            "use TrackingCallbackHandler to observe it",
            function,
            total,
        )


def _hash(data: Any) -> str:
    try:
        blob = canonical_cbor(data if isinstance(data, dict) else {"v": repr(data)})
    except Exception:
        blob = canonical_cbor({"repr": repr(data)})
    return blake3_hex(blob)


def _record_call(name: str, input_hash: str, started: float, result: Any, failed: bool) -> None:
    recorder = current_recorder()
    if recorder is None:
        return
    latency_ms = int((time.perf_counter() - started) * 1000)
    if failed:
        recorder.record_tool_call(
            ToolCall(
                tool=name,
                protocol="local",
                server="langgraph",
                input_hash=input_hash,
                latency_ms=latency_ms,
                status=CallStatus.ERROR,
            )
        )
        return
    recorder.record_tool_call(
        ToolCall(
            tool=name,
            protocol="local",
            server="langgraph",
            input_hash=input_hash,
            output_hash=_hash(result),
            latency_ms=latency_ms,
        )
    )


def _traced_sync(name: str, original: Any) -> Any:
    def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
        input_hash = _hash(state)
        started = time.perf_counter()
        try:
            result = original(state, *args, **kwargs)
        except Exception:
            _record_call(name, input_hash, started, None, True)
            raise
        _record_call(name, input_hash, started, result, False)
        return result

    return wrapped


def _traced_async(name: str, original: Any) -> Any:
    async def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
        input_hash = _hash(state)
        started = time.perf_counter()
        try:
            result = await original(state, *args, **kwargs)
        except Exception:
            _record_call(name, input_hash, started, None, True)
            raise
        _record_call(name, input_hash, started, result, False)
        return result

    return wrapped


def _wrap_runnable_callable(name: str, node: Any) -> bool:
    """Wrap the function held by a LangGraph ``RunnableCallable`` (``func`` and ``afunc``).

    LangGraph keeps node functions inside a ``RunnableCallable`` that is not itself
    callable, reachable as ``node.runnable`` (state graph spec) or ``node.bound``
    (compiled graph). Replacing ``func`` keeps the signature LangGraph already
    inspected, since the wrapper forwards every argument.
    """
    target: Any = getattr(node, "runnable", None) or getattr(node, "bound", None)
    wrapped = False
    func = getattr(target, "func", None)
    if callable(func):
        target.func = _traced_sync(name, func)
        wrapped = True
    afunc = getattr(target, "afunc", None)
    if callable(afunc):
        target.afunc = _traced_async(name, afunc)
        wrapped = True
    return wrapped


def instrument_langgraph(graph: Any) -> Any:
    """Wrap each node in a LangGraph state graph so executions emit a ToolCall.

    Lazy-imports ``langgraph``. Raises ImportError with a helpful message
    if not installed.

    Each wrapped node, when invoked inside a ``@trace_agent_run``, records a
    :class:`ToolCall` with input/output hashes and latency. Plain callables and
    function nodes of a real ``StateGraph`` (sync and async) are wrapped; other
    Runnable nodes are not, and a warning is logged when nothing is wrapped (use
    ``TrackingCallbackHandler`` for those).

    Example:
        >>> # graph = instrument_langgraph(StateGraph(MyState))  # doctest: +SKIP
    """
    try:
        import langgraph  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "langgraph not installed. Install with: pip install agenomic[langgraph]"
        ) from e

    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return graph
    count = 0
    for name, node in list(nodes.items()):
        original = getattr(node, "runnable", None) or node
        if not callable(original):
            if _wrap_runnable_callable(name, node):
                count += 1
            continue
        count += 1
        wrapped = _traced_sync(name, original)
        if hasattr(node, "runnable"):
            node.runnable = wrapped
        else:
            nodes[name] = wrapped
    _warn_if_nothing_wrapped("instrument_langgraph", len(nodes), count)
    return graph


def instrument_langgraph_canonical(
    graph: Any,
    run: CanonicalRun,
    *,
    llm_nodes: Iterable[str] = (),
) -> Any:
    """Auto-instrument a LangGraph state graph into **canonical v0.3** events
    on `run`, with no manual instrumentation in the nodes themselves.

    Each node execution maps to a canonical event: a node named in `llm_nodes`
    emits ``llm.requested`` + ``llm.responded`` (the dynamic *Agent Decision
    Graph*); every other node emits ``tool.call.proposed`` +
    ``tool.call.executed`` (the deterministic *Workflow Graph*). A raising node
    emits ``error.raised`` and re-raises.

    Works on a duck-typed ``graph.nodes`` mapping whose nodes (or their
    ``runnable``) are plain callables. A real LangGraph graph holds Runnable
    objects, so nothing is wrapped and a warning is logged; use
    ``TrackingCallbackHandler`` for those.

    Example:
        >>> # graph = instrument_langgraph_canonical(compiled, run, llm_nodes=["agent"])  # doctest: +SKIP
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        raise TypeError("graph has no `.nodes` mapping to instrument")
    llm_set = set(llm_nodes)

    count = 0
    for name, node in list(nodes.items()):
        original = getattr(node, "runnable", None) or node
        if not callable(original):
            continue
        count += 1

        def wrapped(
            state: Any,
            *,
            _name: str = name,
            _original: Any = original,
            _is_llm: bool = name in llm_set,
        ) -> Any:
            try:
                result = _original(state)
            except Exception as exc:
                run.log_error(message=str(exc), kind=type(exc).__name__)
                raise
            if _is_llm:
                run.log_llm(prompt=state, response=result)
            else:
                run.log_tool_call(tool=_name, arguments=state, result=result)
            return result

        if hasattr(node, "runnable"):
            node.runnable = wrapped
        else:
            nodes[name] = wrapped
    _warn_if_nothing_wrapped("instrument_langgraph_canonical", len(nodes), count)
    return graph


__all__ = ["instrument_langgraph", "instrument_langgraph_canonical"]
