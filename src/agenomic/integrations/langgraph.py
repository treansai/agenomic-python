"""LangGraph integration — lazy-imported wrapper for state-graph nodes."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.trace.context import current_recorder
from agenomic.types.trace import CallStatus, ToolCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agenomic.canonical.recorder import CanonicalRun


def _hash(data: Any) -> str:
    try:
        blob = canonical_cbor(data if isinstance(data, dict) else {"v": repr(data)})
    except Exception:
        blob = canonical_cbor({"repr": repr(data)})
    return blake3_hex(blob)


def instrument_langgraph(graph: Any) -> Any:
    """Wrap each node in a LangGraph state graph so executions emit a ToolCall.

    Lazy-imports ``langgraph``. Raises ImportError with a helpful message
    if not installed.

    Each wrapped node, when invoked inside a ``@trace_agent_run``, records a
    :class:`ToolCall` with input/output hashes and latency.

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
    for name, node in list(nodes.items()):
        original = getattr(node, "runnable", None) or node
        if not callable(original):
            continue

        def wrapped(
            state: Any,
            *,
            _name: str = name,
            _original: Any = original,
        ) -> Any:
            recorder = current_recorder()
            input_hash = _hash(state)
            started = time.perf_counter()
            try:
                result = _original(state)
            except Exception:
                if recorder is not None:
                    recorder.record_tool_call(
                        ToolCall(
                            tool=_name,
                            protocol="local",
                            server="langgraph",
                            input_hash=input_hash,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status=CallStatus.ERROR,
                        )
                    )
                raise
            if recorder is not None:
                recorder.record_tool_call(
                    ToolCall(
                        tool=_name,
                        protocol="local",
                        server="langgraph",
                        input_hash=input_hash,
                        output_hash=_hash(result),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
            return result

        if hasattr(node, "runnable"):
            node.runnable = wrapped
        else:
            nodes[name] = wrapped
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

    Works on the duck-typed ``graph.nodes`` mapping (each node's ``runnable`` or
    the node itself), so it drives both real compiled LangGraph graphs and
    lightweight test graphs.

    Example:
        >>> # graph = instrument_langgraph_canonical(compiled, run, llm_nodes=["agent"])  # doctest: +SKIP
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        raise TypeError("graph has no `.nodes` mapping to instrument")
    llm_set = set(llm_nodes)

    for name, node in list(nodes.items()):
        original = getattr(node, "runnable", None) or node
        if not callable(original):
            continue

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
    return graph


__all__ = ["instrument_langgraph", "instrument_langgraph_canonical"]
