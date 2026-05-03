"""LangGraph integration — lazy-imported wrapper for state-graph nodes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentlock.crypto.canonical import canonical_cbor
from agentlock.crypto.hashing import blake3_hex
from agentlock.trace.context import current_recorder
from agentlock.types.trace import CallStatus, ToolCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


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
            "langgraph not installed. Install with: pip install agentlock[langgraph]"
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


__all__ = ["instrument_langgraph"]
