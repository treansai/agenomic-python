"""MCP integration helper — manual call recording.

Unlike OpenAI/Anthropic/LangGraph, MCP doesn't have a single SDK to wrap —
call :func:`trace_mcp_call` after invoking your MCP client.
"""

from __future__ import annotations

from typing import Optional

from agentlock.crypto.canonical import canonical_cbor
from agentlock.crypto.hashing import blake3_hex
from agentlock.trace.context import current_recorder
from agentlock.types.trace import CallStatus, ToolCall


def trace_mcp_call(
    server: str,
    tool: str,
    input_data: dict[str, object],
    output_data: dict[str, object],
    *,
    status: CallStatus = CallStatus.SUCCESS,
    latency_ms: Optional[int] = None,
    requires_human_approval: bool = False,
    approval_present: Optional[bool] = None,
) -> None:
    """Record an MCP tool call on the current TraceRecorder.

    No-op when called outside of a ``@trace_agent_run`` context.

    Example:
        >>> trace_mcp_call("server-1", "search", {"q": "x"}, {"hits": 0})
    """
    recorder = current_recorder()
    if recorder is None:
        return
    recorder.record_tool_call(
        ToolCall(
            tool=tool,
            protocol="mcp",
            server=server,
            input_hash=blake3_hex(canonical_cbor(input_data)),
            output_hash=blake3_hex(canonical_cbor(output_data)),
            latency_ms=latency_ms,
            status=status,
            requires_human_approval=requires_human_approval,
            approval_present=approval_present,
        )
    )


__all__ = ["trace_mcp_call"]
