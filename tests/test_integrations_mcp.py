"""Tests for MCP helper."""

from __future__ import annotations

from agentlock.integrations.mcp import trace_mcp_call
from agentlock.trace.context import set_current_recorder
from agentlock.trace.recorder import TraceRecorder
from agentlock.types.trace import CallStatus


def test_no_recorder_is_noop() -> None:
    # Must not raise.
    trace_mcp_call("server", "tool", {"x": 1}, {"y": 2})


def test_records_tool_call() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        trace_mcp_call("s1", "search", {"q": "x"}, {"hits": 0})
    finally:
        set_current_recorder(None)
    assert len(rec.tool_calls) == 1
    tc = rec.tool_calls[0]
    assert tc.protocol == "mcp"
    assert tc.server == "s1"
    assert tc.tool == "search"
    assert tc.status is CallStatus.SUCCESS
    assert tc.input_hash
    assert tc.output_hash
