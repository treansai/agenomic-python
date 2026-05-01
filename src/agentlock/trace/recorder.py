"""In-memory accumulator for one agent run."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentlock.types.trace import ModelCall, ToolCall


class TraceRecorder:
    """Accumulates model_calls, tool_calls and metadata during one agent run.

    Example:
        >>> r = TraceRecorder("agent://a/b", "run1", "trace1")
        >>> r.add_label("env", "test")
        >>> r.labels["env"]
        'test'
    """

    def __init__(self, agent_id: str, run_id: str, trace_id: str) -> None:
        self.agent_id = agent_id
        self.run_id = run_id
        self.trace_id = trace_id
        self.model_calls: list[ModelCall] = []
        self.tool_calls: list[ToolCall] = []
        self.labels: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.started_at = datetime.now(timezone.utc)

    def record_model_call(self, call: ModelCall) -> None:
        self.model_calls.append(call)

    def record_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def add_label(self, key: str, value: str) -> None:
        self.labels[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
