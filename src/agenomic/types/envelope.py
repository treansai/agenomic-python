"""Trace envelope — single agent run wire format."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from agenomic.types.trace import ModelCall, ToolCall, TraceInput, TraceOutput


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceEnvelope(BaseModel):
    """Single agent run envelope.

    Matches `agenomic-spec` `trace-event.schema.json` (v0.1).

    Example:
        >>> env = TraceEnvelope(
        ...     trace_id="01H",
        ...     run_id="01H",
        ...     agent_id="agent://acme/demo",
        ...     input=TraceInput(payload_inline={"q": "hi"}),
        ...     final_output=TraceOutput(payload_inline={"a": "ok"}),
        ... )
        >>> env.schema_version
        'agenomic-trace/v0.1'
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = "agenomic-trace/v0.1"
    trace_id: str
    run_id: str
    agent_id: str
    release: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utc_now)
    input: TraceInput
    model_calls: list[ModelCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_output: TraceOutput
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: Optional[int] = None
