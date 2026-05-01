"""Helper that assembles a TraceEnvelope from a recorder + final output."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agentlock.redaction.engine import RedactionEngine
from agentlock.trace.recorder import TraceRecorder
from agentlock.types.envelope import TraceEnvelope
from agentlock.types.trace import TraceInput, TraceOutput


def build_envelope(
    recorder: TraceRecorder,
    *,
    raw_input: Any,
    raw_output: Any,
    error: Optional[str],
    release: Optional[str],
    capture_input: bool,
    capture_output: bool,
    redaction: Optional[RedactionEngine],
) -> TraceEnvelope:
    """Materialize a TraceEnvelope, applying redaction and capture flags last."""
    if capture_input:
        payload_in = redaction.apply(raw_input) if redaction is not None else raw_input
        trace_input = TraceInput(payload_inline=payload_in)
    else:
        trace_input = TraceInput(payload_inline={"redacted": True})

    if capture_output:
        payload_out = (
            redaction.apply(raw_output) if redaction is not None else raw_output
        )
        trace_output = TraceOutput(payload_inline=payload_out)
    else:
        trace_output = TraceOutput(payload_inline={"redacted": True})

    finished_at = datetime.now(timezone.utc)
    duration_ms = int((finished_at - recorder.started_at).total_seconds() * 1000)

    return TraceEnvelope(
        trace_id=recorder.trace_id,
        run_id=recorder.run_id,
        agent_id=recorder.agent_id,
        release=release,
        timestamp=recorder.started_at,
        input=trace_input,
        model_calls=list(recorder.model_calls),
        tool_calls=list(recorder.tool_calls),
        final_output=trace_output,
        labels=dict(recorder.labels),
        metadata=dict(recorder.metadata),
        error=error,
        duration_ms=duration_ms,
    )
