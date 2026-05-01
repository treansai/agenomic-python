"""Trace decorator + recorder + envelope builder."""

from agentlock.trace.context import (
    current_recorder,
    reset_current_recorder,
    set_current_recorder,
)
from agentlock.trace.decorator import trace_agent_run
from agentlock.trace.envelope_builder import build_envelope
from agentlock.trace.recorder import TraceRecorder

__all__ = [
    "TraceRecorder",
    "build_envelope",
    "current_recorder",
    "reset_current_recorder",
    "set_current_recorder",
    "trace_agent_run",
]
