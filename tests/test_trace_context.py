"""Tests for trace context propagation."""
from __future__ import annotations

from agentlock.trace.context import (
    current_recorder,
    reset_current_recorder,
    set_current_recorder,
)
from agentlock.trace.recorder import TraceRecorder


def test_default_is_none() -> None:
    assert current_recorder() is None


def test_set_and_reset() -> None:
    rec = TraceRecorder("agent://a/b", "r", "t")
    token = set_current_recorder(rec)
    assert current_recorder() is rec
    reset_current_recorder(token)
    assert current_recorder() is None
