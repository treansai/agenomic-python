"""Context-variable based propagation of the current TraceRecorder."""
from __future__ import annotations

import contextvars
from typing import Optional

from agentlock.trace.recorder import TraceRecorder

_current_recorder: contextvars.ContextVar[Optional[TraceRecorder]] = (
    contextvars.ContextVar("agentlock_recorder", default=None)
)


def current_recorder() -> Optional[TraceRecorder]:
    """Return the active TraceRecorder, or None if no run is in progress.

    Example:
        >>> current_recorder() is None
        True
    """
    return _current_recorder.get()


def set_current_recorder(
    recorder: Optional[TraceRecorder],
) -> contextvars.Token[Optional[TraceRecorder]]:
    """Install `recorder` as the current and return a Token to reset later."""
    return _current_recorder.set(recorder)


def reset_current_recorder(
    token: contextvars.Token[Optional[TraceRecorder]],
) -> None:
    """Restore the previous recorder using a Token from set_current_recorder."""
    _current_recorder.reset(token)
