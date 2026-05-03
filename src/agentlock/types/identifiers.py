"""Typed identifiers used across the SDK."""

from __future__ import annotations

import re
from typing import NewType

AgentId = NewType("AgentId", str)
TraceId = NewType("TraceId", str)
RunId = NewType("RunId", str)
ReleaseId = NewType("ReleaseId", str)

_AGENT_ID_RE = re.compile(r"^agent://[a-z0-9-]+/[a-z0-9-]+$")


def validate_agent_id(value: str) -> AgentId:
    """Validate and convert a string to an AgentId.

    Example:
        >>> validate_agent_id("agent://acme/claims")
        'agent://acme/claims'
    """
    if not _AGENT_ID_RE.match(value):
        raise ValueError(f"invalid agent_id: {value!r} (must match agent://org/name)")
    return AgentId(value)
