"""Trace primitives — model calls, tool calls, input/output payloads."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class CallStatus(str, Enum):
    """Outcome of a model or tool call."""

    SUCCESS = "success"
    ERROR = "error"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


class ModelCall(BaseModel):
    """One LLM invocation captured during an agent run.

    Example:
        >>> ModelCall(provider="openai", model="gpt-4o-mini").status
        <CallStatus.SUCCESS: 'success'>
    """

    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    fingerprint: Optional[str] = None
    temperature: Optional[float] = None
    prompt_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: Optional[int] = None
    cost_estimate: Optional[float] = None
    status: CallStatus = CallStatus.SUCCESS


class ToolCall(BaseModel):
    """One tool invocation (MCP, HTTP, gRPC, local) captured during a run.

    Example:
        >>> ToolCall(tool="search", protocol="mcp").requires_human_approval
        False
    """

    model_config = ConfigDict(extra="allow")

    tool: str
    protocol: str
    server: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: Optional[int] = None
    status: CallStatus = CallStatus.SUCCESS
    requires_human_approval: bool = False
    approval_present: Optional[bool] = None


class TraceInput(BaseModel):
    """Input payload to an agent run, inline or by reference.

    Example:
        >>> TraceInput(payload_inline={"q": "hello"}).type
        'json'
    """

    model_config = ConfigDict(extra="allow")

    type: str = "json"
    payload_inline: Optional[Any] = None
    payload_ref: Optional[str] = None


class TraceOutput(BaseModel):
    """Output payload of an agent run, inline or by reference.

    Example:
        >>> TraceOutput(payload_inline={"answer": 42}).hash is None
        True
    """

    model_config = ConfigDict(extra="allow")

    hash: Optional[str] = None
    payload_inline: Optional[Any] = None
    payload_ref: Optional[str] = None
