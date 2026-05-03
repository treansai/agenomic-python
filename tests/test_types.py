"""Tests for type primitives and exception hierarchy."""

from __future__ import annotations

import pytest

from agentlock.exceptions import (
    AgentLockError,
    AtepError,
    AuthenticationError,
    CloudError,
    CryptoError,
    ExportError,
    RedactionError,
    ValidationError,
)
from agentlock.types import (
    CallStatus,
    ModelCall,
    ToolCall,
    TraceEnvelope,
    TraceInput,
    TraceOutput,
    validate_agent_id,
)


def test_validate_agent_id_accepts_valid() -> None:
    assert validate_agent_id("agent://acme/claims") == "agent://acme/claims"


@pytest.mark.parametrize(
    "bad",
    [
        "claims",
        "agent://Acme/claims",
        "agent://acme",
        "http://acme/claims",
        "agent://acme/",
        "agent:///claims",
    ],
)
def test_validate_agent_id_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_agent_id(bad)


def test_envelope_round_trip() -> None:
    env = TraceEnvelope(
        trace_id="01HABC",
        run_id="01HXYZ",
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": "hello"}),
        final_output=TraceOutput(payload_inline={"a": "world"}),
        model_calls=[ModelCall(provider="openai", model="gpt-4o-mini")],
        tool_calls=[ToolCall(tool="search", protocol="mcp")],
        labels={"env": "prod"},
        metadata={"trace_source": "decorator"},
    )
    dumped = env.model_dump(mode="json")
    restored = TraceEnvelope.model_validate(dumped)
    assert restored.trace_id == env.trace_id
    assert restored.model_calls[0].provider == "openai"
    assert restored.tool_calls[0].status is CallStatus.SUCCESS
    assert restored.labels == {"env": "prod"}


@pytest.mark.parametrize(
    "exc_cls",
    [
        ValidationError,
        CryptoError,
        AtepError,
        ExportError,
        CloudError,
        AuthenticationError,
        RedactionError,
    ],
)
def test_all_exceptions_inherit_agentlock_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, AgentLockError)


def test_authentication_error_is_cloud_error() -> None:
    assert issubclass(AuthenticationError, CloudError)


def test_call_status_values() -> None:
    assert CallStatus.SUCCESS.value == "success"
    assert CallStatus.ERROR.value == "error"
    assert CallStatus.ABORTED.value == "aborted"
    assert CallStatus.TIMEOUT.value == "timeout"
