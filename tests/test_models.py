import pytest
from pydantic import ValidationError

from agentlock.models import ModelCall, TraceEnvelope, validate_trace_envelope, validate_trace_event


def test_validate_trace_event_returns_typed_model() -> None:
    event = validate_trace_event(
        {
            "event_type": "model.call",
            "name": "gpt-4.1-mini",
            "provider": "openai",
            "request": {"prompt": "review"},
            "response": {"decision": "approve"},
            "success": True,
        }
    )

    assert isinstance(event, ModelCall)
    assert event.name == "gpt-4.1-mini"


def test_trace_envelope_auto_populates_events() -> None:
    envelope = TraceEnvelope(
        trace_id="trace-1",
        run_id="run-1",
        agent_id="claims-agent",
        release="dev",
        input={"claim_id": "123"},
        model_calls=[ModelCall(name="gpt-4.1-mini", success=True)],
        final_output={"decision": "approve"},
        metadata={"input_hash": "abc", "output_hash": "def", "success": True},
    )

    assert envelope.events[0].event_type == "agent.run"
    assert envelope.events[-1].event_type == "run.completed"


def test_validate_trace_event_rejects_unknown_event_type() -> None:
    with pytest.raises(ValidationError):
        validate_trace_event({"event_type": "unknown"})


def test_validate_trace_envelope_normalizes_payloads() -> None:
    envelope = validate_trace_envelope(
        {
            "trace_id": "trace-2",
            "run_id": "run-2",
            "agent_id": "claims-agent",
            "release": "dev",
            "input": {"claim_id": "123"},
            "final_output": {"decision": "deny"},
            "labels": ["manual"],
            "metadata": {"success": True},
        }
    )

    assert envelope.input == {"claim_id": "123"}
    assert envelope.final_output == {"decision": "deny"}
