"""Tests for the online-tracking SDK instrumentation."""

from __future__ import annotations

import json

import pytest

from agenomic import Client
from agenomic.tracking import TrackingSession


def test_local_mode_buffers_spec_shaped_events() -> None:
    client = Client()  # no base_url => local mode
    assert not client.is_cloud
    session = client.tracking.start(
        agent="agent://treans/claims-agent",
        release_id="release_123",
        environment="production",
    )
    assert isinstance(session, TrackingSession)

    with session.step("classify_claim"):
        session.model_call(provider="openai", model="gpt-4o", input_hash="blake3:a")
        session.tool_call(tool="classify_claim", input_hash="blake3:b")
        session.intent("verify_claim_validity")
        session.memory_write(schema_version="1.0.0")
    session.stop()

    events = session.events
    assert [e["type"] for e in events] == [
        "agent.step.started",
        "model.call.completed",
        "tool.call.completed",
        "intent.detected",
        "memory.write",
        "agent.step.completed",
    ]
    assert events[0]["spec_version"] == "agenomic/v0.3"
    assert all(e["agent_id"] == "agent://treans/claims-agent" for e in events)
    assert [e["sequence_number"] for e in events] == [0, 1, 2, 3, 4, 5]
    # tool / model metadata is nested in the spec snake_case shape
    assert events[1]["model"] == {"provider": "openai", "model": "gpt-4o"}
    assert events[2]["tool"] == {"name": "classify_claim"}

    # JSONL export round-trips
    lines = [json.loads(line) for line in session.to_jsonl().splitlines()]
    assert len(lines) == 6


def test_step_emits_agent_failed_on_exception() -> None:
    client = Client()
    session = client.tracking.start(agent="agent://acme/a")
    with pytest.raises(ValueError), session.step("classify"):
        raise ValueError("boom")
    assert session.events[-1]["type"] == "agent.failed"
    assert session.events[-1]["metadata"] == {"status": "error"}


def test_unknown_event_type_is_rejected() -> None:
    client = Client()
    session = client.tracking.start(agent="agent://acme/a")
    with pytest.raises(ValueError):
        session.event("frobnicate.everything")


def test_refuses_events_after_stop() -> None:
    client = Client()
    session = client.tracking.start(agent="agent://acme/a")
    session.stop()
    with pytest.raises(RuntimeError):
        session.intent("x")


def test_context_manager_stops_session() -> None:
    client = Client()
    with client.tracking.start(agent="agent://acme/a") as session:
        session.intent("answer")
    assert session._stopped is True  # noqa: SLF001


def test_report_requires_cloud_mode() -> None:
    client = Client()
    session = client.tracking.start(agent="agent://acme/a")
    with pytest.raises(RuntimeError):
        session.report()


def test_cloud_mode_posts_start_event_stop(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions",
        json={"session": {"session_id": "sess_1"}},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions/sess_1/events",
        json={},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions/sess_1/stop",
        json={},
    )

    client = Client(api_key="key_123", base_url="https://api.test")
    assert client.is_cloud
    session = client.tracking.start(agent="agent://treans/claims-agent", release_id="r1")
    assert session.session_id == "sess_1"

    session.tool_call(tool="claims_db.lookup", input_hash="blake3:x")
    session.stop()

    requests = httpx_mock.get_requests()
    assert len(requests) == 3
    assert requests[0].headers["Authorization"] == "Bearer key_123"
    event_body = json.loads(requests[1].content)
    assert event_body["type"] == "tool.call.completed"
    assert event_body["tool"]["name"] == "claims_db.lookup"
    # cloud mode does not buffer locally
    assert session.events == []


def test_cloud_error_is_wrapped(httpx_mock) -> None:
    from agenomic.exceptions import CloudError

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions",
        status_code=500,
    )
    client = Client(base_url="https://api.test")
    with pytest.raises(CloudError):
        client.tracking.start(agent="agent://a/b")


def test_cloud_start_without_session_id_raises(httpx_mock) -> None:
    from agenomic.exceptions import CloudError

    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions",
        json={"session": {}},
    )
    client = Client(base_url="https://api.test")
    with pytest.raises(CloudError):
        client.tracking.start(agent="agent://a/b")


def test_cloud_stop_stays_retryable_on_failure(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions",
        json={"session": {"session_id": "s1"}},
    )
    # first stop fails, second succeeds
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions/s1/stop",
        status_code=503,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/tracking/sessions/s1/stop",
        json={},
    )
    client = Client(base_url="https://api.test")
    session = client.tracking.start(agent="agent://a/b")
    with pytest.raises(Exception):
        session.stop()
    # not marked stopped → retry issues another request and succeeds
    session.stop()
