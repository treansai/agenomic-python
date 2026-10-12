"""Wire-format conformance of AgenomicClient against the gateway DTOs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from agenomic.client.client import AgenomicClient
from agenomic.exceptions import CloudError
from agenomic.tracking import TrackingResource
from agenomic.types.envelope import TraceEnvelope
from agenomic.types.trace import ModelCall, ToolCall, TraceInput, TraceOutput

ENDPOINT = "http://127.0.0.1:8090"
AGENT_UUID = "1e004ebf-0a2f-4b93-ad7e-d87c91fdb385"
AGENTS_LISTING = {
    "agents": [
        {"id": "2f1b3c52-0000-4000-8000-000000000001", "name": "other", "slug": "other-aa11"},
        {"id": AGENT_UUID, "name": "portab-claims", "slug": "portab-claims-03cca6"},
    ]
}


def _env() -> TraceEnvelope:
    return TraceEnvelope(
        trace_id="01TRACE",
        run_id="01RUN",
        agent_id="agent://portab/portab-claims",
        release="v1.1.0",
        input=TraceInput(payload_inline={"q": "hi"}),
        final_output=TraceOutput(payload_inline={"a": "HI"}),
        model_calls=[ModelCall(provider="fake", model="fake-1")],
        tool_calls=[ToolCall(tool="claims.policy.evaluate", protocol="local")],
    )


@pytest.mark.asyncio
async def test_upload_traces_matches_batch_create_trace_request(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/agents", json=AGENTS_LISTING)
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/traces/batch", status_code=201, json={"traces": []})
    client = AgenomicClient(ENDPOINT, "k")
    await client.upload_traces([_env(), _env()])
    await client.aclose()

    requests = httpx_mock.get_requests()
    assert [r.url.path for r in requests] == ["/v1/agents", "/v1/traces/batch"]
    body = json.loads(requests[1].read())
    assert set(body) == {"items"}
    first = body["items"][0]
    assert set(first) == {"agent_id", "external_id", "payload", "events"}
    assert first["agent_id"] == AGENT_UUID
    assert first["external_id"] == "01TRACE"
    assert first["payload"]["agent_id"] == "agent://portab/portab-claims"
    assert [e["event_type"] for e in first["events"]] == [
        "model_call",
        "tool_call",
        "run_completed",
    ]
    assert all(set(e) == {"event_type", "payload"} for e in first["events"])


@pytest.mark.asyncio
async def test_unknown_agent_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/agents", json={"agents": []})
    client = AgenomicClient(ENDPOINT, "k")
    with pytest.raises(CloudError):
        await client.upload_traces([_env()])
    await client.aclose()


@pytest.mark.asyncio
async def test_uuid_agent_skips_lookup(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/traces/batch", json={"traces": []})
    client = AgenomicClient(ENDPOINT, "k")
    env = _env()
    env.agent_id = AGENT_UUID
    await client.upload_traces([env])
    await client.aclose()
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_upload_bundle_sends_json_create_bundle_request(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    archive = tmp_path / "portab-1.0.0.bundle.tar.zst"
    archive.write_bytes(b"\x28\xb5\x2f\xfd-archive")
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/bundles", status_code=201, json={"bundle": {"id": "b1"}}
    )
    client = AgenomicClient(ENDPOINT, "k")
    res = await client.upload_bundle(AGENT_UUID, archive)
    await client.aclose()
    assert res == {"bundle": {"id": "b1"}}
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.read())
    assert body["agent_id"] == AGENT_UUID
    assert body["version"] == "1.0.0"
    assert body["hash"] == ""
    assert body["metadata"] == {}
    assert base64.b64decode(body["archive_base64"]) == archive.read_bytes()


@pytest.mark.asyncio
async def test_replay_report_uses_replay_jobs_route(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/replay-jobs/job-1/report", json={"report": {}})
    client = AgenomicClient(ENDPOINT, "k")
    assert await client.get_replay_report("job-1") == {"report": {}}
    await client.aclose()


@pytest.mark.asyncio
async def test_atep_segment_ingested_as_governance_events(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    import ulid

    from agenomic.atep.clock import Hlc
    from agenomic.atep.event import AtepEvent, EventHeader, StreamId
    from agenomic.atep.segment import SegmentWriter
    from agenomic.crypto.signing import SigningKey

    key = SigningKey.generate()
    path = tmp_path / "s.atep"
    hashes: list[bytes] = []
    with SegmentWriter(path) as writer:
        for seq in range(2):
            header = EventHeader(
                event_id=ulid.new().bytes,
                agent_id="agent://a/b",
                stream=StreamId.RUNTIME,
                stream_seq=seq,
                clock=Hlc(1_760_000_000_000 + seq, 0, 0),
                event_type="runtime.step",
                payload_schema_uri="atep://schemas/v1/step",
                parents=hashes[-1:],
            )
            event = AtepEvent.seal(header, {"n": seq}, key)
            hashes.append(event.causal_hash)
            writer.append(event)
    for seq in range(2):
        httpx_mock.add_response(
            url=f"{ENDPOINT}/v1/governance/events",
            status_code=202,
            json={"outcome": {"event_id": f"gov-{seq}"}},
        )
    client = AgenomicClient(ENDPOINT, "k")
    res = await client.upload_atep_segment("agent://a/b", path)
    await client.aclose()
    assert res["ingested"] == 2
    bodies = [json.loads(r.read()) for r in httpx_mock.get_requests()]
    assert [b["sequence_number"] for b in bodies] == [0, 1]
    assert all(b["source"] == "atep" and b["event_type"] == "runtime.step" for b in bodies)
    assert bodies[0]["timestamp"].startswith("2025-")
    assert bodies[0]["monitored_agent_id"] == "agent://a/b"
    # header.parents holds causal hashes; the gateway id of the parent is sent instead.
    assert bodies[0]["parent_event_id"] is None
    assert bodies[1]["parent_event_id"] == "gov-0"
    assert bodies[1]["metadata"]["parent_causal_hashes"] == [hashes[0].hex()]
    keys = [r.headers["Idempotency-Key"] for r in httpx_mock.get_requests()]
    assert keys == [f"atep:{h.hex()}" for h in hashes]


@pytest.mark.asyncio
async def test_atep_segment_retry_reuses_idempotency_keys(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    import ulid

    from agenomic.atep.clock import Hlc
    from agenomic.atep.event import AtepEvent, EventHeader, StreamId
    from agenomic.atep.segment import SegmentWriter
    from agenomic.crypto.signing import SigningKey

    key = SigningKey.generate()
    path = tmp_path / "s.atep"
    with SegmentWriter(path) as writer:
        for seq in range(2):
            header = EventHeader(
                event_id=ulid.new().bytes,
                agent_id="agent://a/b",
                stream=StreamId.RUNTIME,
                stream_seq=seq,
                clock=Hlc(1_760_000_000_000 + seq, 0, 0),
                event_type="runtime.step",
                payload_schema_uri="atep://schemas/v1/step",
            )
            writer.append(AtepEvent.seal(header, {"n": seq}, key))
    url = f"{ENDPOINT}/v1/governance/events"
    httpx_mock.add_response(url=url, status_code=202, json={"outcome": {"event_id": "a"}})
    httpx_mock.add_response(url=url, status_code=400, json={"error": "bad"})
    httpx_mock.add_response(url=url, status_code=202, json={"outcome": {"event_id": "a"}})
    httpx_mock.add_response(url=url, status_code=202, json={"outcome": {"event_id": "b"}})
    client = AgenomicClient(ENDPOINT, "k")
    with pytest.raises(CloudError):
        await client.upload_atep_segment("agent://a/b", path)
    res = await client.upload_atep_segment("agent://a/b", path)
    await client.aclose()
    assert res["ingested"] == 2
    keys = [r.headers["Idempotency-Key"] for r in httpx_mock.get_requests()]
    assert keys[:2] == keys[2:]


class _RecordingCloud:
    is_cloud = True

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def _post(self, path: str, body: dict) -> dict:
        self.posts.append((path, body))
        return {"session": {"session_id": "01SESSION", "id": "uuid"}}


def test_tracking_start_sends_baseline_and_parent() -> None:
    cloud = _RecordingCloud()
    TrackingResource(cloud).start(
        agent="agent://a/b",
        allowed_tools=["search"],
        baseline={"model_id": "m", "allowed_tools": ["ignored"]},
        parent_session_id="01PARENT",
    )
    path, body = cloud.posts[0]
    assert path == "/v1/tracking/sessions"
    assert body["baseline"] == {"model_id": "m", "allowed_tools": ["search"]}
    assert body["parent_session_id"] == "01PARENT"


def test_tracking_start_omits_optional_fields_by_default() -> None:
    cloud = _RecordingCloud()
    TrackingResource(cloud).start(agent="agent://a/b")
    _, body = cloud.posts[0]
    assert "baseline" not in body
    assert "parent_session_id" not in body
