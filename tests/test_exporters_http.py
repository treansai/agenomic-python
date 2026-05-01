"""Tests for the batched HTTP exporter."""
from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from agentlock.client.client import AgentLockClient
from agentlock.exporters.http import HttpExporter
from agentlock.types.envelope import TraceEnvelope
from agentlock.types.trace import TraceInput, TraceOutput

ENDPOINT = "https://cloud.example.com"


def _env(i: int) -> TraceEnvelope:
    return TraceEnvelope(
        trace_id=f"t{i}",
        run_id=f"r{i}",
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": i}),
        final_output=TraceOutput(payload_inline={"a": i}),
    )


@pytest.mark.asyncio
async def test_flush_sends_all(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/traces", json={"accepted": 3})
    client = AgentLockClient(ENDPOINT, "k")
    exp = HttpExporter(client, batch_size=10, batch_interval_ms=0)
    for i in range(3):
        exp.export(_env(i))
    await exp.flush()
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    body = requests[0].read()
    assert body.count(b"agent://acme/demo") == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_aclose_flushes(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/traces", json={"accepted": 1})
    client = AgentLockClient(ENDPOINT, "k")
    exp = HttpExporter(client)
    exp.export(_env(0))
    await exp.aclose()
    assert len(httpx_mock.get_requests()) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_flush_empty_is_noop() -> None:
    client = AgentLockClient(ENDPOINT, "k")
    exp = HttpExporter(client)
    await exp.flush()
    await client.aclose()
