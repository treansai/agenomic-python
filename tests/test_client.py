"""Tests for AgentLockClient."""
from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from agentlock.client.client import AgentLockClient
from agentlock.client.retry import RetryPolicy
from agentlock.exceptions import AuthenticationError, CloudError
from agentlock.types.envelope import TraceEnvelope
from agentlock.types.trace import TraceInput, TraceOutput

ENDPOINT = "https://cloud.example.com"


def _env() -> TraceEnvelope:
    return TraceEnvelope(
        trace_id="t",
        run_id="r",
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": 1}),
        final_output=TraceOutput(payload_inline={"a": 1}),
    )


@pytest.mark.asyncio
async def test_whoami_sends_auth_header(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/whoami",
        json={"id": "u1"},
    )
    client = AgentLockClient(ENDPOINT, "secret-key")
    res = await client.whoami()
    assert res == {"id": "u1"}
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["Authorization"] == "Bearer secret-key"
    await client.aclose()


@pytest.mark.asyncio
async def test_upload_traces_sends_json(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/traces", json={"accepted": 1}
    )
    client = AgentLockClient(ENDPOINT, "k")
    res = await client.upload_traces([_env()])
    assert res == {"accepted": 1}
    req = httpx_mock.get_request()
    assert req is not None
    body = req.read()
    assert b"agent://acme/demo" in body
    await client.aclose()


@pytest.mark.asyncio
async def test_idempotency_key_unique(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/traces", json={}, is_reusable=True)
    client = AgentLockClient(ENDPOINT, "k")
    await client.upload_traces([_env()])
    await client.upload_traces([_env()])
    requests = httpx_mock.get_requests()
    keys = [r.headers["Idempotency-Key"] for r in requests]
    assert len(set(keys)) == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_429_with_retry_after(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/whoami",
        status_code=429,
        headers={"Retry-After": "0"},
    )
    httpx_mock.add_response(url=f"{ENDPOINT}/v1/whoami", json={"id": "u"})
    client = AgentLockClient(
        ENDPOINT, "k", retry_policy=RetryPolicy(max_retries=2, base_delay=0)
    )
    res = await client.whoami()
    assert res == {"id": "u"}
    await client.aclose()


@pytest.mark.asyncio
async def test_401_no_retry(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/whoami", status_code=401, text="bad token"
    )
    client = AgentLockClient(ENDPOINT, "k")
    with pytest.raises(AuthenticationError):
        await client.whoami()
    assert len(httpx_mock.get_requests()) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_503_retried_then_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/whoami",
        status_code=503,
        text="x",
        is_reusable=True,
    )
    client = AgentLockClient(
        ENDPOINT, "k", retry_policy=RetryPolicy(max_retries=2, base_delay=0)
    )
    with pytest.raises(CloudError):
        await client.whoami()
    # 1 initial + 2 retries
    assert len(httpx_mock.get_requests()) == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_400_no_retry(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{ENDPOINT}/v1/whoami", status_code=400, text="bad"
    )
    client = AgentLockClient(ENDPOINT, "k")
    with pytest.raises(CloudError):
        await client.whoami()
    assert len(httpx_mock.get_requests()) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_user_agent_default() -> None:
    from agentlock._version import __version__
    client = AgentLockClient(ENDPOINT, "k")
    ua = client._client.headers["User-Agent"]
    assert __version__ in ua
    await client.aclose()
