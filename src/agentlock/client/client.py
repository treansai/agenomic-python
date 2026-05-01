"""Async HTTP client for AgentLock Cloud."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
import ulid

from agentlock._version import __version__
from agentlock.client.auth import bearer_header
from agentlock.client.retry import RetryPolicy
from agentlock.exceptions import AuthenticationError, CloudError
from agentlock.types.envelope import TraceEnvelope

logger = logging.getLogger("agentlock.client")

_RETRY_STATUS = {429, 502, 503, 504}


class AgentLockClient:
    """Asynchronous HTTP client for AgentLock Cloud.

    Adds an ``Idempotency-Key`` to every POST. Retries on network errors and
    on 429/502/503/504 with exponential backoff. ``Retry-After`` headers are
    honored. 401 raises :class:`AuthenticationError` immediately, 4xx (other
    than 401, 429) raise :class:`CloudError` immediately.

    Example:
        >>> # client = AgentLockClient("https://cloud.example", "key")  # doctest: +SKIP
        >>> # await client.whoami()  # doctest: +SKIP
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        retry_policy: Optional[RetryPolicy] = None,
        user_agent: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._retry = retry_policy or RetryPolicy()
        headers = {
            **bearer_header(api_key),
            "User-Agent": user_agent or f"agentlock-python/{__version__}",
        }
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            transport=transport,
            verify=True,
        )

    async def whoami(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/whoami")

    async def upload_traces(
        self, envelopes: list[TraceEnvelope]
    ) -> dict[str, Any]:
        body = {
            "traces": [e.model_dump(mode="json", exclude_none=True) for e in envelopes],
        }
        return await self._request("POST", "/v1/traces", json=body)

    async def upload_bundle(
        self, agent_id: str, archive_path: Path
    ) -> dict[str, Any]:
        archive_path = Path(archive_path)
        body = await asyncio.to_thread(archive_path.read_bytes)
        files = {"bundle": (archive_path.name, body, "application/octet-stream")}
        data = {"agent_id": agent_id}
        return await self._request("POST", "/v1/bundles", data=data, files=files)

    async def upload_atep_segment(
        self, agent_id: str, segment_path: Path
    ) -> dict[str, Any]:
        segment_path = Path(segment_path)
        body = await asyncio.to_thread(segment_path.read_bytes)
        files = {"segment": (segment_path.name, body, "application/octet-stream")}
        data = {"agent_id": agent_id}
        return await self._request("POST", "/v1/atep/segments", data=data, files=files)

    async def create_release(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/v1/releases", json=request)

    async def get_replay_report(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/replays/{job_id}")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers: dict[str, str] = {}
        if method.upper() == "POST":
            headers["Idempotency-Key"] = ulid.new().str

        last_error: Optional[Exception] = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    headers=headers,
                )
            except httpx.HTTPError as e:
                last_error = e
                if attempt >= self._retry.max_retries:
                    raise CloudError(f"network error after retries: {e}") from e
                await asyncio.sleep(self._retry.delay_for(attempt))
                continue

            if response.status_code == 401:
                raise AuthenticationError(
                    f"authentication failed: {response.text}"
                )
            if response.status_code in _RETRY_STATUS:
                if attempt >= self._retry.max_retries:
                    raise CloudError(
                        f"giving up after {attempt + 1} attempts: "
                        f"{response.status_code} {response.text}"
                    )
                ra = response.headers.get("Retry-After")
                delay = float(ra) if ra else self._retry.delay_for(attempt)
                await asyncio.sleep(delay)
                continue
            if 400 <= response.status_code < 500:
                raise CloudError(
                    f"client error {response.status_code}: {response.text}"
                )
            if response.status_code >= 500:
                raise CloudError(
                    f"server error {response.status_code}: {response.text}"
                )
            try:
                payload: dict[str, Any] = response.json()
            except ValueError:
                payload = {"raw": response.text}
            return payload

        raise CloudError(f"exhausted retries: {last_error}")


class SyncAgentLockClient:
    """Synchronous wrapper. Use only when async is not feasible.

    Example:
        >>> # SyncAgentLockClient("https://cloud.example", "k").whoami()  # doctest: +SKIP
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inner = AgentLockClient(*args, **kwargs)

    def whoami(self) -> dict[str, Any]:
        return asyncio.run(self._inner.whoami())

    def upload_traces(self, envelopes: list[TraceEnvelope]) -> dict[str, Any]:
        return asyncio.run(self._inner.upload_traces(envelopes))

    def upload_bundle(self, agent_id: str, archive_path: Path) -> dict[str, Any]:
        return asyncio.run(self._inner.upload_bundle(agent_id, archive_path))

    def upload_atep_segment(self, agent_id: str, segment_path: Path) -> dict[str, Any]:
        return asyncio.run(self._inner.upload_atep_segment(agent_id, segment_path))

    def create_release(self, request: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._inner.create_release(request))

    def get_replay_report(self, job_id: str) -> dict[str, Any]:
        return asyncio.run(self._inner.get_replay_report(job_id))

    def close(self) -> None:
        asyncio.run(self._inner.aclose())
