"""Async HTTP client for Agenomic Cloud."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import ulid

from agenomic._version import __version__
from agenomic.client.auth import bearer_header
from agenomic.client.retry import RetryPolicy
from agenomic.exceptions import AuthenticationError, CloudError
from agenomic.types.envelope import TraceEnvelope

logger = logging.getLogger("agenomic.client")

_RETRY_STATUS = {429, 502, 503, 504}
_VERSION_IN_NAME = re.compile(r"-(\d+\.\d+\.\d+[0-9A-Za-z.+-]*?)\.bundle")


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _envelope_to_gateway(envelope: TraceEnvelope, agent_uuid: str) -> dict[str, Any]:
    """Project a :class:`TraceEnvelope` onto the gateway ``CreateTraceRequest`` shape."""
    full = envelope.model_dump(mode="json", exclude_none=True)
    events: list[dict[str, Any]] = []
    for call in full.get("model_calls", []):
        events.append({"event_type": "model_call", "payload": call})
    for call in full.get("tool_calls", []):
        events.append({"event_type": "tool_call", "payload": call})
    events.append(
        {
            "event_type": "run_completed",
            "payload": {
                "status": "error" if envelope.error else "success",
                "duration_ms": envelope.duration_ms,
                "error": envelope.error,
            },
        }
    )
    return {
        "agent_id": agent_uuid,
        "external_id": envelope.trace_id,
        "payload": full,
        "events": events,
    }


def _bundle_version(archive_path: Path) -> str:
    match = _VERSION_IN_NAME.search(archive_path.name)
    return match.group(1) if match else archive_path.name.split(".")[0]


def _hlc_to_iso(physical_ms: int) -> str:
    moment = datetime.fromtimestamp(physical_ms / 1000.0, tz=timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


class AgenomicClient:
    """Asynchronous HTTP client for Agenomic Cloud.

    Adds an ``Idempotency-Key`` to every POST. Retries on network errors and
    on 429/502/503/504 with exponential backoff. ``Retry-After`` headers are
    honored. 401 raises :class:`AuthenticationError` immediately, 4xx (other
    than 401, 429) raise :class:`CloudError` immediately.

    Example:
        >>> # client = AgenomicClient("https://cloud.example", "key")  # doctest: +SKIP
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
        agent_ids: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._agent_ids: dict[str, str] = dict(agent_ids or {})
        # ATEP causal hash -> governance event id, kept across segment uploads
        # because AtepStore writes one event per segment.
        self._atep_event_ids: dict[bytes, str] = {}
        self._retry = retry_policy or RetryPolicy()
        headers = {
            **bearer_header(api_key),
            "User-Agent": user_agent or f"agenomic-python/{__version__}",
        }
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            transport=transport,
            verify=True,
        )

    async def whoami(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/whoami")

    async def resolve_agent_id(self, agent: str) -> str:
        """Resolve an agent reference (UUID, slug, name or ``agent://org/name``) to the gateway UUID.

        Example:
            >>> # await client.resolve_agent_id("agent://portab/portab-claims")  # doctest: +SKIP
            >>> # '1e004ebf-0a2f-4b93-ad7e-d87c91fdb385'
        """
        if _is_uuid(agent):
            return agent
        cached = self._agent_ids.get(agent)
        if cached:
            return cached
        tail = agent.rstrip("/").split("/")[-1]
        listing = await self._request("GET", "/v1/agents")
        for item in listing.get("agents", []):
            if item.get("slug") in (agent, tail) or item.get("name") in (agent, tail):
                self._agent_ids[agent] = str(item["id"])
                return self._agent_ids[agent]
        raise CloudError(f"cannot resolve agent {agent!r} to a registered agent id")

    async def upload_traces(self, envelopes: list[TraceEnvelope]) -> dict[str, Any]:
        items = [
            _envelope_to_gateway(e, await self.resolve_agent_id(e.agent_id)) for e in envelopes
        ]
        return await self._request("POST", "/v1/traces/batch", json={"items": items})

    async def upload_bundle(
        self,
        agent_id: str,
        archive_path: Path,
        *,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        archive_path = Path(archive_path)
        raw = await asyncio.to_thread(archive_path.read_bytes)
        body = {
            "agent_id": await self.resolve_agent_id(agent_id),
            "version": version or _bundle_version(archive_path),
            "hash": "",
            "metadata": metadata or {},
            "archive_base64": base64.b64encode(raw).decode("ascii"),
        }
        return await self._request("POST", "/v1/bundles", json=body)

    async def upload_atep_segment(self, agent_id: str, segment_path: Path) -> dict[str, Any]:
        """Ingest every event of an ATEP segment as a gateway governance event.

        ``header.parents`` carries causal hashes, so ``parent_event_id`` is the
        governance id the gateway returned for the parent event (the hash itself
        stays in ``metadata.parent_causal_hashes``). The hash-to-id mapping lives
        on the client, so upload segments in order with the same client to keep
        links across segments (``AtepStore`` writes one event per segment). Each
        event is posted with an ``Idempotency-Key`` derived from its causal hash, so
        re-running a partially uploaded segment replays the accepted events
        instead of creating them twice. Every causal hash and the segment Merkle
        root are recomputed before the first POST, and each event's signature is
        forwarded in ``metadata`` so the gateway can verify it.

        Example:
            >>> # await client.upload_atep_segment("agent://org/a", Path("s.atep"))  # doctest: +SKIP
        """
        from agenomic.atep.event import AtepEvent
        from agenomic.atep.segment import SegmentReader

        def _read() -> list[AtepEvent]:
            reader = SegmentReader(Path(segment_path))
            read = list(reader.iter_events())
            for e in read:
                if AtepEvent.compute_causal_hash(e.header, e.payload) != e.causal_hash:
                    raise CloudError(
                        f"segment {segment_path}: event {e.header.event_id.hex()} does not "
                        "match its causal hash; nothing uploaded"
                    )
            if not reader.verify_merkle_root():
                raise CloudError(f"segment {segment_path}: Merkle root mismatch; nothing uploaded")
            return read

        events = await asyncio.to_thread(_read)
        foreign = sorted({e.header.agent_id for e in events if e.header.agent_id != agent_id})
        if foreign:
            raise CloudError(
                f"segment {segment_path} holds events of {foreign}, not {agent_id!r}; nothing uploaded"
            )
        created_ids = self._atep_event_ids
        outcomes: list[dict[str, Any]] = []
        for event in events:
            header = event.header
            parent_id = next((created_ids[p] for p in header.parents if p in created_ids), None)
            outcome = await self._request(
                "POST",
                "/v1/governance/events",
                json={
                    "source": "atep",
                    "event_type": header.event_type,
                    "monitored_agent_id": agent_id,
                    "sequence_number": header.stream_seq,
                    "timestamp": _hlc_to_iso(header.clock.physical_ms),
                    "payload": event.payload,
                    "parent_event_id": parent_id,
                    "metadata": {
                        "event_id": header.event_id.hex(),
                        "agent_id": header.agent_id,
                        "stream": header.stream.value,
                        "causal_hash": event.causal_hash.hex(),
                        "parent_causal_hashes": [p.hex() for p in header.parents],
                        "signer_key_id": event.attestation.signer_key_id,
                        "signature": event.attestation.signature.hex(),
                        "signature_algo": event.attestation.algo,
                    },
                },
                idempotency_key=f"atep:{event.causal_hash.hex()}",
            )
            body = outcome.get("outcome")
            created = body.get("event_id") if isinstance(body, dict) else None
            if not created:
                raise CloudError(
                    f"gateway response for ATEP event {header.event_id.hex()} carries no "
                    f"outcome.event_id; stopped after {len(outcomes)} of {len(events)} events"
                )
            created_ids[event.causal_hash] = str(created)
            outcomes.append(outcome)
        return {"ingested": len(outcomes), "outcomes": outcomes}

    async def create_release(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/v1/releases", json=request)

    async def get_replay_report(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/replay-jobs/{job_id}/report")

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
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers: dict[str, str] = {}
        if method.upper() == "POST":
            headers["Idempotency-Key"] = idempotency_key or ulid.new().str

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
                raise AuthenticationError(f"authentication failed: {response.text}")
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
                raise CloudError(f"client error {response.status_code}: {response.text}")
            if response.status_code >= 500:
                raise CloudError(f"server error {response.status_code}: {response.text}")
            try:
                payload: dict[str, Any] = response.json()
            except ValueError:
                payload = {"raw": response.text}
            return payload

        raise CloudError(f"exhausted retries: {last_error}")


class SyncAgenomicClient:
    """Synchronous wrapper. Use only when async is not feasible.

    Example:
        >>> # SyncAgenomicClient("https://cloud.example", "k").whoami()  # doctest: +SKIP
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inner = AgenomicClient(*args, **kwargs)

    def whoami(self) -> dict[str, Any]:
        return asyncio.run(self._inner.whoami())

    def resolve_agent_id(self, agent: str) -> str:
        """Resolve an agent reference to the gateway UUID. See :meth:`AgenomicClient.resolve_agent_id`.

        Example:
            >>> # SyncAgenomicClient("https://cloud.example", "k").resolve_agent_id("agent://o/a")  # doctest: +SKIP
        """
        return asyncio.run(self._inner.resolve_agent_id(agent))

    def upload_traces(self, envelopes: list[TraceEnvelope]) -> dict[str, Any]:
        return asyncio.run(self._inner.upload_traces(envelopes))

    def upload_bundle(
        self,
        agent_id: str,
        archive_path: Path,
        *,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self._inner.upload_bundle(agent_id, archive_path, version=version, metadata=metadata)
        )

    def upload_atep_segment(self, agent_id: str, segment_path: Path) -> dict[str, Any]:
        return asyncio.run(self._inner.upload_atep_segment(agent_id, segment_path))

    def create_release(self, request: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._inner.create_release(request))

    def get_replay_report(self, job_id: str) -> dict[str, Any]:
        return asyncio.run(self._inner.get_replay_report(job_id))

    def close(self) -> None:
        asyncio.run(self._inner.aclose())
