from __future__ import annotations

from typing import Any

import httpx

from agentlock.models import TraceEnvelope, validate_trace_envelope


class HTTPTraceExporter:
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not endpoint:
            msg = "endpoint is required for HTTP export"
            raise ValueError(msg)
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport
        self.async_transport = async_transport

    def export(self, trace: TraceEnvelope | dict[str, Any]) -> httpx.Response:
        envelope = validate_trace_envelope(trace)
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.post(
                self.endpoint,
                headers=self._headers(),
                json=envelope.model_dump(mode="json"),
            )
            response.raise_for_status()
            return response

    async def export_async(self, trace: TraceEnvelope | dict[str, Any]) -> httpx.Response:
        envelope = validate_trace_envelope(trace)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.async_transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                headers=self._headers(),
                json=envelope.model_dump(mode="json"),
            )
            response.raise_for_status()
            return response

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers
