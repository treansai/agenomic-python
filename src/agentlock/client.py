from __future__ import annotations

from typing import Any

from agentlock.exporters.http import HTTPTraceExporter
from agentlock.exporters.jsonl import JSONLTraceExporter
from agentlock.models import TraceEnvelope, validate_trace_envelope


class AgentLockClient:
    def __init__(self, api_key: str | None = None, endpoint: str | None = None) -> None:
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.local_traces: list[TraceEnvelope] = []
        self._jsonl_exporter = JSONLTraceExporter()
        self._http_exporter = (
            HTTPTraceExporter(endpoint=self.endpoint, api_key=self.api_key)
            if self.endpoint is not None
            else None
        )

    def emit_trace(self, trace: TraceEnvelope | dict[str, Any]) -> Any:
        envelope = validate_trace_envelope(trace)
        self.local_traces.append(envelope)
        if self._http_exporter is None:
            return None
        return self._http_exporter.export(envelope)

    async def emit_trace_async(self, trace: TraceEnvelope | dict[str, Any]) -> Any:
        envelope = validate_trace_envelope(trace)
        self.local_traces.append(envelope)
        if self._http_exporter is None:
            return None
        return await self._http_exporter.export_async(envelope)

    def export_jsonl(self, path: str, traces: list[TraceEnvelope]) -> str:
        return str(self._jsonl_exporter.export(path=path, traces=traces))
