from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentlock.models import TraceEnvelope, validate_trace_envelope


class JSONLTraceExporter:
    def export(self, path: str | Path, traces: Sequence[TraceEnvelope | dict[str, Any]]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                envelope = validate_trace_envelope(trace)
                handle.write(envelope.model_dump_json())
                handle.write("\n")

        return output_path
