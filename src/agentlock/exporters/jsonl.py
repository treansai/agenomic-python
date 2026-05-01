"""JSONL exporter — one trace envelope per line, append-only."""
from __future__ import annotations

from pathlib import Path
from typing import IO, Optional

from agentlock.exporters.base import Exporter
from agentlock.types.envelope import TraceEnvelope


class JsonlExporter(Exporter):
    """Append-only JSONL file. One :class:`TraceEnvelope` per line.

    Example:
        >>> import tempfile, json
        >>> path = tempfile.mktemp(suffix=".jsonl")
        >>> from agentlock.types.envelope import TraceEnvelope
        >>> from agentlock.types.trace import TraceInput, TraceOutput
        >>> with JsonlExporter(path) as exp:
        ...     exp.export(TraceEnvelope(
        ...         trace_id="t", run_id="r", agent_id="agent://a/b",
        ...         input=TraceInput(payload_inline={"q": "hi"}),
        ...         final_output=TraceOutput(payload_inline={"a": "ok"}),
        ...     ))
        >>> json.loads(open(path).readline())["agent_id"]
        'agent://a/b'
    """

    def __init__(self, path: Path | str, *, flush_each: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: Optional[IO[str]] = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self._flush_each = flush_each

    def export(self, envelope: TraceEnvelope) -> None:
        if self._fp is None:
            raise RuntimeError("JsonlExporter is closed")
        line = envelope.model_dump_json(exclude_none=True)
        self._fp.write(line + "\n")
        if self._flush_each:
            self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
