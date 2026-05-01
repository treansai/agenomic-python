"""Multi-exporter — fan out each envelope to several exporters."""

from __future__ import annotations

import logging

from agentlock.exporters.base import Exporter
from agentlock.types.envelope import TraceEnvelope

logger = logging.getLogger("agentlock.exporters.multi")


class MultiExporter(Exporter):
    """Fan out each envelope to multiple exporters.

    Errors in any single exporter are logged but do NOT prevent other
    exporters from receiving the envelope.

    Example:
        >>> from agentlock.exporters.jsonl import JsonlExporter
        >>> import tempfile
        >>> a = JsonlExporter(tempfile.mktemp(suffix=".jsonl"))
        >>> b = JsonlExporter(tempfile.mktemp(suffix=".jsonl"))
        >>> m = MultiExporter(a, b)
    """

    def __init__(self, *exporters: Exporter) -> None:
        self.exporters: tuple[Exporter, ...] = exporters

    def export(self, envelope: TraceEnvelope) -> None:
        for exp in self.exporters:
            try:
                exp.export(envelope)
            except Exception:
                logger.exception("exporter %r raised", exp)

    def close(self) -> None:
        for exp in self.exporters:
            try:
                exp.close()
            except Exception:
                logger.exception("close on %r raised", exp)
