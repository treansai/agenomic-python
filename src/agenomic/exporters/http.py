"""HTTP exporter — async batched delivery to Agenomic Cloud."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from agenomic.exporters.base import Exporter
from agenomic.types.envelope import TraceEnvelope

if TYPE_CHECKING:
    from agenomic.client.client import AgenomicClient

logger = logging.getLogger("agenomic.exporters.http")


class HttpExporter(Exporter):
    """Asynchronous HTTP exporter with batching.

    Buffers envelopes and forwards via :class:`AgenomicClient.upload_traces`
    when the buffer hits ``batch_size`` or ``batch_interval_ms`` elapses,
    whichever comes first. ``close()`` flushes the queue and waits for
    in-flight requests.

    Example:
        >>> from agenomic.client.client import AgenomicClient  # doctest: +SKIP
        >>> client = AgenomicClient("https://cloud.example", "key")  # doctest: +SKIP
        >>> exp = HttpExporter(client)  # doctest: +SKIP
    """

    def __init__(
        self,
        client: AgenomicClient,
        *,
        batch_size: int = 100,
        batch_interval_ms: int = 5000,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.batch_interval = batch_interval_ms / 1000.0
        self._queue: list[TraceEnvelope] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._closed = False

    def export(self, envelope: TraceEnvelope) -> None:
        """Non-blocking enqueue.

        If a running event loop exists, schedule a flush check; otherwise the
        envelope sits until ``await flush()`` or ``aclose()`` is called.
        """
        if self._closed:
            raise RuntimeError("HttpExporter is closed")
        self._queue.append(envelope)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if len(self._queue) >= self.batch_size:
            loop.create_task(self.flush())

    async def flush(self) -> None:
        async with self._lock:
            if not self._queue:
                return
            batch, self._queue = self._queue, []
        try:
            await self.client.upload_traces(batch)
        except Exception:
            logger.exception("upload_traces failed for %d envelopes", len(batch))

    async def aclose(self) -> None:
        self._closed = True
        await self.flush()

    def close(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        # Inside an event loop — schedule the flush; caller should await it.
        asyncio.ensure_future(self.aclose())
