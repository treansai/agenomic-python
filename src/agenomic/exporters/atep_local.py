"""ATEP local exporter — convert envelopes into signed ATEP interaction events."""

from __future__ import annotations

import logging
from typing import Optional

import ulid

from agenomic.atep.clock import Hlc
from agenomic.atep.event import AtepEvent, EventHeader, StreamId
from agenomic.atep.segment import SegmentReader
from agenomic.atep.store import AtepStore
from agenomic.crypto.signing import SigningKey
from agenomic.exporters.base import Exporter
from agenomic.types.envelope import TraceEnvelope

logger = logging.getLogger("agenomic.exporters.atep_local")


class AtepLocalExporter(Exporter):
    """Convert each trace envelope into one ATEP ``interaction`` event and
    append to a local :class:`AtepStore`.

    The event payload is the (already-redacted) trace envelope as a dict.
    Causal parent: the previous interaction event in the store. v0.1 emits a
    linear chain — DAG support arrives when integrations record multi-cause
    events.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from agenomic.atep.store import AtepStore
        >>> from agenomic.crypto.signing import SigningKey
        >>> store = AtepStore.open_or_init(Path(tempfile.mkdtemp()), "agent://a/b")
        >>> exp = AtepLocalExporter(store, SigningKey.generate())
    """

    def __init__(
        self,
        store: AtepStore,
        signing_key: SigningKey,
        node_id: int = 0,
    ) -> None:
        self.store = store
        self.signing_key = signing_key
        self.node_id = node_id
        self._stream_seq = 0
        self._last_clock = Hlc.now(node_id)
        self._last_hash: Optional[bytes] = None
        self._init_from_store()

    def _init_from_store(self) -> None:
        for path in self.store.list_segments(StreamId.INTERACTION):
            for ev in SegmentReader(path).iter_events():
                self._stream_seq = max(self._stream_seq, ev.header.stream_seq + 1)
                if ev.header.clock > self._last_clock:
                    self._last_clock = ev.header.clock
                self._last_hash = ev.causal_hash

    def export(self, envelope: TraceEnvelope) -> None:
        self._last_clock = self._last_clock.tick_after(self._last_clock)
        header = EventHeader(
            event_id=ulid.new().bytes,
            agent_id=envelope.agent_id,
            stream=StreamId.INTERACTION,
            stream_seq=self._stream_seq,
            clock=self._last_clock,
            parents=[self._last_hash] if self._last_hash else [],
            event_type="interaction.run_completed",
            payload_schema_uri="atep://schemas/v1/interaction/run_completed",
        )
        payload = envelope.model_dump(mode="json")
        event = AtepEvent.seal(header, payload, self.signing_key)
        self.store.append_event(event)
        self._stream_seq += 1
        self._last_hash = event.causal_hash
