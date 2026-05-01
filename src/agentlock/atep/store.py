"""Filesystem-backed ATEP store for one agent."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from agentlock.atep.event import AtepEvent, StreamId
from agentlock.atep.segment import SegmentReader, SegmentWriter
from agentlock.crypto.hashing import blake3_bytes
from agentlock.crypto.signing import VerifyingKey
from agentlock.exceptions import AtepError

MANIFEST_NAME = "manifest.json"
STREAMS_DIR = "streams"


@dataclass
class VerificationReport:
    """Outcome of :meth:`AtepStore.verify_all`."""

    ok: bool
    segments_checked: int = 0
    events_checked: int = 0
    failures: list[str] = field(default_factory=list)


class AtepStore:
    """Directory-backed ATEP store for one agent.

    Layout::

        root/
          manifest.json
          streams/
            identity-00000001.atep
            capability-00000001.atep
            interaction-00000001.atep
            ...

    For v0.1 each :meth:`append_event` creates a one-event segment. Multi-event
    segments and compaction land in v0.2.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> root = Path(tempfile.mkdtemp())
        >>> store = AtepStore.open_or_init(root, "agent://acme/demo")
        >>> store.agent_id
        'agent://acme/demo'
    """

    def __init__(self, root: Path, agent_id: str) -> None:
        self.root = Path(root)
        self.agent_id = agent_id
        self._streams_dir = self.root / STREAMS_DIR
        self._counters: dict[StreamId, int] = {s: 0 for s in StreamId}
        self._sync_counters_from_disk()

    @classmethod
    def open_or_init(cls, root: Path, agent_id: str) -> AtepStore:
        """Open an existing store or initialize a new one at `root`."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / STREAMS_DIR).mkdir(exist_ok=True)
        manifest_path = root / MANIFEST_NAME
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = data.get("agent_id")
            if existing and existing != agent_id:
                raise AtepError(
                    f"store at {root} belongs to {existing!r}, not {agent_id!r}"
                )
        else:
            manifest_path.write_text(
                json.dumps({"agent_id": agent_id, "schema_version": 1}, indent=2),
                encoding="utf-8",
            )
        return cls(root, agent_id)

    def _sync_counters_from_disk(self) -> None:
        if not self._streams_dir.exists():
            return
        for path in self._streams_dir.iterdir():
            if not path.name.endswith(".atep"):
                continue
            stem = path.stem  # e.g. identity-00000001
            try:
                stream_name, seq_str = stem.rsplit("-", 1)
                stream = StreamId(stream_name)
                seq = int(seq_str)
            except (ValueError, KeyError):
                continue
            if seq > self._counters[stream]:
                self._counters[stream] = seq

    def _next_segment_path(self, stream: StreamId) -> Path:
        self._counters[stream] += 1
        seq = self._counters[stream]
        return self._streams_dir / f"{stream.value}-{seq:08d}.atep"

    def append_event(self, event: AtepEvent) -> Path:
        """Write a single-event segment for this event. Returns the segment path."""
        if event.header.agent_id != self.agent_id:
            raise AtepError(
                f"event agent_id {event.header.agent_id!r} != store {self.agent_id!r}"
            )
        path = self._next_segment_path(event.header.stream)
        with SegmentWriter(path) as w:
            w.append(event)
        return path

    def list_segments(self, stream: Optional[StreamId] = None) -> list[Path]:
        if not self._streams_dir.exists():
            return []
        files = sorted(self._streams_dir.glob("*.atep"))
        if stream is None:
            return files
        prefix = f"{stream.value}-"
        return [f for f in files if f.name.startswith(prefix)]

    def read_stream(self, stream: StreamId) -> Iterator[AtepEvent]:
        for path in self.list_segments(stream):
            yield from SegmentReader(path).iter_events()

    def verify_all(self, verifying_key: VerifyingKey) -> VerificationReport:
        """Iterate every segment, verify CRC, Merkle root, and signatures."""
        report = VerificationReport(ok=True)
        for path in self.list_segments():
            try:
                reader = SegmentReader(path)
            except AtepError as e:
                report.ok = False
                report.failures.append(f"{path.name}: envelope: {e}")
                continue
            report.segments_checked += 1
            if not reader.verify_merkle_root():
                report.ok = False
                report.failures.append(f"{path.name}: merkle root mismatch")
                continue
            for ev in reader.iter_events():
                report.events_checked += 1
                if not ev.verify(verifying_key):
                    report.ok = False
                    report.failures.append(
                        f"{path.name}: event {ev.header.event_id.hex()}"
                        " signature/causal_hash invalid"
                    )
        return report

    def compute_root_hash(self) -> bytes:
        """``BLAKE3`` over sorted ``(stream, file)`` Merkle roots.

        Deterministic across two runs that contain the same segments.
        """
        roots: list[bytes] = []
        for path in sorted(self.list_segments()):
            reader = SegmentReader(path)
            roots.append(path.name.encode("utf-8") + b"\x00" + reader.merkle_root)
        if not roots:
            return b"\x00" * 32
        return blake3_bytes(b"".join(roots))

    def manifest(self) -> dict[str, object]:
        manifest_path = self.root / MANIFEST_NAME
        data: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data
