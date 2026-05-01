"""ATEP binary segment file format.

Layout::

    MAGIC          "ATEP"      4 B
    VERSION        u16         2 B
    FLAGS          u16         2 B
    EVENT_COUNT    u32         4 B
    FIRST_HLC                  16 B
    LAST_HLC                   16 B
    MERKLE_ROOT                32 B
    ─────────── 76-byte header
    FRAMES (event_count of):
      FRAME_LEN    u32         4 B
      EVENT_BYTES  variable (canonical CBOR)
    INDEX (placeholder for v0.1, length tracked by INDEX_LEN)
    INDEX_OFFSET   u64         8 B
    INDEX_LEN      u64         8 B
    CRC32                      4 B
    MAGIC_TAIL     "PETA"      4 B
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from agentlock.atep.clock import Hlc
from agentlock.atep.event import AtepEvent
from agentlock.crypto.canonical import canonical_cbor, canonical_cbor_decode
from agentlock.crypto.hashing import blake3_bytes
from agentlock.exceptions import AtepError

SEGMENT_MAGIC_HEAD = b"ATEP"
SEGMENT_MAGIC_TAIL = b"PETA"
SEGMENT_VERSION = 1
HEADER_LEN = 76
TRAILER_LEN = 8 + 8 + 4 + 4  # index_offset + index_len + crc + tail


def _merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return b"\x00" * 32
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [blake3_bytes(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


@dataclass
class SegmentSummary:
    event_count: int
    first_hlc: Hlc
    last_hlc: Hlc
    merkle_root: bytes
    bytes_written: int


class SegmentWriter:
    """Append-only writer for a ``.atep`` segment file.

    Use as a context manager — finalize is called on exit.

    Example:
        >>> import tempfile, ulid
        >>> from agentlock.atep.event import EventHeader, StreamId, AtepEvent
        >>> from agentlock.crypto.signing import SigningKey
        >>> sk = SigningKey.generate()
        >>> with tempfile.NamedTemporaryFile(suffix=".atep", delete=False) as f:
        ...     path = f.name
        >>> with SegmentWriter(path) as w:
        ...     hdr = EventHeader(event_id=ulid.new().bytes, agent_id="agent://a/b",
        ...         stream=StreamId.IDENTITY, stream_seq=0, clock=Hlc(1, 0, 0),
        ...         event_type="t", payload_schema_uri="atep://t")
        ...     w.append(AtepEvent.seal(hdr, {"x": 1}, sk))
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._opened = False
        self._events: list[bytes] = []
        self._causal_hashes: list[bytes] = []
        self._first_hlc: Optional[Hlc] = None
        self._last_hlc: Optional[Hlc] = None
        self._finalized = False

    def __enter__(self) -> SegmentWriter:
        self._opened = True
        return self

    def __exit__(self, *a: object) -> None:
        if not self._finalized:
            self.finalize()

    def append(self, event: AtepEvent) -> None:
        """Buffer one event. Bytes are written on finalize."""
        if not self._opened:
            raise AtepError("writer is not open")
        if self._first_hlc is None:
            self._first_hlc = event.header.clock
        self._last_hlc = event.header.clock
        body = canonical_cbor(event.model_dump(mode="python"))
        self._events.append(body)
        self._causal_hashes.append(event.causal_hash)

    def finalize(self) -> SegmentSummary:
        """Flush header, frames, trailer, CRC, magic tail."""
        if not self._opened:
            raise AtepError("writer is not open")
        merkle_root = _merkle_root(self._causal_hashes)
        first = self._first_hlc or Hlc(0, 0, 0)
        last = self._last_hlc or Hlc(0, 0, 0)
        buf = bytearray()
        buf += SEGMENT_MAGIC_HEAD
        buf += struct.pack("<HH", SEGMENT_VERSION, 0)
        buf += struct.pack("<I", len(self._events))
        buf += first.to_le_bytes()
        buf += last.to_le_bytes()
        buf += merkle_root
        assert len(buf) == HEADER_LEN
        for body in self._events:
            buf += struct.pack("<I", len(body))
            buf += body
        index_offset = len(buf)
        index_data = b""
        buf += index_data
        buf += struct.pack("<QQ", index_offset, len(index_data))
        crc = zlib.crc32(bytes(buf))
        buf += struct.pack("<I", crc)
        buf += SEGMENT_MAGIC_TAIL
        self.path.write_bytes(bytes(buf))
        self._finalized = True
        self._opened = False
        return SegmentSummary(
            event_count=len(self._events),
            first_hlc=first,
            last_hlc=last,
            merkle_root=merkle_root,
            bytes_written=self.path.stat().st_size,
        )


class SegmentReader:
    """Read events from a ``.atep`` segment file.

    Validates magic markers and CRC32 on construction. Call
    :meth:`verify_merkle_root` to confirm event-level integrity.

    Example:
        >>> # See SegmentWriter docstring for a full roundtrip
        >>> True
        True
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data = self.path.read_bytes()
        self._validate_envelope()
        self._parse_header()

    def _validate_envelope(self) -> None:
        if len(self._data) < HEADER_LEN + TRAILER_LEN:
            raise AtepError("segment too short")
        if self._data[:4] != SEGMENT_MAGIC_HEAD:
            raise AtepError("invalid magic head")
        if self._data[-4:] != SEGMENT_MAGIC_TAIL:
            raise AtepError("invalid magic tail")
        body = self._data[:-8]  # everything before CRC + tail
        crc_stored = struct.unpack("<I", self._data[-8:-4])[0]
        crc_actual = zlib.crc32(body)
        if crc_stored != crc_actual:
            raise AtepError(f"CRC mismatch: stored={crc_stored:08x} actual={crc_actual:08x}")

    def _parse_header(self) -> None:
        h = self._data[:HEADER_LEN]
        self.version, self.flags = struct.unpack("<HH", h[4:8])
        self.event_count = struct.unpack("<I", h[8:12])[0]
        self.first_hlc = Hlc.from_le_bytes(h[12:28])
        self.last_hlc = Hlc.from_le_bytes(h[28:44])
        self.merkle_root = h[44:76]

    def iter_events(self) -> Iterator[AtepEvent]:
        offset = HEADER_LEN
        for _ in range(self.event_count):
            (frame_len,) = struct.unpack("<I", self._data[offset : offset + 4])
            offset += 4
            body = self._data[offset : offset + frame_len]
            offset += frame_len
            payload = canonical_cbor_decode(body)
            yield AtepEvent.model_validate(payload)

    def verify_merkle_root(self) -> bool:
        """Recompute the binary BLAKE3 Merkle root and compare to the header."""
        causal_hashes = [e.causal_hash for e in self.iter_events()]
        return _merkle_root(causal_hashes) == self.merkle_root
