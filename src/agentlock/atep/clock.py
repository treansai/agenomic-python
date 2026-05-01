"""Hybrid Logical Clock (HLC) per Kulkarni et al. 2014.

Wire format: 16 bytes little-endian
    physical_ms : u64
    logical     : u32
    node_id     : u32
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, order=True)
class Hlc:
    """Hybrid Logical Clock tuple.

    Example:
        >>> a = Hlc(1, 0, 0)
        >>> b = a.tick_after(a, now_ms=1)
        >>> b > a
        True
    """

    physical_ms: int
    logical: int
    node_id: int

    @classmethod
    def now(cls, node_id: int = 0) -> Hlc:
        return cls(int(time.time() * 1000), 0, node_id)

    def tick_after(self, received: Hlc, now_ms: Optional[int] = None) -> Hlc:
        """HLC update on event receipt (Kulkarni et al. 2014)."""
        physical = now_ms if now_ms is not None else int(time.time() * 1000)
        new_physical = max(self.physical_ms, received.physical_ms, physical)
        if new_physical == self.physical_ms == received.physical_ms:
            new_logical = max(self.logical, received.logical) + 1
        elif new_physical == self.physical_ms:
            new_logical = self.logical + 1
        elif new_physical == received.physical_ms:
            new_logical = received.logical + 1
        else:
            new_logical = 0
        return Hlc(new_physical, new_logical, self.node_id)

    def to_le_bytes(self) -> bytes:
        """16 bytes: physical_ms (u64 LE) || logical (u32 LE) || node_id (u32 LE).

        Example:
            >>> len(Hlc(1, 2, 3).to_le_bytes())
            16
        """
        return struct.pack("<QII", self.physical_ms, self.logical, self.node_id)

    @classmethod
    def from_le_bytes(cls, data: bytes) -> Hlc:
        """Decode 16-byte little-endian HLC."""
        if len(data) != 16:
            raise ValueError("HLC must be 16 bytes")
        p, lo, n = struct.unpack("<QII", data)
        return cls(p, lo, n)
