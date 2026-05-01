"""Tests for HLC."""
from __future__ import annotations

from agentlock.atep.clock import Hlc


def test_le_bytes_roundtrip() -> None:
    h = Hlc(1234567890123, 5, 2)
    assert Hlc.from_le_bytes(h.to_le_bytes()) == h


def test_le_bytes_length() -> None:
    assert len(Hlc(0, 0, 0).to_le_bytes()) == 16


def test_tick_after_increments_logical_when_same_physical() -> None:
    a = Hlc(100, 0, 0)
    b = a.tick_after(a, now_ms=100)
    assert b.physical_ms == 100
    assert b.logical == 1


def test_tick_after_jumps_to_now_when_clock_advances() -> None:
    a = Hlc(100, 5, 0)
    b = a.tick_after(a, now_ms=200)
    assert b.physical_ms == 200
    assert b.logical == 0


def test_ordering() -> None:
    assert Hlc(1, 0, 0) < Hlc(1, 1, 0)
    assert Hlc(1, 1, 0) < Hlc(2, 0, 0)
