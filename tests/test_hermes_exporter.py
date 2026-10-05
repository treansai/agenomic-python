from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agenomic.integrations.hermes.exporter import (
    EVENT_SCHEMA,
    EventBuilder,
    EventExporter,
    mask_text,
    redacted_preview,
)

SECRETS = ("sk-livesecretvalue123", "agmhr_runtimesecret", "hunter2-password")


def raw_content() -> dict[str, Any]:
    return {
        "input": {
            "command": f"curl -H 'Authorization: Bearer {SECRETS[1]}' https://x",
            "api_key": SECRETS[0],
            "nested": {"password": SECRETS[2]},
        },
        "output": f"done with {SECRETS[0]}",
    }


def test_builder_shape_and_metadata_default() -> None:
    builder = EventBuilder()
    e1 = builder.build(
        "tool.call.requested", hermes_session_id="s", tool={"name": "t"}, content=raw_content()
    )
    e2 = builder.build("tool.call.completed", hermes_session_id="s")
    assert e1["schema_version"] == EVENT_SCHEMA
    assert len(e1["event_id"]) == 26
    assert e1["event_id"] != e2["event_id"]
    assert (e1["seq"], e2["seq"]) == (1, 2)
    assert e1["occurred_at"].endswith("Z")
    assert set(e1["extra"]["content_hashes"]) == {"input", "output"}
    assert "previews" not in e1["extra"]
    blob = json.dumps(e1)
    assert not any(s in blob for s in SECRETS)
    with pytest.raises(ValueError, match="unknown event field"):
        builder.build("x", prompt="raw")


def test_builder_redacted_preview_masks_and_truncates() -> None:
    builder = EventBuilder("redacted_preview", preview_chars=80)
    event = builder.build("tool.call.requested", content=raw_content())
    blob = json.dumps(event)
    assert not any(s in blob for s in SECRETS)
    assert all(len(p) <= 80 for p in event["extra"]["previews"].values())
    assert mask_text("token agmhs_abc and ghp_" + "a" * 20) == "token *** and ***"
    assert redacted_preview(42, 10) == "42"


def collect() -> tuple[list[list[dict[str, Any]]], Any]:
    batches: list[list[dict[str, Any]]] = []

    def post(batch: list[dict[str, Any]]) -> dict[str, int]:
        batches.append(list(batch))
        return {"accepted": len(batch)}

    return batches, post


def test_batching_and_dedup() -> None:
    batches, post = collect()
    builder = EventBuilder()
    exporter = EventExporter(post, batch_size=3, flush_interval_s=0.05)
    events = [builder.build("x") for _ in range(7)]
    for e in events:
        assert exporter.submit(e)
    assert exporter.submit(events[0])  # duplicate accepted silently, not resent
    assert exporter.flush(3.0)
    sent = [e["event_id"] for b in batches for e in b]
    assert sorted(sent) == sorted(e["event_id"] for e in events)
    assert all(len(b) <= 3 for b in batches)
    assert exporter.stats() == {
        "buffered": 0,
        "dropped": 0,
        "buffer_full": False,
        "last_flush_error": None,
    }
    assert exporter.delivered == 7
    assert exporter.close(1.0)
    assert not exporter.submit(EventBuilder().build("late"))
    assert exporter.stats()["dropped"] == 1


def test_backpressure_drops_and_counts_without_blocking() -> None:
    gate = threading.Event()

    def stalled(batch: list[dict[str, Any]]) -> None:
        gate.wait(5)

    builder = EventBuilder()
    exporter = EventExporter(
        stalled, max_events=2, batch_size=1, flush_interval_s=0.01, max_retries=0
    )
    time.sleep(0.05)
    started = time.monotonic()
    results = [exporter.submit(builder.build("x")) for _ in range(10)]
    assert time.monotonic() - started < 1.0  # never blocks the agent loop
    stats = exporter.stats()
    assert results.count(False) >= 7
    assert stats["dropped"] == results.count(False)
    assert stats["buffer_full"] is True
    gate.set()
    exporter.close(2.0)


def test_retry_limit_then_drop() -> None:
    attempts: list[int] = []

    def failing(batch: list[dict[str, Any]]) -> None:
        attempts.append(len(batch))
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=2, backoff_s=0.01, flush_interval_s=0.01)
    exporter.submit(EventBuilder().build("x"))
    assert not exporter.flush(3.0)
    assert len(attempts) == 3
    stats = exporter.stats()
    assert stats["dropped"] == 1
    assert stats["last_flush_error"] == "ConnectionError"
    exporter.close(1.0)


def test_spool_is_bounded_redacted_and_replayed(tmp_path: Path) -> None:
    spool = tmp_path / "spool" / "events.jsonl"
    up = threading.Event()
    delivered: list[dict[str, Any]] = []

    def post(batch: list[dict[str, Any]]) -> None:
        if not up.is_set():
            raise ConnectionError("down")
        delivered.extend(batch)

    builder = EventBuilder("redacted_preview")
    exporter = EventExporter(
        post, max_retries=0, flush_interval_s=0.02, spool_path=str(spool), spool_max_bytes=4096
    )
    for _ in range(40):
        exporter.submit(builder.build("tool.call.requested", content=raw_content()))
    exporter.flush(3.0)
    assert spool.exists()
    text = spool.read_text()
    assert 0 < spool.stat().st_size <= 4096
    assert not any(s in text for s in SECRETS)
    assert oct(spool.stat().st_mode & 0o777) == "0o600"
    stats = exporter.stats()
    spooled = len(text.splitlines())
    assert stats["dropped"] == 40 - spooled  # what did not fit is counted, never silent
    up.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(delivered) < spooled:
        exporter.submit(builder.build("nudge"))
        time.sleep(0.05)
    assert len({e["event_id"] for e in delivered if e["type"] == "tool.call.requested"}) == spooled
    exporter.close(1.0)


def test_close_spools_leftovers(tmp_path: Path) -> None:
    spool = tmp_path / "s.jsonl"

    def failing(batch: list[dict[str, Any]]) -> None:
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=0, flush_interval_s=10, spool_path=str(spool))
    exporter.submit(EventBuilder().build("x"))
    exporter.close(0.2)
    assert len(spool.read_text().splitlines()) == 1


def test_oversized_and_invalid_events() -> None:
    batches, post = collect()
    exporter = EventExporter(post, flush_interval_s=0.01)
    big = EventBuilder().build("x", extra={"blob": "a" * 100_000})
    assert exporter.submit(big)
    assert not exporter.submit({"type": "no-id"})
    exporter.flush(2.0)
    assert batches[0][0]["extra"] == {"truncated": True}
    assert exporter.stats()["dropped"] == 1
    exporter.close(1.0)
