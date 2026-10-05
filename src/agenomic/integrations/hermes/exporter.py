"""Event construction, redaction and asynchronous export for the Hermes adapter.

Events follow ``agenomic.hermes.event/v1``. They are redacted when they are
built, before they reach the queue or the spool: by default an event carries
metadata and ``blake3:`` hashes only; ``capture.content = "redacted_preview"``
adds previews that went through :class:`~agenomic.redaction.RedactionEngine`,
secret pattern masking and truncation.

:class:`EventExporter` mirrors the LangChain ``_Dispatcher``: a bounded
buffer, one daemon thread, never blocking the agent loop. It batches up to 500
events per ``POST /v1/hermes/runtime/events``, retries a batch a limited number
of times with backoff, deduplicates by ``event_id``, and counts what it drops.
An optional spool keeps undelivered batches in a size capped JSONL file.

Telemetry is not the security record: decisions are stored server side when
they are made, so a dropped event never changes a decision.

Example:
    >>> sent = []
    >>> exporter = EventExporter(lambda batch: sent.extend(batch) or {"accepted": len(batch)})
    >>> builder = EventBuilder()
    >>> exporter.submit(builder.build("session.started", hermes_session_id="s1"))
    True
    >>> exporter.flush(2.0)
    True
    >>> sent[0]["type"], exporter.stats()["dropped"]
    ('session.started', 0)
    >>> exporter.close(1.0)
    True
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import ulid

from agenomic.integrations.hermes.canonical import CanonicalError, arguments_hash
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule

logger = logging.getLogger("agenomic.integrations.hermes.exporter")

EVENT_SCHEMA = "agenomic.hermes.event/v1"
MAX_BATCH = 500
MAX_EVENT_BYTES = 64 * 1024
_DEDUP_WINDOW = 50_000

#: Keys masked anywhere in a preview.
SECRET_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "client_secret",
    "cookie",
)
_SECRET_PATTERNS = re.compile(
    r"(agmh[rs]_[A-Za-z0-9_\-]+"
    r"|sk-[A-Za-z0-9_\-]{8,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]+"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9\-]+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_PREVIEW_ENGINE = RedactionEngine(
    [RedactionRule(path=f"**.{key}", mode=RedactionMode.MASK) for key in SECRET_KEYS]
)

CaptureMode = Literal["metadata", "redacted_preview"]


def now_iso() -> str:
    """RFC 3339 UTC timestamp with milliseconds.

    Example:
        >>> now_iso().endswith("Z")
        True
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def content_hash(value: object) -> str:
    """``blake3:`` hash of any value; values without a canonical form hash their ``repr``.

    Example:
        >>> content_hash("x") == content_hash("x")
        True
    """
    try:
        return arguments_hash(value)
    except CanonicalError:
        return arguments_hash({"repr": repr(value)})


def mask_text(text: str) -> str:
    """Mask credential shaped substrings.

    Example:
        >>> mask_text("key sk-abcdefghijkl and agmhr_123")
        'key *** and ***'
    """
    return _SECRET_PATTERNS.sub("***", text)


def redacted_preview(value: object, limit: int) -> str:
    """Redact (secret keys masked, secret patterns masked) then truncate to ``limit`` chars.

    Example:
        >>> redacted_preview({"path": "/tmp/a", "api_key": "k"}, 200)
        '{"api_key":"***","path":"/tmp/a"}'
    """
    if isinstance(value, (dict, list)):
        try:
            redacted = _PREVIEW_ENGINE.apply(value)
            text = json.dumps(redacted, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            text = repr(type(value).__name__)
    elif isinstance(value, str):
        text = value
    else:
        text = repr(value)
    text = mask_text(text)
    return text[:limit]


class EventBuilder:
    """Builds ``agenomic.hermes.event/v1`` documents with a per process ``seq``.

    Raw content passed as ``content`` is never copied into the event: its hash
    goes into ``extra.content_hashes`` and, only in ``redacted_preview``
    capture, a redacted truncated preview goes into ``extra.previews``.

    Example:
        >>> b = EventBuilder()
        >>> e = b.build("tool.call.requested", hermes_session_id="s", content={"input": {"q": "x"}})
        >>> e["seq"], "input" in e["extra"]["content_hashes"], "previews" in e["extra"]
        (1, True, False)
    """

    _FIELDS = (
        "hermes_session_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "turn_id",
        "action_id",
        "attempt_id",
        "tool",
        "model",
        "decision",
        "reason",
        "status",
        "latency_ms",
        "input_hash",
        "output_hash",
        "usage",
    )

    def __init__(self, capture: CaptureMode = "metadata", preview_chars: int = 200) -> None:
        self.capture = capture
        self.preview_chars = preview_chars
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def build(
        self,
        event_type: str,
        *,
        content: Optional[Mapping[str, object]] = None,
        extra: Optional[Mapping[str, object]] = None,
        **fields: object,
    ) -> dict[str, Any]:
        """Return a redacted event; unknown ``fields`` are rejected to keep the schema closed."""
        with self._lock:
            seq = next(self._seq)
        event: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA,
            "event_id": ulid.new().str,
            "type": event_type,
            "seq": seq,
            "occurred_at": now_iso(),
        }
        for key, value in fields.items():
            if key not in self._FIELDS:
                raise ValueError(f"unknown event field {key}")
            if value is not None:
                event[key] = value
        extra_doc: dict[str, Any] = dict(extra or {})
        if content:
            extra_doc["content_hashes"] = {k: content_hash(v) for k, v in content.items()}
            if self.capture == "redacted_preview":
                extra_doc["previews"] = {
                    k: redacted_preview(v, self.preview_chars) for k, v in content.items()
                }
        event["extra"] = extra_doc
        return event


class ExporterStats(dict[str, Any]):
    """``{"buffered", "dropped", "buffer_full", "last_flush_error"}`` for heartbeats."""


class _Spool:
    """Append only JSONL file capped at ``max_bytes``; lines are already redacted events."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def append(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write what fits; return the events that did not fit."""
        with self._lock:
            size = self.size()
            lines: list[str] = []
            rejected: list[dict[str, Any]] = []
            for event in events:
                line = json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n"
                cost = len(line.encode("utf-8"))
                if size + cost > self.max_bytes:
                    rejected.append(event)
                    continue
                size += cost
                lines.append(line)
            if lines:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.writelines(lines)
            return rejected

    def drain(self, limit: int) -> list[dict[str, Any]]:
        """Read up to ``limit`` events and rewrite the file without them."""
        with self._lock:
            try:
                raw = self.path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                return []
            taken: list[dict[str, Any]] = []
            for line in raw[:limit]:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    taken.append(item)
            rest = raw[limit:]
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.writelines(line + "\n" for line in rest)
            os.replace(tmp, self.path)
            return taken


class EventExporter:
    """Bounded, batching, retrying event exporter on a daemon thread.

    ``post`` receives a list of at most ``batch_size`` events and returns the
    server answer; any exception counts as a failed attempt. ``submit`` never
    blocks: when the buffer is full the event goes to the spool, or is dropped
    and counted.
    """

    def __init__(
        self,
        post: Callable[[list[dict[str, Any]]], Any],
        *,
        max_events: int = 10_000,
        max_bytes: int = 16 * 1024 * 1024,
        flush_interval_s: float = 1.0,
        batch_size: int = MAX_BATCH,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        spool_path: Optional[str] = None,
        spool_max_bytes: int = 64 * 1024 * 1024,
        name: str = "agenomic-hermes-exporter",
    ) -> None:
        self._post = post
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._interval = flush_interval_s
        self._batch_size = max(1, min(batch_size, MAX_BATCH))
        self._max_retries = max(0, max_retries)
        self._backoff = backoff_s
        self._spool = _Spool(Path(spool_path).expanduser(), spool_max_bytes) if spool_path else None
        self._buf: deque[tuple[dict[str, Any], int]] = deque()
        self._bytes = 0
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._cond = threading.Condition()
        self._closed = False
        self._in_flight = 0
        self._dropped = 0
        self._buffer_full = False
        self._last_error: Optional[str] = None
        self._delivered = 0
        self._last_failure_at = float("-inf")
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- producer side -------------------------------------------------
    def submit(self, event: dict[str, Any]) -> bool:
        """Queue one already redacted event; ``False`` when it was dropped or spooled."""
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            self._count_drop("event without event_id")
            return False
        try:
            encoded = json.dumps(event, separators=(",", ":"), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            self._count_drop("event not serializable")
            return False
        size = len(encoded.encode("utf-8"))
        if size > MAX_EVENT_BYTES:
            event = {k: v for k, v in event.items() if k != "extra"}
            event["extra"] = {"truncated": True}
            size = len(json.dumps(event, default=str).encode("utf-8"))
        with self._cond:
            if self._closed:
                self._count_drop_locked("exporter closed")
                return False
            if event_id in self._seen:
                return True
            if len(self._buf) >= self._max_events or self._bytes + size > self._max_bytes:
                self._buffer_full = True
                overflow = True
            else:
                overflow = False
                self._buf.append((event, size))
                self._bytes += size
                self._remember(event_id)
                if len(self._buf) >= self._batch_size:
                    self._cond.notify_all()
        if overflow:
            self._overflow([event], "buffer full")
            return False
        return True

    def _remember(self, event_id: str) -> None:
        self._seen[event_id] = None
        while len(self._seen) > _DEDUP_WINDOW:
            self._seen.popitem(last=False)

    def _count_drop(self, reason: str, n: int = 1) -> None:
        with self._cond:
            self._count_drop_locked(reason, n)

    def _count_drop_locked(self, reason: str, n: int = 1) -> None:
        self._dropped += n
        if self._dropped <= 3:
            logger.warning("Hermes event(s) dropped: %s", reason)

    def _overflow(self, events: list[dict[str, Any]], reason: str) -> None:
        if self._spool is not None:
            try:
                rejected = self._spool.append(events)
            except OSError as e:
                logger.warning("event spool write failed: %s", type(e).__name__)
                rejected = events
            if rejected:
                self._count_drop(f"{reason}; spool full", len(rejected))
        else:
            self._count_drop(reason, len(events))

    # -- introspection -------------------------------------------------
    def stats(self) -> ExporterStats:
        """Exporter counters for the runtime heartbeat."""
        with self._cond:
            buffered = len(self._buf) + self._in_flight
            return ExporterStats(
                buffered=buffered,
                dropped=self._dropped,
                buffer_full=self._buffer_full,
                last_flush_error=self._last_error,
            )

    @property
    def delivered(self) -> int:
        """Events the server acknowledged."""
        return self._delivered

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until the buffer is empty; ``False`` on timeout or when anything was dropped."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._cond.notify_all()
            while self._buf or self._in_flight:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(0.05 if remaining is None else min(0.05, remaining))
            return self._dropped == 0

    def close(self, timeout: Optional[float] = 5.0) -> bool:
        """Refuse new events, drain, stop the worker. Undelivered events go to the spool."""
        drained = self.flush(timeout)
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout)
        with self._cond:
            leftover = [e for e, _ in self._buf]
            self._buf.clear()
            self._bytes = 0
        if leftover:
            self._overflow(leftover, "exporter closed before delivery")
        return drained

    # -- worker --------------------------------------------------------
    def _take_batch(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while self._buf and len(batch) < self._batch_size:
            event, size = self._buf.popleft()
            self._bytes -= size
            batch.append(event)
        if len(self._buf) < self._max_events:
            self._buffer_full = False
        self._in_flight = len(batch)
        return batch

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._buf and not self._closed:
                    self._cond.wait(self._interval)
                if self._closed and not self._buf:
                    return
                batch = self._take_batch()
            if batch:
                self._deliver(batch)
            else:
                self._replay_spool()
            with self._cond:
                self._in_flight = 0
                self._cond.notify_all()

    def _deliver(self, batch: list[dict[str, Any]]) -> bool:
        for attempt in range(self._max_retries + 1):
            try:
                self._post(batch)
            except Exception as exc:  # the transport's failure is telemetry only
                self._last_error = f"{type(exc).__name__}: {getattr(exc, 'code', '')}".rstrip(": ")
                if attempt < self._max_retries:
                    with self._cond:
                        if self._closed:
                            break
                        self._cond.wait(self._backoff * (2**attempt))
                    continue
                break
            else:
                self._last_error = None
                self._delivered += len(batch)
                return True
        self._last_failure_at = time.monotonic()
        self._overflow(batch, f"delivery failed after {self._max_retries + 1} attempt(s)")
        return False

    def _replay_spool(self) -> None:
        if self._spool is None or self._spool.size() == 0:
            return
        # After a failed delivery, wait a few intervals before replaying the spool again.
        if time.monotonic() - self._last_failure_at < 5 * self._interval:
            return
        try:
            events = self._spool.drain(self._batch_size)
        except OSError as e:
            logger.warning("event spool read failed: %s", type(e).__name__)
            return
        if events:
            with self._cond:
                self._in_flight = len(events)
            self._deliver(events)
            with self._cond:
                self._in_flight = 0
