"""Event construction, redaction and asynchronous export for the Hermes adapter.

Events follow ``agenomic.hermes.event/v1``. They are redacted when they are
built, before they reach the queue or the spool: by default an event carries
metadata and ``blake3:`` hashes only; ``capture.content = "redacted_preview"``
adds previews whose credential-named keys are masked (case and separators
ignored), then secret pattern masking and truncation.

:class:`EventExporter` mirrors the LangChain ``_Dispatcher``: a bounded
buffer, one daemon thread, never blocking the agent loop. It batches up to 500
events per ``POST /v1/hermes/runtime/events``, retries a batch a limited number
of times with backoff, deduplicates by ``event_id``, and counts what it drops.
An optional spool keeps undelivered batches in a size capped JSONL file; it is
replayed one batch at a time when the buffer is idle and, under continuous
load, after every ``REPLAY_EVERY`` live batches or once per flush interval.

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
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence, Set
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import ulid
from pydantic import JsonValue

from agenomic.integrations.hermes.canonical import CanonicalError, arguments_hash

logger = logging.getLogger("agenomic.integrations.hermes.exporter")

EVENT_SCHEMA = "agenomic.hermes.event/v1"
MAX_BATCH = 500
MAX_EVENT_BYTES = 64 * 1024
_DEDUP_WINDOW = 50_000
#: Under continuous load one spool batch is replayed after this many live batches (or once
#: a flush interval passed since the last replay), so a busy process still drains its spool.
REPLAY_EVERY = 10

#: Key families masked anywhere in a preview, matched case insensitively after removing
#: separators: a normalized key containing one of these (or ending in ``token``) is masked,
#: so ``API_KEY``, ``X-Api-Key``, ``Authorization`` and ``db_password`` are all covered.
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
#: Credential shaped tokens, masked whole wherever they appear.
_SECRET_PATTERNS = re.compile(
    r"(agmh[rs]_[A-Za-z0-9_\-]+"
    r"|sk-[A-Za-z0-9_\-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9\-]+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
#: HTTP authorization schemes kept readable in front of a masked credential.
_AUTH_SCHEMES = (
    r"bearer|basic|digest|token|apikey|api-key|aws4-hmac-sha256|hmac-sha256|negotiate|ntlm"
    r"|dpop|gnap|hoba|mutual|signature|sharedkey|sharedkeylite|vapid"
)
#: ``Authorization: <scheme> <credential>`` (also ``Proxy-Authorization``) and cookie
#: headers: the header name and scheme stay, the rest of the value is masked up to the
#: quote that encloses the header (shell argument, JSON string) or the end of the line.
_HEADER_SECRET = re.compile(
    r"(?i)(?<![\w-])([\"']?)((?:proxy-)?authorization|(?:set-)?cookie)"
    r"([\"']?[ \t]*[:=][ \t]*)([\"']?)"
    rf"((?:{_AUTH_SCHEMES})[ \t]+)?"
)
#: A bare ``Bearer <token>`` outside a header.
_BEARER_SECRET = re.compile(r"(?i)(?<![\w-])(bearer[ \t]+)([A-Za-z0-9._~+/\-]+=*)")
#: ``key=value`` / ``key: value`` pairs whose key names a credential (``api_key``,
#: ``x-api-key``, ``password``, ``auth_token``, ``client_secret``, AWS ``Credential`` and
#: ``Signature``...). The key must be followed by ``:`` or ``=``, so prose such as
#: "the token budget" or ``max_tokens=512`` stays readable.
_KEY_VALUE_SECRET = re.compile(
    r"(?i)(?<![\w-])([\w-]*?(?:api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key"
    r"|token|password|passwd|secret|credentials?|signature))"
    r"([\"']?[ \t]*[:=][ \t]*)"
    r"(\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s\"'&,;<>=][^\s\"'&,;<>]*)"
)
#: ``scheme://user:password@host``: the password is masked, the user and host stay.
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s/:@\"']+:)([^\s/@\"']+)(@)")
_SECRET_KEY_PARTS = tuple(
    sorted({re.sub(r"[^a-z0-9]", "", key) for key in SECRET_KEYS} - {"token"})
) + ("credential",)
_MASK = "***"


def is_secret_key(key: str) -> bool:
    """Whether a mapping key names a credential, whatever its case or separators.

    Example:
        >>> [is_secret_key(k) for k in ("API_KEY", "X-Api-Key", "auth_token", "max_tokens", "path")]
        [True, True, True, False, False]
    """
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized.endswith("token") or any(part in normalized for part in _SECRET_KEY_PARTS)


def _is_container(value: object) -> bool:
    """Whether ``json.dumps`` (with ``default=str``) would expand ``value`` into members."""
    return isinstance(value, (Mapping, Set)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )


def _walk(value: object, leaf: Callable[[object], object]) -> object:
    """Mask credential-named keys in every mapping, sequence or set; ``leaf`` maps the rest.

    Every container becomes a ``dict`` or ``list``, so no tuple, custom mapping or set
    reaches ``json.dumps`` with its members unredacted.
    """
    if isinstance(value, Mapping):
        return {
            k: _MASK if isinstance(k, str) and is_secret_key(k) else _walk(v, leaf)
            for k, v in value.items()
        }
    if _is_container(value):
        return [_walk(item, leaf) for item in cast(Iterable[object], value)]
    return leaf(value)


def _redact_leaf(value: object) -> object:
    """A JSON scalar for any leaf, with credential-shaped text masked.

    Strings are masked; ``None``, booleans, integers and finite floats stay; a non-finite
    float (``json.dumps`` would write ``NaN``, which is not JSON) becomes ``None``; anything
    else (an exception, bytes, a custom object) becomes its masked ``str``, so no ``str()``
    taken later by ``json.dumps(default=str)`` can carry an unmasked credential.

    Example:
        >>> _redact_leaf(ValueError("Authorization: Basic cGxhaW4=")), _redact_leaf(float("nan"))
        ('Authorization: Basic ***', None)
    """
    if isinstance(value, str):
        return mask_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        text = str(value)
    except Exception:  # a broken __str__ must not drop the event
        text = f"<{type(value).__name__}>"
    return mask_text(text)


def _redact_field(value: object) -> object:
    """Mask credential-named keys and credential-shaped text in any event field."""
    return _walk(value, _redact_leaf)


def _mask_secret_keys(value: object) -> object:
    return _walk(value, lambda leaf: leaf)


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


def _mask_quoted(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[0] + _MASK + value[0]
    return _MASK


def _mask_headers(text: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _HEADER_SECRET.finditer(text):
        if m.start() < pos:
            continue
        quote = m.group(4) or m.group(1)
        end = m.end()
        while end < len(text) and text[end] not in "\r\n":
            if quote and text[end] == "\\":
                end += 2
                continue
            if quote and text[end] == quote:
                break
            end += 1
        end = min(end, len(text))
        if end > m.end():
            out.append(text[pos : m.end()] + _MASK)
            pos = end
    out.append(text[pos:])
    return "".join(out)


def mask_text(text: str) -> str:
    """Mask credential shaped substrings; schemes and key names stay readable.

    Example:
        >>> mask_text("key sk-abcdefghijkl and agmhr_123")
        'key *** and ***'
        >>> mask_text("Authorization: Basic cGxhaW5zZWNyZXQ=")
        'Authorization: Basic ***'
        >>> mask_text("password=hunter2 max_tokens=512 https://u:pw@db.example/x")
        'password=*** max_tokens=512 https://u:***@db.example/x'
    """
    text = _mask_headers(text)
    text = _BEARER_SECRET.sub(lambda m: m.group(1) + _MASK, text)
    text = _KEY_VALUE_SECRET.sub(lambda m: m.group(1) + m.group(2) + _mask_quoted(m.group(3)), text)
    text = _URL_USERINFO.sub(lambda m: m.group(1) + _MASK + m.group(3), text)
    return _SECRET_PATTERNS.sub(_MASK, text)


def redacted_preview(value: object, limit: int) -> str:
    """Redact (secret keys masked, secret patterns masked) then truncate to ``limit`` chars.

    Example:
        >>> redacted_preview({"path": "/tmp/a", "api_key": "k"}, 200)
        '{"api_key":"***","path":"/tmp/a"}'
    """
    if _is_container(value):
        try:
            redacted = _mask_secret_keys(value)
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
    ) -> dict[str, JsonValue]:
        """Return a redacted event; unknown ``fields`` are rejected to keep the schema closed.

        Example:
            >>> e = EventBuilder().build("session.started", hermes_session_id="s1", span_id="s1")
            >>> e["schema_version"], e["type"], e["seq"], e["span_id"], e["extra"]
            ('agenomic.hermes.event/v1', 'session.started', 1, 's1', {})
        """
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
                # Free-form fields (reason, explanation, error text) can carry credentials
                # in metadata mode too, so every field is redacted, not only content.
                event[key] = _redact_field(value)
        # The whole mapping is redacted, so its own credential-named keys are masked too.
        extra_doc = cast(dict[str, Any], _redact_field(dict(extra or {})))
        if content:
            extra_doc["content_hashes"] = {k: content_hash(v) for k, v in content.items()}
            if self.capture == "redacted_preview":
                extra_doc["previews"] = {
                    k: _MASK if is_secret_key(k) else redacted_preview(v, self.preview_chars)
                    for k, v in content.items()
                }
        event["extra"] = extra_doc
        return event


class ExporterStats(dict[str, JsonValue]):
    """``{"buffered", "dropped", "buffer_full", "last_flush_error"}`` for heartbeats.

    Example:
        >>> ExporterStats(buffered=0, dropped=0, buffer_full=False, last_flush_error=None)["dropped"]
        0
    """


def open_private(path: Path, flags: int) -> int:
    """Open ``path`` for writing as a file only its owner can read.

    ``os.open``'s mode applies only when the file is created, so an existing file is
    narrowed to ``0600`` too; a file owned by another user, or a symbolic link, is refused
    (``OSError``). Windows has no POSIX modes: the file is opened as is.

    Example:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "f"
        >>> p.write_text("x") and None
        >>> os.chmod(p, 0o644)
        >>> os.close(open_private(p, os.O_WRONLY | os.O_APPEND))
        >>> oct(p.stat().st_mode & 0o777) if sys.platform != "win32" else "0o600"
        '0o600'
    """
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if sys.platform == "win32":
        return fd
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid():
            raise PermissionError(f"spool file {path.name} is owned by another user")
        if st.st_mode & 0o777 != 0o600:
            os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd


class _Spool:
    """Append only JSONL file capped at ``max_bytes``; lines are already redacted events.

    The file (and its temporary copy) is ``0600`` even when it existed with a wider mode;
    a directory the spool creates is ``0700``.
    """

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

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
                line = (
                    json.dumps(event, separators=(",", ":"), ensure_ascii=False, default=str) + "\n"
                )
                cost = len(line.encode("utf-8"))
                if size + cost > self.max_bytes:
                    rejected.append(event)
                    continue
                size += cost
                lines.append(line)
            if lines:
                fd = open_private(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
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
            fd = open_private(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.writelines(line + "\n" for line in rest)
                fh.flush()
                os.fsync(fh.fileno())
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
        post: Callable[[list[dict[str, JsonValue]]], object],
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
        self._live_since_replay = 0
        self._last_replay_at = float("-inf")
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- producer side -------------------------------------------------
    def submit(self, event: dict[str, JsonValue]) -> bool:
        """Queue one already redacted event; ``False`` when it was dropped or spooled.

        Example:
            >>> ex = EventExporter(lambda events: None)
            >>> ex.submit(EventBuilder().build("adapter.loaded")), ex.submit({"no": "event_id"})
            (True, False)
            >>> ex.close()
            False
        """
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
            except (OSError, TypeError, ValueError) as e:
                logger.warning("event spool write failed: %s", type(e).__name__)
                rejected = events
            if rejected:
                self._count_drop(f"{reason}; spool full", len(rejected))
        else:
            self._count_drop(reason, len(events))

    # -- introspection -------------------------------------------------
    def stats(self) -> ExporterStats:
        """Exporter counters for the runtime heartbeat.

        Example:
            >>> ex = EventExporter(lambda events: None)
            >>> ex.stats()
            {'buffered': 0, 'dropped': 0, 'buffer_full': False, 'last_flush_error': None}
            >>> ex.close()
            True
        """
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
        """Events the server acknowledged.

        Example:
            >>> ex = EventExporter(lambda events: None)
            >>> ex.submit({"event_id": "e1"}), ex.flush(5.0), ex.delivered
            (True, True, 1)
            >>> ex.close()
            True
        """
        return self._delivered

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until the buffer is empty; ``False`` on timeout or when anything was dropped.

        Example:
            >>> ex = EventExporter(lambda events: None)
            >>> ex.flush(timeout=5.0)
            True
            >>> ex.close()
            True
        """
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
        """Refuse new events, drain, stop the worker. Undelivered events go to the spool.

        Example:
            >>> ex = EventExporter(lambda events: None)
            >>> ex.close()
            True
            >>> ex.submit({"event_id": "e1"})
            False
        """
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
                self._live_since_replay += 1
                if (
                    self._live_since_replay >= REPLAY_EVERY
                    or time.monotonic() - self._last_replay_at >= self._interval
                ):
                    self._replay_spool()
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
        self._live_since_replay = 0
        self._last_replay_at = time.monotonic()
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
