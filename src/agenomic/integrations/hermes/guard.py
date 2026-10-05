"""``agenomic-hermes-guard``: the complementary ``pre_tool_call`` shell hook.

Configured in Hermes with ``fail_closed: true``. It is a second, independent
check that the Agenomic adapter is loaded and that the instance is neither
paused, quarantined nor revoked, read from the status file the plugin keeps
fresh (``$HERMES_HOME/agenomic/status.json``). It makes no network call.

Upstream allows a ``fail_closed`` hook that exits non zero with an empty
stdout, so every failure path here prints a block directive on stdout and
exits 2, including internal errors. Allowing prints nothing and exits 0.

Example:
    >>> import io, json, os, tempfile
    >>> home = tempfile.mkdtemp()
    >>> out = io.StringIO()
    >>> main(stdin=io.StringIO("{}"), stdout=out, environ={"HERMES_HOME": home})
    2
    >>> json.loads(out.getvalue())["action"]
    'block'
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

STATUS_SCHEMA = "agenomic.hermes.status/v1"
DEFAULT_MAX_AGE_S = 120.0
BLOCK_EXIT = 2
_MAX_STDIN = 4 * 1024 * 1024
_BLOCKING_STATES = {"paused", "quarantined", "revoked"}
_FALLBACK = '{"action":"block","message":"Agenomic guard failed; the action was not executed."}\n'


def status_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``$HERMES_HOME/agenomic/status.json`` (``~/.hermes`` when ``HERMES_HOME`` is unset).

    Example:
        >>> str(status_path({"HERMES_HOME": "/h"}))
        '/h/agenomic/status.json'
    """
    env = os.environ if environ is None else environ
    home = env.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home).expanduser() / "agenomic" / "status.json"


def _parse_time(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def evaluate(
    status: Optional[dict[str, Any]],
    *,
    now: Optional[float] = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> Optional[str]:
    """Return a block message, or ``None`` to allow.

    Example:
        >>> evaluate(None)
        'Agenomic adapter status is missing; the plugin is not loaded. The action was not executed.'
        >>> from datetime import timezone
        >>> fresh = {"loaded": True, "instance_status": "active", "updated_at": "2026-10-05T12:00:00Z"}
        >>> evaluate(fresh, now=datetime(2026, 10, 5, 12, 0, 30, tzinfo=timezone.utc).timestamp()) is None
        True
    """
    if status is None:
        return "Agenomic adapter status is missing; the plugin is not loaded. The action was not executed."
    if status.get("loaded") is not True:
        return "Agenomic adapter is not loaded; the action was not executed."
    updated = _parse_time(status.get("updated_at"))
    current = time.time() if now is None else now
    if updated is None:
        return "Agenomic adapter status has no valid timestamp; the action was not executed."
    if current - updated > max_age_s:
        return "Agenomic adapter status is stale; the action was not executed."
    if updated - current > 60:
        return "Agenomic adapter status is dated in the future; the action was not executed."
    for key in ("instance_status", "effective_state"):
        value = status.get(key)
        if isinstance(value, str) and value in _BLOCKING_STATES:
            return f"Agenomic instance is {value}; the action was not executed."
    return None


def _read_status(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("status file is not an object")
    return data


def _block(stdout: IO[str], message: str) -> int:
    stdout.write(json.dumps({"action": "block", "message": message}) + "\n")
    stdout.flush()
    return BLOCK_EXIT


def main(
    argv: Optional[list[str]] = None,
    *,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Run the guard once. Returns the exit code (0 allow, 2 block)."""
    out = sys.stdout if stdout is None else stdout
    try:
        env = os.environ if environ is None else environ
        source = sys.stdin if stdin is None else stdin
        payload_text = source.read(_MAX_STDIN + 1)
        if len(payload_text) > _MAX_STDIN:
            return _block(out, "Agenomic guard input too large; the action was not executed.")
        payload = json.loads(payload_text) if payload_text.strip() else {}
        if not isinstance(payload, dict):
            return _block(
                out, "Agenomic guard input is not an object; the action was not executed."
            )
        max_age = float(env.get("AGENOMIC_HERMES_GUARD_MAX_AGE_S") or DEFAULT_MAX_AGE_S)
        try:
            status = _read_status(status_path(env))
        except (OSError, ValueError):
            return _block(
                out, "Agenomic adapter status is unreadable; the action was not executed."
            )
        message = evaluate(status, max_age_s=max_age)
        if message is not None:
            return _block(out, message)
        return 0
    except Exception:
        # Any failure still blocks: an empty stdout with a non zero exit would be allowed upstream.
        try:
            out.write(_FALLBACK)
            out.flush()
        except (OSError, ValueError):
            pass
        return BLOCK_EXIT


def cli() -> None:
    """Console script entry point."""
    try:
        code = main(sys.argv[1:])
    except BaseException:
        sys.stdout.write(_FALLBACK)
        sys.stdout.flush()
        code = BLOCK_EXIT
    raise SystemExit(code)
