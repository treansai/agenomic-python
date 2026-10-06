"""Hermes plugin entry point: ``register(ctx)``.

Installed as ``[project.entry-points."hermes_agent.plugins"] agenomic`` and
loaded by Hermes only when ``agenomic`` is listed in ``plugins.enabled``.

What it does (contracts verified against Hermes v2026.9.24, 0.21.5):

* observer hooks (``**kwargs`` callbacks, return ignored) mirror sessions,
  model calls, tool calls, subagents and approvals as
  ``agenomic.hermes.event/v1`` events, redacted before export;
* ``pre_tool_call`` and the ``tool_execution`` middleware gate covered tool
  actions on an Agenomic decision. In the agent loop the middleware runs
  OUTSIDE ``pre_tool_call`` (``agent/tool_executor.py``) and in
  ``model_tools.handle_function_call`` it runs inside it, so whichever runs
  first for a ``tool_call_id`` asks the gateway and the other reuses that
  decision. The middleware never raises before ``next_call`` (Hermes would
  skip the frame and execute: fail open); it returns an error result instead.
  After execution it reports the result with the signed permit;
* ``llm_request`` middleware adds ``X-Agenomic-Hermes-Session`` to requests
  sent to the Agenomic Model Gateway and changes nothing else;
* a heartbeat thread reports liveness and exporter stats, executes plugin
  commands and keeps ``$HERMES_HOME/agenomic/status.json`` fresh for
  ``agenomic-hermes-guard``.

The server decides the mode. Until it answered, the adapter behaves as in
enforce: no valid decision, no execution. This module never answers a Hermes
approval and is not a security boundary.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue, SecretStr

from agenomic.integrations.hermes import ADAPTER_VERSION, COMPATIBLE_HERMES
from agenomic.integrations.hermes.canonical import CanonicalError, arguments_hash, schema_hash
from agenomic.integrations.hermes.client import HermesApiError, RuntimeClient
from agenomic.integrations.hermes.config import (
    CONFIG_SCHEMA,
    GUARD_COMMAND,
    MODEL_GATEWAY_PATH,
    AdapterConfig,
    ConfigError,
    build_config,
    settings_from_context,
)
from agenomic.integrations.hermes.exporter import (
    EventBuilder,
    EventExporter,
    content_hash,
    create_private_temp,
    mask_text,
    now_iso,
    redact,
    redacted_preview,
)
from agenomic.integrations.hermes.guard import (
    _SHARING_BACKOFF_S,
    _SHARING_RETRIES,
    DEFAULT_MAX_AGE_S,
    GUARD_EPOCH_ENV,
    MIN_MAX_AGE_S,
    STATUS_SCHEMA,
    status_path,
)

logger = logging.getLogger("agenomic.integrations.hermes.plugin")

LocalMode = Literal["observe", "shadow", "enforce"]

OBSERVER_HOOKS = (
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_auxiliary_call",
    "post_auxiliary_call",
    "post_tool_call",
    "subagent_start",
    "subagent_stop",
    "pre_approval_request",
    "post_approval_response",
    "agent_loop_stopped",
)
_DECIDING_STATES = {"observe", "shadow", "enforce"}


def _is_blocking_state(state: object) -> bool:
    """Every known state but observe, shadow and enforce blocks, and so does any state this
    adapter does not know (a newer server): an unknown value is never allowed.
    ``None`` (no answer yet) is not a state: it means enforce without a decision.

    Example:
        >>> _is_blocking_state("paused"), _is_blocking_state("something_new")
        (True, True)
        >>> _is_blocking_state("enforce"), _is_blocking_state(None)
        (False, False)
    """
    return isinstance(state, str) and bool(state) and state not in _DECIDING_STATES


_BLOCKING_STATUS = {"paused", "quarantined", "revoked"}
_WRITE_TOOLS = {"write_file", "patch"}
_DELEGATE_TOOL = "delegate_task"
_DELEGATE_CONTROL_ACTIONS = {"list", "steer", "stop"}
_MAX_AUTH = 10_000
_MAX_REPORT_RETRIES = 10
_DEFAULT_HEARTBEAT_S = 15.0
APPROVAL_MESSAGE = (
    "Agenomic approval {approval_id} required; the action was not executed. "
    "Retry the same call after approval."
)
APPROVAL_IN_USE_MESSAGE = (
    "Agenomic approval {approval_id} authorizes a single execution and another call is "
    "using it; the action was not executed."
)
_SESSION_HEADER = "X-Agenomic-Hermes-Session"
_DECISION_STATUS = {"allow": 200, "observe": 200, "require_approval": 202, "deny": 403}
_DELEGATION_STATUS = {"allow": 200, "observe": 200, "deny": 403}
_MAX_ADMISSION_RETRIES_PER_TICK = 3
NO_AUTH_MESSAGE = "Agenomic: no valid authorization for this action"
#: Reason recorded when a call's arguments have no canonical form (``agenomic.canon/v1``).
NOT_CANONICAL_REASON = "arguments_not_canonical"
#: Reason recorded when a tool call is blocked because a cancel of its session or subagent is pending.
CANCEL_PENDING_REASON = "cancel_pending"
#: Reason recorded when a tool call is blocked because a ``pause``, ``quarantine`` or
#: ``revoke`` command was applied locally.
INSTANCE_STOPPED_REASON = "instance_stopped"
#: Reason codes of the local checks enforce applies after the gateway allowed; outside
#: enforce they are recorded as counterfactuals.
PROTECTED_PATH_REASON = "protected_path"
HERMES_INCOMPATIBLE_REASON = "hermes_incompatible"
FOREIGN_MUTATORS_REASON = "foreign_mutators_unconfirmed"
#: Reason recorded in observe when an approval required in enforce for the same action is
#: still outstanding locally: enforce would have retried under it instead of running.
APPROVAL_PENDING_REASON = "approval_pending"


@dataclass
class _Authorization:
    tool_call_id: str
    session_id: str
    tool: str
    logical_call_id: str
    attempt: int
    local_hash: str
    arguments: dict[str, Any]
    effective_mode: str
    record_id: Optional[str] = None
    permit: Optional[dict[str, Any]] = None
    server_hash: Optional[str] = None
    decision_id: Optional[str] = None
    # The delegation reservation this call queued for its children (``delegate_task``).
    delegation: Optional[list[Any]] = None
    # authorized -> executing -> done
    state: str = "authorized"
    # Set when observe or shadow let the call run with other arguments than the ones
    # authorized: the permit no longer describes the execution and is never reported.
    detached: bool = False


@dataclass
class _PendingEnd:
    final: bool
    status: str
    reason: str
    subagent_id: Optional[str]
    how: str


@dataclass
class _Pending:
    logical_call_id: str
    attempt: int
    approval_id: str
    # The invocation currently retrying under this approval. One approval authorizes one
    # execution: while an invocation holds the claim, any other identical invocation is
    # blocked instead of being resumed under the same identity (and the same permit).
    claimed_by: Optional[str] = None


@dataclass
class _Provisional:
    """A delegation reservation waiting for its action to be allowed.

    It belongs to one logical invocation at a time: ``claimed_by`` is the invocation
    deciding it now. An identical invocation arriving meanwhile reserves its own instead
    of sharing it, so every allowed ``delegate_task`` queues exactly one reservation.
    Unclaimed, it waits for a retry of the same action (after an approval or a transport
    error), which reuses it.
    """

    reservation: list[Any]
    claimed_by: Optional[str] = None
    settled: bool = False
    # Where it waits: (session, tool, arguments hash, approval id or ""). A reservation made
    # by an invocation that then required approval follows that approval, so the retry
    # resumed under it (and no other identical invocation) reuses it.
    slot: tuple[str, str, str, str] = ("", "", "", "")
    # The tool call it was reserved for: a retry under the same id takes it back first.
    tool_call_id: str = ""


@dataclass
class _Session:
    hermes_session_id: str
    platform: str = ""
    model: Optional[str] = None
    parent: Optional[str] = None
    subagent_id: Optional[str] = None
    delegation_id: Optional[str] = None
    agenomic_id: Optional[str] = None
    admitted: bool = False
    active: bool = True


@dataclass
class _Verdict:
    block: Optional[str] = None
    authorization: Optional[_Authorization] = None


@dataclass
class _ReportRetry:
    session_id: str
    body: dict[str, Any]
    attempts: int = 0


@dataclass
class _ExecutionPlan:
    proceed: bool
    error: Optional[str] = None
    auth: Optional[_Authorization] = None
    observe: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


_MAX_TERMINAL_ACKS = 10_000


def _queue_ack_retry(
    retries: deque[tuple[str, str, dict[str, Any]]],
    seen: set[str],
    terminal: OrderedDict[str, tuple[str, dict[str, Any]]],
    item: tuple[str, str, dict[str, Any]],
) -> None:
    """Queue a failed acknowledgement. When the bounded queue is full the oldest one is
    dropped, unless another acknowledgement of the same command is still queued:

    * a terminal one (``applied``, ``refused``) is kept as a tombstone, so the gateway's
      redelivery is answered with that result instead of executing the command again
      (a cancel cannot be applied again once its session ended);
    * a ``received`` one makes the command no longer seen, so its redelivery is executed.
    """
    evicted = retries[0] if retries.maxlen is not None and len(retries) == retries.maxlen else None
    retries.append(item)
    if evicted is None or any(queued[0] == evicted[0] for queued in retries):
        return
    command_id, status, detail = evicted
    if _ACK_RANK.get(status, 0) >= _ACK_RANK["applied"]:
        terminal[command_id] = (status, detail)
        while len(terminal) > _MAX_TERMINAL_ACKS:
            oldest, _ = terminal.popitem(last=False)
            seen.discard(oldest)
    else:
        seen.discard(command_id)


def _add_once(waiting: dict[str, list[str]], target: str, command_id: str) -> None:
    """Record ``command_id`` as waiting for ``target``'s end, once."""
    ids = waiting.setdefault(target, [])
    if command_id not in ids:
        ids.append(command_id)


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _is_guard_command(command: str) -> bool:
    """The configured guard and nothing else: ``agenomic-hermes-guard`` or an absolute path
    to it, one word. A wrapper or a compound command that merely mentions it is not.

    Example:
        >>> _is_guard_command("agenomic-hermes-guard"), _is_guard_command("/usr/bin/agenomic-hermes-guard")
        (True, True)
        >>> _is_guard_command("agenomic-hermes-guard && other-check")
        False
    """
    command = command.strip()
    if not command or any(c.isspace() for c in command):
        return False
    if command == GUARD_COMMAND:
        return True
    return os.path.isabs(command) and os.path.basename(command) == GUARD_COMMAND


def _redact_url(url: str) -> str:
    """``url`` without credentials: userinfo removed and every query value masked.

    Example:
        >>> _redact_url("https://u:p@gw.example/v1?key=abc&mode=x")
        'https://***@gw.example/v1?key=***&mode=***'
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "***"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]
    query = (
        "&".join(
            f"{pair.split('=', 1)[0]}=***" if pair else pair for pair in parts.query.split("&")
        )
        if parts.query
        else ""
    )
    return mask_text(urlunsplit((parts.scheme, netloc, parts.path, query, "")))


def _block(message: str) -> dict[str, str]:
    return {"action": "block", "message": message}


def _error_result(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _hermes_home(environ: Optional[Mapping[str, str]] = None) -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # Hermes absent or not initialised: fall back to its documented default
        env = os.environ if environ is None else environ
        return Path(env.get("HERMES_HOME") or str(Path.home() / ".hermes")).expanduser()


def _hermes_identity() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"version": None, "release_date": None, "commit": None}
    try:
        import hermes_cli

        out["version"] = str(getattr(hermes_cli, "__version__", "") or "") or None
        out["release_date"] = str(getattr(hermes_cli, "__release_date__", "") or "") or None
    except ImportError:
        return out
    try:
        from hermes_cli.build_info import get_code_identity

        sha = get_code_identity().get("sha")
        out["commit"] = str(sha) if sha else None
    except Exception:  # optional metadata; any failure leaves commit unknown
        out["commit"] = None
    return out


def _hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # Hermes absent or config unreadable
        return {}


def _result_is_error(result: object) -> bool:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(parsed, dict) and bool(parsed.get("error"))
    return isinstance(result, dict) and bool(result.get("error"))


def write_status(
    path: Path,
    *,
    loaded: bool,
    instance_status: str,
    effective_state: Optional[str],
    error: Optional[str] = None,
    epoch: Optional[str] = None,
) -> None:
    """Atomically write the status file read by ``agenomic-hermes-guard``.

    ``epoch`` binds the file to the Hermes process whose environment carries it in
    ``AGENOMIC_HERMES_GUARD_EPOCH``; the guard allows nothing without that match.

    Example:
        >>> import tempfile, json
        >>> p = Path(tempfile.mkdtemp()) / "agenomic" / "status.json"
        >>> write_status(p, loaded=True, instance_status="active", effective_state="observe")
        >>> json.loads(p.read_text())["loaded"]
        True
    """
    doc: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "loaded": loaded,
        "instance_status": instance_status,
        "effective_state": effective_state,
        "adapter_version": ADAPTER_VERSION,
        "pid": os.getpid(),
        "updated_at": now_iso(),
    }
    if error:
        doc["error"] = error
    if epoch:
        doc["epoch"] = epoch
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unpredictable name, created exclusively with mode 0600: a file or link planted in
    # the directory is never truncated or followed.
    tmp, fd = create_private_temp(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # On Windows the replace fails while a reader (the guard) has the file open; retry
    # briefly so a concurrent read never costs a heartbeat's status update.
    for attempt in range(_SHARING_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _SHARING_RETRIES - 1:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            time.sleep(_SHARING_BACKOFF_S)


class HermesAdapter:
    """State and behaviour of the Agenomic plugin inside one Hermes process.

    Built by :func:`register`; tests build it directly with a fake context.

    Example:
        >>> a = _demo_adapter()
        >>> a.config.endpoint, a.hermes_compatible
        ('https://a.example', True)
    """

    def __init__(
        self,
        config: AdapterConfig,
        token: SecretStr,
        *,
        ctx: object = None,
        client: Optional[RuntimeClient] = None,
        exporter: Optional[EventExporter] = None,
        hermes_home: Optional[Path] = None,
        start_threads: bool = True,
        identity: Optional[dict[str, Optional[str]]] = None,
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.client = client or RuntimeClient(
            config.endpoint,
            token.get_secret_value(),
            connect_s=config.timeouts.connect_s,
            decision_s=config.timeouts.decision_s,
            report_s=config.timeouts.report_s,
        )
        buf = config.buffer
        self.exporter = exporter or EventExporter(
            self.client.post_events,
            max_events=buf.max_events,
            max_bytes=buf.max_bytes,
            flush_interval_s=buf.flush_interval_s,
            batch_size=buf.batch_size,
            spool_path=buf.spool_path,
            spool_max_bytes=buf.spool_max_bytes,
        )
        self.builder = EventBuilder(config.capture.content, config.capture.preview_chars)
        self.home = hermes_home or _hermes_home()
        self.status_file = status_path({"HERMES_HOME": str(self.home)})
        self._start_threads = start_threads
        self._lock = threading.RLock()
        # Server state updates are ordered by when their request was sent: a response never
        # replaces a state applied from a request sent after it (heartbeat, hello, session
        # admission and authorize overlap across threads).
        self._state_lock = threading.Lock()
        self._state_sent = 0
        self._state_applied = 0
        self._started = False
        self._shut_down = False
        self._hello_ok = False
        self._hello_attempt_at = float("-inf")
        self._tools_sent = False
        self._effective_state: Optional[str] = None
        self._local_status: Optional[str] = None
        self._instance_id: Optional[str] = None
        self._profile: dict[str, Any] = {}
        # The heartbeat refreshes the guard status file, so even before the server sets an
        # interval it runs within a third of the guard's staleness deadline.
        self._heartbeat_s = min(_DEFAULT_HEARTBEAT_S, _guard_max_age_s() / 3)
        self._last_status_write = 0.0
        self._platform = ""
        self._identity = identity if identity is not None else _hermes_identity()
        self._contracts: dict[str, Any] = {
            "pre_tool_call": False,
            "tool_execution": False,
            "llm_request": False,
            "observer_hooks": [],
        }
        self._foreign: list[dict[str, str]] = []
        self._schema_hashes: dict[str, str] = {}
        self._sessions: dict[str, _Session] = {}
        self._agenomic_sessions: dict[str, str] = {}
        # A session cancel that names an Agenomic id the adapter does not know yet waits
        # while an admission is in flight (its id is published when the admission returns).
        self._admissions_in_flight = 0
        self._unresolved_cancels: dict[str, list[str]] = {}
        # Hermes ids of active sessions whose admission failed (the gateway may have created
        # the session and lost the answer): a cancel naming an unknown Agenomic id may be
        # theirs, so it stays unresolved until they are admitted or end.
        # Ordered: a retry that fails moves its session to the end, so one failing session
        # never starves the others of their per-heartbeat retries.
        self._unadmitted: OrderedDict[str, None] = OrderedDict()
        # Terminal session ends whose report failed transiently, retried every heartbeat;
        # the cancels waiting for them are acknowledged once they are reported.
        self._pending_ends: OrderedDict[str, _PendingEnd] = OrderedDict()
        self._end_locks = [threading.Lock() for _ in range(64)]
        self._children: dict[str, tuple[str, Optional[str]]] = {}
        self._delegations: dict[str, deque[list[Any]]] = {}
        # Hermes builds a delegate_task's children on the thread running that tool, inside
        # tool_execution: the reservation of the invocation running on this thread, and the
        # one each started child was built under, so a child never takes the reservation of
        # another invocation of the same parent.
        self._invocation = threading.local()
        self._child_delegations: dict[str, list[Any]] = {}
        # Reservations wait here, per (session, tool, arguments hash, approval id or ""),
        # until the action itself is allowed; a retry after an approval (of that approval)
        # or a transport error reuses them.
        # One invocation at a time claims an entry (see ``_Provisional``).
        # Every reservation waiting in a slot, oldest first: identical invocations that
        # failed (transport error) each keep their own until a retry takes it.
        self._provisional_delegations: dict[tuple[str, str, str, str], list[_Provisional]] = {}
        # Keyed by (session, tool, tool_call_id): providers reuse call ids across sessions,
        # and an authorization must never serve another session's or tool's call.
        self._auth: OrderedDict[tuple[str, str, str], _Authorization] = OrderedDict()
        # Call identities (session, tool, tool_call_id) whose authorization is in flight.
        self._authorizing: set[tuple[str, str, str]] = set()
        # Every approval issued for an identical action, in issue order: concurrent identical
        # calls can each get their own approval, and each keeps its own identity.
        self._pending: dict[tuple[str, str, str], list[_Pending]] = {}
        self._post_status: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        # Calls whose arguments had no canonical form and were already recorded, so the
        # second gate of the same call does not record it again.
        # Which gate ("pre" or "execution") recorded a call's non-canonical arguments.
        self._not_canonical: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        # Which gate recorded a call's local checks in observe (same pattern).
        self._observed_local: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._report_retries: deque[_ReportRetry] = deque(maxlen=1000)
        self._report_lock = threading.Lock()
        self._commands_seen: set[str] = set()
        # Acknowledgements that failed in transport; retried on every tick until accepted.
        self._ack_retries: deque[tuple[str, str, dict[str, Any]]] = deque(maxlen=500)
        # Serializes every read-modify-write of the ack retry queue (heartbeat thread and
        # terminal callbacks): a queued terminal ack is never lost to a concurrent rebuild.
        self._ack_lock = threading.Lock()
        # Terminal results whose acknowledgement was dropped from the full retry queue.
        self._terminal_acks: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
        # Every cancel command waiting for the end of a session or subagent, oldest first:
        # each one is acknowledged when Hermes reports that end.
        self._cancel_sessions: dict[str, list[str]] = {}
        self._cancel_subagents: dict[str, list[str]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Serializes status writes so a heartbeat in flight cannot undo shutdown's "not loaded".
        self._status_lock = threading.Lock()
        # Per adapter: published in the Hermes process environment at install, so only
        # the shell hooks of this process accept the status file it writes.
        self._guard_epoch = secrets.token_hex(16)

    # ------------------------------------------------------------------
    # installation and lifecycle
    # ------------------------------------------------------------------
    def install(self, ctx: object) -> None:
        """Register hooks and middleware on a Hermes ``PluginContext`` and probe them.

        Example:
            >>> import types
            >>> seen = []
            >>> ctx = types.SimpleNamespace(register_hook=lambda n, cb: seen.append(n),
            ...     register_middleware=lambda k, cb: seen.append(k))
            >>> _demo_adapter().install(ctx)
            >>> seen[-3:]
            ['pre_tool_call', 'tool_execution', 'llm_request']
        """
        self.ctx = ctx
        os.environ[GUARD_EPOCH_ENV] = self._guard_epoch
        # Duck typed Hermes ``PluginContext``: a missing method is caught and reported below.
        hermes_ctx = cast(Any, ctx)
        registered: list[str] = []
        for name in OBSERVER_HOOKS:
            try:
                hermes_ctx.register_hook(name, getattr(self, name))
                registered.append(name)
            except Exception as exc:  # report the missing contract instead of failing the load
                logger.warning("hook %s not registered: %s", name, type(exc).__name__)
        try:
            hermes_ctx.register_hook("pre_tool_call", self.pre_tool_call)
            self._contracts["pre_tool_call"] = True
        except Exception as exc:
            logger.error("pre_tool_call not registered: %s", type(exc).__name__)
        for kind, cb in (
            ("tool_execution", self.tool_execution),
            ("llm_request", self.llm_request),
        ):
            try:
                hermes_ctx.register_middleware(kind, cb)
                self._contracts[kind] = True
            except Exception as exc:
                logger.error("%s middleware not registered: %s", kind, type(exc).__name__)
        self._contracts["observer_hooks"] = registered
        self._probe_registrations()
        on_unload = getattr(ctx, "on_unload", None)
        if callable(on_unload):
            try:
                on_unload(self.shutdown)
            except Exception as exc:
                logger.debug("on_unload not available: %s", type(exc).__name__)
        self._write_status()
        self._emit(
            "adapter.loaded",
            None,
            extra={
                "adapter_version": ADAPTER_VERSION,
                "hermes": self._identity,
                "contracts": self._contracts,
                "compatible": self.hermes_compatible,
            },
        )

    def _manager(self) -> Any:
        return getattr(self.ctx, "_manager", None)

    def _probe_registrations(self) -> None:
        manager = self._manager()
        if manager is None:
            return
        hooks = getattr(manager, "_hooks", {}) or {}
        middleware = getattr(manager, "_middleware", {}) or {}
        self._contracts["pre_tool_call"] = any(
            cb == self.pre_tool_call for cb in hooks.get("pre_tool_call", [])
        )
        self._contracts["tool_execution"] = any(
            cb == self.tool_execution for cb in middleware.get("tool_execution", [])
        )
        self._contracts["llm_request"] = any(
            cb == self.llm_request for cb in middleware.get("llm_request", [])
        )

    def foreign_mutators(self) -> list[dict[str, str]]:
        """Callbacks other than ours that can change arguments or execution.

        Other plugins' ``pre_tool_call`` callbacks (they may return ``modify``),
        any ``tool_request`` middleware and other ``tool_execution`` middleware.
        Our own guard shell hook is not counted.

        Example:
            >>> _demo_adapter().foreign_mutators()  # no Hermes plugin manager here
            []
        """
        manager = self._manager()
        if manager is None:
            return []
        found: list[dict[str, str]] = []
        try:
            hooks = getattr(manager, "_hooks", {}) or {}
            middleware = getattr(manager, "_middleware", {}) or {}
            candidates: list[tuple[str, str, Any]] = [
                ("hook", "pre_tool_call", cb) for cb in list(hooks.get("pre_tool_call", []))
            ]
            candidates += [
                ("middleware", "tool_request", cb)
                for cb in list(middleware.get("tool_request", []))
            ]
            candidates += [
                ("middleware", "tool_execution", cb)
                for cb in list(middleware.get("tool_execution", []))
            ]
            for kind, name, cb in candidates:
                if cb == self.pre_tool_call or cb == self.tool_execution:
                    continue
                # Never repr(): a callable object's repr can carry its fields (tokens,
                # prompts, configuration), and this label leaves the process unredacted.
                label = str(
                    getattr(cb, "__qualname__", None)
                    or getattr(cb, "__name__", None)
                    or f"<{type(cb).__qualname__} instance>"
                )
                # Only the guard itself is ours: a wrapper or a compound command that merely
                # mentions it (``agenomic-hermes-guard && other``) is foreign.
                prefix = "shell_hook[pre_tool_call:"
                if (
                    label.startswith(prefix)
                    and label.endswith("]")
                    and _is_guard_command(label[len(prefix) : -1])
                ):
                    continue
                found.append(
                    {
                        "kind": kind,
                        "name": name,
                        # Sent in /hello without the event pipeline: a shell hook's label is
                        # its whole command line, which can carry a token.
                        "callback": mask_text(label)[:200],
                        "module": mask_text(str(getattr(cb, "__module__", "") or ""))[:200],
                    }
                )
        except Exception as exc:  # defensive: the manager internals are not a public API
            logger.debug("foreign mutator scan failed: %s", type(exc).__name__)
            return [{"kind": "unknown", "name": "scan_failed", "callback": "", "module": ""}]
        return found

    @property
    def hermes_compatible(self) -> bool:
        """``hermes_cli.__version__`` is in ``COMPATIBLE_HERMES``.

        Example:
            >>> _demo_adapter().hermes_compatible
            True
        """
        return self._identity.get("version") in COMPATIBLE_HERMES

    def _ensure_started(self, platform: str = "") -> None:
        if platform and not self._platform:
            self._platform = platform
        with self._lock:
            # A shut down adapter never starts again (no heartbeat, no atexit hook).
            first = not self._started and not self._shut_down
            self._started = True
        if not first:
            return
        try:
            if self._hello():
                try:
                    self.discover_tools()
                except HermesApiError as exc:
                    logger.warning("tool discovery failed (%s)", exc.code)
        except Exception as exc:
            # Never fatal: the heartbeat thread started below retries hello and discovery
            # on every tick, and keeps the guard status and command polling alive.
            logger.warning("Agenomic start up failed: %s", type(exc).__name__)
        try:
            self._write_status()
        except Exception as exc:
            logger.warning("guard status write failed: %s", type(exc).__name__)
        if self._start_threads:
            self._thread = threading.Thread(
                target=self._heartbeat_loop, name="agenomic-hermes-heartbeat", daemon=True
            )
            self._thread.start()
            atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Stop the heartbeat, mark the guard status not loaded, drain the exporter (bounded),
        then close the HTTP client and drop the ``atexit`` hook. Idempotent.

        Example:
            >>> a = _demo_adapter()
            >>> a.shutdown()
            >>> a.exporter.submit({"event_id": "e1"})
            False
            >>> json.loads(a.status_file.read_text())["loaded"]
            False
            >>> a.client.closed
            True
            >>> a.shutdown()  # a second call does nothing
        """
        with self._lock:
            if self._shut_down:
                return
            self._shut_down = True
        # The hook holds this bound method, and with it the adapter, its client and its
        # exporter: a reload must not keep every unloaded adapter alive until exit.
        atexit.unregister(self.shutdown)
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)
        # No callback of this adapter will ask the gateway any more: the guard must block now
        # instead of allowing tool calls until the status file goes stale.
        self._write_status()
        try:
            self.exporter.close(2.0)
        except Exception as exc:
            logger.debug("exporter close failed: %s", type(exc).__name__)
        # The heartbeat (acks, report retries) is joined and the exporter drained: nothing
        # of this adapter sends any more. A late callback gets a transport error and, in
        # enforce, blocks.
        try:
            self.client.close()
        except Exception as exc:
            logger.debug("client close failed: %s", type(exc).__name__)

    # ------------------------------------------------------------------
    # server state
    # ------------------------------------------------------------------
    def local_mode(self) -> LocalMode:
        """``observe`` / ``shadow`` / ``enforce`` from the last server answer.

        Unknown (no answer yet) and every enforce like state count as enforce.

        Example:
            >>> a = _demo_adapter()
            >>> a.local_mode()  # no server answer yet
            'enforce'
            >>> a.tick()
            >>> a.local_mode()
            'observe'
        """
        if self._local_status in _BLOCKING_STATUS:
            return "enforce"
        state = self._effective_state
        if state == "observe":
            return "observe"
        if state == "shadow":
            return "shadow"
        return "enforce"

    def _state_request(self) -> int:
        """Sequence number of a request about to be sent whose response carries a state."""
        with self._state_lock:
            self._state_sent += 1
            return self._state_sent

    def _note_state_seq(self, seq: int) -> None:
        """Record that a request sent as ``seq`` was answered without changing the state."""
        with self._state_lock:
            self._state_applied = max(self._state_applied, seq)

    def _set_state(self, state: object, seq: Optional[int] = None) -> bool:
        """Apply a server state unless a request sent after ``seq`` already applied one.
        Without ``seq`` the state counts as the newest. Returns whether it was applied."""
        if not (isinstance(state, str) and state):
            return False
        with self._state_lock:
            if seq is None:
                self._state_sent += 1
                seq = self._state_sent
            if seq < self._state_applied:
                return False
            self._state_applied = seq
            self._effective_state = state
            return True

    def _instance_status(self) -> str:
        if self._local_status:
            return self._local_status
        state = self._effective_state
        if state is None:
            return "unknown"
        return state if state in _BLOCKING_STATUS else "active"

    def _gates_registered(self) -> bool:
        # Both gates are needed: in each Hermes ordering one of them runs after the
        # other mutators and is the only place an argument change after authorization is
        # detected. With either missing, the guard keeps blocking every tool.
        return bool(self._contracts["pre_tool_call"] and self._contracts["tool_execution"])

    def _refresh_status_if_due(self) -> None:
        """Rewrite the guard status file when a heartbeat interval has passed since the last
        write: retry queues drained against a slow gateway never let it go stale (the
        guard would then block every call, in shadow and observe too)."""
        if time.monotonic() - self._last_status_write >= self._heartbeat_s:
            self._write_status()

    def _write_status(self) -> None:
        self._last_status_write = time.monotonic()
        if not self._gates_registered():
            logger.error("no enforcement gate registered; the guard keeps blocking tools")
        with self._status_lock:
            try:
                write_status(
                    self.status_file,
                    # After shutdown every write says "not loaded", so a late heartbeat cannot
                    # reopen the guard.
                    loaded=self._gates_registered() and not self._stop.is_set(),
                    instance_status=self._instance_status(),
                    effective_state=self._effective_state,
                    epoch=self._guard_epoch,
                )
            except OSError as exc:
                logger.warning("status file not written: %s", type(exc).__name__)

    def _provider(self) -> dict[str, Any]:
        model = _hermes_config().get("model")
        if not isinstance(model, dict):
            return {}
        return {
            # Sent outside the event pipeline: a custom provider id can carry a credential.
            "provider": mask_text(_str(model.get("provider"))) or None,
            "base_url": _str(model.get("base_url")) or None,
        }

    def _compat_results(self) -> list[dict[str, str]]:
        cfg = _hermes_config()
        results: list[dict[str, str]] = []

        def add(check: str, ok: Optional[bool], detail: str = "") -> None:
            status = "unknown" if ok is None else ("pass" if ok else "fail")
            results.append({"check": check, "status": status, "detail": detail})

        add("hermes_version_compatible", self.hermes_compatible, str(self._identity.get("version")))
        add("pre_tool_call_registered", bool(self._contracts["pre_tool_call"]))
        add("tool_execution_registered", bool(self._contracts["tool_execution"]))
        add("llm_request_registered", bool(self._contracts["llm_request"]))
        try:
            from hermes_cli import plugins_dispatch

            add(
                "pre_tool_call_fail_closed",
                "pre_tool_call"
                in getattr(plugins_dispatch, "_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS", set()),
            )
        except ImportError:
            add("pre_tool_call_fail_closed", None, "hermes_cli not importable")
        hooks = cfg.get("hooks") if isinstance(cfg.get("hooks"), dict) else {}
        entries = hooks.get("pre_tool_call") if isinstance(hooks, dict) else None
        guard = [
            e
            for e in (entries if isinstance(entries, list) else [])
            if isinstance(e, dict) and _is_guard_command(_str(e.get("command")))
        ]
        add(
            "guard_hook_configured",
            bool(guard) and all(bool(e.get("fail_closed") or e.get("failClosed")) for e in guard),
        )
        skills = cfg.get("skills") if isinstance(cfg.get("skills"), dict) else {}
        add(
            "skills_write_approval",
            bool(isinstance(skills, dict) and skills.get("write_approval") is True),
        )
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        timeout = plugins.get("hook_callback_timeout", 30) if isinstance(plugins, dict) else 30
        guard_timeout = max((int(e.get("timeout", 60)) for e in guard), default=0)
        add(
            "hook_callback_timeout",
            isinstance(timeout, (int, float)) and (timeout == 0 or timeout >= guard_timeout),
            str(timeout),
        )
        provider = self._provider()
        add(
            "model_gateway_provider",
            provider.get("provider") == "custom"
            and _str(provider.get("base_url")).startswith(
                self.config.endpoint + MODEL_GATEWAY_PATH
            ),
        )
        return results

    def _hello_body(self, foreign: list[dict[str, str]]) -> dict[str, Any]:
        provider = self._provider()
        if provider.get("base_url"):
            # Sent without the event redaction pipeline: userinfo and query values (an
            # api_key, a signed parameter) never leave the process. The compatibility
            # check above uses the local, unredacted value.
            provider = {**provider, "base_url": _redact_url(_str(provider["base_url"]))}
        return {
            "hermes": self._identity,
            "adapter": {"version": ADAPTER_VERSION, "config_schema": CONFIG_SCHEMA},
            "contracts": self._contracts,
            "foreign_mutators": foreign,
            "provider": provider,
            "compat_results": self._compat_results(),
            # Sent outside the event pipeline: masked like any exported text.
            "platform": mask_text(self._platform or "cli"),
        }

    def _hello(self) -> bool:
        self._hello_attempt_at = time.monotonic()
        foreign = self.foreign_mutators()
        seq = self._state_request()
        try:
            resp = self.client.hello(self._hello_body(foreign))
        except HermesApiError as exc:
            logger.warning("Agenomic hello failed (%s)", exc.code)
            return False
        # Only a delivered hello updates the list the server knows; a failed one is re-sent
        # on the next tick because the list still differs.
        self._foreign = foreign
        self._hello_ok = True
        self._instance_id = _str(resp.get("instance_id")) or None
        self._set_state(resp.get("effective_state"), seq)
        profile = resp.get("profile")
        if isinstance(profile, dict) and isinstance(profile.get("document"), dict):
            self._profile = cast(dict[str, Any], profile["document"])
        interval = resp.get("heartbeat_interval_secs")
        if isinstance(interval, (int, float)) and 1 <= interval <= 3600:
            # The heartbeat also refreshes the guard status file, so it must run well within
            # the guard's staleness deadline or healthy tool calls would be blocked.
            self._heartbeat_s = min(float(interval), _guard_max_age_s() / 3)
        return True

    def discover_tools(self) -> int:
        """Send the Hermes tool registry with schema hashes. Returns the number sent.

        Example:
            >>> _demo_adapter().discover_tools()  # Hermes' tool registry is not importable here
            0
        """
        try:
            from tools.registry import registry
        except ImportError:
            return 0
        plugin_tools: set[str] = set()
        manager = self._manager()
        if manager is not None:
            plugin_tools = set(getattr(manager, "_plugin_tool_names", set()) or set())
        tools: list[dict[str, Any]] = []
        withheld = 0
        for name in sorted(registry.get_all_tool_names()):
            toolset = _str(registry.get_toolset_for_tool(name))
            if mask_text(name) != name or mask_text(toolset) != toolset:
                # A credential-shaped identifier never leaves the process: the tool is not
                # catalogued (in enforce the gateway then refuses it as unknown).
                withheld += 1
                continue
            schema = registry.get_schema(name)
            if not isinstance(schema, dict):
                continue
            try:
                digest = schema_hash(schema)
            except CanonicalError:
                continue
            self._schema_hashes[name] = digest
            entry: dict[str, Any] = {
                "tool_name": name,
                "source": "builtin",
                "schema_hash": digest,
                # Sent outside the event pipeline: a default, example or description can
                # carry a credential. The hash above is of the original, local schema.
                "input_schema": redact(schema.get("parameters"))
                if isinstance(schema.get("parameters"), dict)
                else {},
            }
            if toolset.startswith("mcp-"):
                entry["source"] = "mcp"
                entry["mcp_server"] = toolset[4:]
            elif name in plugin_tools:
                entry["source"] = "plugin"
            tools.append(entry)
        if withheld:
            logger.warning("%d tool(s) with a credential-shaped name were not catalogued", withheld)
        for start in range(0, len(tools), 500):
            self.client.tools_discovered(tools[start : start + 500])
        self._tools_sent = True
        return len(tools)

    def _schema_hash_for(self, tool: str) -> Optional[str]:
        digest = self._schema_hashes.get(tool)
        if digest is not None:
            return digest
        try:
            from tools.registry import registry

            schema = registry.get_schema(tool)
            if isinstance(schema, dict):
                digest = schema_hash(schema)
                self._schema_hashes[tool] = digest
        except (ImportError, CanonicalError):
            return None
        return digest

    # ------------------------------------------------------------------
    # heartbeat and commands
    # ------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            try:
                self.tick()
            except Exception as exc:  # the loop must survive any single failure
                logger.warning("heartbeat tick failed: %s", type(exc).__name__)

    def tick(self) -> None:
        """One heartbeat: hello and tool discovery if pending, heartbeat, commands, retries.

        Example:
            >>> a = _demo_adapter()
            >>> a.tick()
            >>> json.loads(a.status_file.read_text())["effective_state"]
            'observe'
        """
        try:
            if not self._hello_ok or self.foreign_mutators() != self._foreign:
                self._hello()
            if self._hello_ok and not self._tools_sent:
                try:
                    self.discover_tools()
                except HermesApiError as exc:
                    logger.warning("tool discovery failed (%s)", exc.code)
                except Exception as exc:
                    # A third party registry failing must never stop the heartbeat: queued
                    # pause and cancel commands are fetched below. Retried next tick.
                    logger.warning("tool discovery failed: %s", type(exc).__name__)
            with self._lock:
                active: list[JsonValue] = [
                    s.hermes_session_id for s in self._sessions.values() if s.active
                ]
            seq = self._state_request()
            resp = self.client.heartbeat(
                {"active_sessions": active, "exporter": dict(self.exporter.stats())}
            )
            self._set_state(resp.get("effective_state"), seq)
            commands = resp.get("commands")
            if isinstance(commands, list):
                for command in commands:
                    if isinstance(command, dict):
                        self.handle_command(command)
        except HermesApiError as exc:
            logger.warning("Agenomic heartbeat failed (%s)", exc.code)
        finally:
            self._retry_admissions()
            self._retry_ends()
            self._retry_acks()
            self._retry_reports()
            self._write_status()

    def _ack(self, command_id: str, status: str, detail: dict[str, Any]) -> bool:
        try:
            self.client.ack_command(command_id, status, detail)
        except HermesApiError as exc:
            logger.warning("command %s ack %s failed (%s)", command_id, status, exc.code)
            if exc.retryable:
                with self._ack_lock:
                    _queue_ack_retry(
                        self._ack_retries,
                        self._commands_seen,
                        self._terminal_acks,
                        (command_id, status, detail),
                    )
            return False
        self._drop_superseded_acks(command_id, status)
        self._emit("command." + status, None, extra={"command_id": command_id, "detail": detail})
        return True

    def _drop_superseded_acks(self, command_id: str, status: str) -> None:
        rank = _ACK_RANK.get(status, 0)
        with self._ack_lock:
            kept = [
                item
                for item in self._ack_retries
                if item[0] != command_id or _ACK_RANK.get(item[1], 0) > rank
            ]
            if len(kept) != len(self._ack_retries):
                self._ack_retries.clear()
                self._ack_retries.extend(kept)

    def _retry_acks(self) -> None:
        with self._ack_lock:
            pending = len(self._ack_retries)
        for _ in range(pending):
            with self._ack_lock:
                try:
                    command_id, status, detail = self._ack_retries.popleft()
                except IndexError:
                    return
            self._ack(command_id, status, detail)
            self._refresh_status_if_due()

    def handle_command(self, command: Mapping[str, JsonValue]) -> None:
        """Execute one plugin command. ``applied`` is only acknowledged once observed.

        Example:
            >>> a = _demo_adapter()
            >>> a.handle_command({"id": "c1", "kind": "pause", "target_kind": "instance"})
            >>> json.loads(a.status_file.read_text())["instance_status"]
            'paused'
        """
        command_id = _str(command.get("id"))
        if command_id in self._terminal_acks:
            # Its terminal acknowledgement was dropped: answer the redelivery with it.
            status, detail = self._terminal_acks.pop(command_id)
            self._ack(command_id, status, detail)
            return
        if not command_id or command_id in self._commands_seen:
            return
        self._commands_seen.add(command_id)
        kind = _str(command.get("kind"))
        target_kind = _str(command.get("target_kind")) or "instance"
        target = _str(command.get("target_ref"))
        if _str(command.get("status")) in ("requested", ""):
            self._ack(command_id, "received", {"executor": "plugin"})
        if target_kind == "instance" and kind in ("pause", "revoke"):
            with self._lock:  # atomic with the final admission of a tool call
                self._local_status = "paused" if kind == "pause" else "revoked"
            self._write_status()
            self._ack(command_id, "applied", {"local_state": self._local_status})
        elif target_kind == "instance" and kind == "resume":
            with self._lock:
                self._local_status = None
            self._write_status()
            self._ack(command_id, "applied", {"local_state": "active"})
        elif target_kind == "instance" and kind == "quarantine":
            # Quarantine is a process stop by the supervisor; the plugin only blocks locally.
            with self._lock:
                self._local_status = "quarantined"
            self._write_status()
        elif kind == "cancel" and target_kind == "subagent":
            self._cancel_subagent(command_id, target)
        elif kind == "cancel" and target_kind == "session":
            self._cancel_session(command_id, target)
        else:
            self._ack(command_id, "refused", {"reason": "unsupported_command", "kind": kind})

    def _interrupt_subagent(self, subagent_id: str) -> bool:
        try:
            from tools.delegate_tool_registry import interrupt_subagent
        except ImportError:
            return False
        try:
            return bool(interrupt_subagent(subagent_id))
        except Exception as exc:
            logger.debug("interrupt_subagent failed: %s", type(exc).__name__)
            return False

    def _cancel_subagent(self, command_id: str, subagent_id: str) -> None:
        if not subagent_id:
            self._ack(command_id, "refused", {"reason": "missing_target"})
            return
        # Registered before the interrupt: the subagent may end on another thread before
        # the interrupt returns, and that end must find the command waiting.
        with self._lock:
            _add_once(self._cancel_subagents, subagent_id, command_id)
        if not self._interrupt_subagent(subagent_id) and self._withdraw_cancel(
            self._cancel_subagents, subagent_id, command_id
        ):
            self._ack(command_id, "refused", {"reason": "subagent_not_running"})

    def _cancel_session(self, command_id: str, target: str) -> None:
        with self._lock:
            sid = self._agenomic_sessions.get(target, target)
            # The active check and the registration are one step: a session ending in
            # between would otherwise leave a waiter no terminal event can reach.
            session = self._sessions.get(sid)
            active = session is not None and session.active
            if active and session is not None:
                _add_once(self._cancel_sessions, sid, command_id)
                if session.subagent_id:
                    _add_once(self._cancel_subagents, session.subagent_id, command_id)
            elif session is None and (self._admissions_in_flight or self._unadmitted):
                # Possibly the Agenomic id of a session being admitted right now, or of one
                # whose admission answer was lost: decided once its id is published (or the
                # session ends unadmitted), never refused before.
                _add_once(self._unresolved_cancels, target, command_id)
                return
        if not active or session is None:
            self._ack(command_id, "refused", {"reason": "session_not_active"})
            return
        if session.subagent_id and not self._interrupt_subagent(session.subagent_id):
            # Not interruptible: the command waits for the session's end only.
            self._withdraw_cancel(self._cancel_subagents, session.subagent_id, command_id)
        # Root sessions expose no interrupt handle to plugins: further tool calls are blocked and
        # the command is applied once Hermes reports the session's end.

    def _withdraw_cancel(self, waiting: dict[str, list[str]], target: str, command_id: str) -> bool:
        """Remove a waiting cancel; ``False`` when it is gone already (its end was observed
        and acknowledged meanwhile)."""
        with self._lock:
            ids = waiting.get(target)
            if not ids or command_id not in ids:
                return False
            ids.remove(command_id)
            if not ids:
                del waiting[target]
            return True

    def _cancel_pending(self, sid: str) -> bool:
        return self._cancel_kind(sid) is not None

    def _cancel_kind(self, sid: str) -> Optional[str]:
        """``"session"`` or ``"subagent"`` when a cancel of this session (or of the subagent
        it runs as) is waiting for Hermes to report its end; ``None`` otherwise."""
        if not sid:
            return None
        with self._lock:
            session = self._sessions.get(sid)
            subagent_id = (
                session.subagent_id if session else self._children.get(sid, (None, None))[1]
            )
        if sid in self._cancel_sessions:
            return "session"
        if subagent_id and subagent_id in self._cancel_subagents:
            return "subagent"
        return None

    def _observe_terminal(
        self, sid: str, subagent_id: Optional[str], how: str, *, delivered: bool = True
    ) -> None:
        """Every cancel waiting for this end (of the session and of the subagent it runs
        as, possibly distinct commands) is settled, each exactly once: ``applied`` when the
        terminal end reached the control plane, ``refused`` when the gateway refused that
        report for good (the session ended here, but no applied cancel can be claimed)."""
        command_ids: list[str] = []
        with self._lock:
            pending = (
                self._cancel_sessions.pop(sid, []) if sid else [],
                self._cancel_subagents.pop(subagent_id, []) if subagent_id else [],
            )
        for waiting in pending:
            for command_id in waiting:
                if command_id not in command_ids:
                    command_ids.append(command_id)
        for command_id in command_ids:
            if delivered:
                self._ack(command_id, "applied", {"observed": how, "hermes_session_id": sid})
            else:
                self._ack(
                    command_id,
                    "refused",
                    {"reason": "session_end_refused", "observed": how, "hermes_session_id": sid},
                )

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _root_of(self, sid: Optional[str]) -> Optional[str]:
        if not sid:
            return None
        seen = 0
        current = sid
        with self._lock:
            while seen < 16:
                session = self._sessions.get(current)
                parent = session.parent if session else self._children.get(current, (None, None))[0]
                if not parent:
                    return current
                current = parent
                seen += 1
        return current

    def _emit(
        self,
        event_type: str,
        sid: Optional[str],
        *,
        content: Optional[Mapping[str, object]] = None,
        extra: Optional[Mapping[str, object]] = None,
        **fields: Any,
    ) -> None:
        try:
            fields.setdefault("trace_id", self._root_of(sid))
            event = self.builder.build(
                event_type, content=content, extra=extra, hermes_session_id=sid or None, **fields
            )
            self.exporter.submit(event)
        except Exception as exc:  # telemetry never interferes with the agent
            logger.debug("event %s not emitted: %s", event_type, type(exc).__name__)

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    def _session(self, sid: str, platform: str = "", model: Optional[str] = None) -> _Session:
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                session = _Session(sid, platform=platform, model=model)
                link = self._children.get(sid)
                if link is not None:
                    session.parent, session.subagent_id = link
                    session.delegation_id = self._take_delegation(
                        link[0], self._child_delegations.pop(sid, None)
                    )
                self._sessions[sid] = session
            if platform and not session.platform:
                session.platform = platform
            if model and not session.model:
                session.model = model
            return session

    def _take_delegation(
        self, parent: str, reservation: Optional[list[Any]] = None
    ) -> Optional[str]:
        """The delegation of the reservation the child was built under (``None`` once it
        is used up or dropped: another invocation's reservation is never taken); without
        one (no authorization on the delegating thread), the oldest one of the parent."""
        queue = self._delegations.get(parent)
        if reservation is not None:
            for i, queued in enumerate(queue or ()):
                if queued is reservation and queued[1] > 0:
                    queued[1] -= 1
                    if queued[1] <= 0:
                        del queue[i]  # type: ignore[union-attr]
                    return str(queued[0])
            return None
        while queue:
            head = queue[0]
            if head[1] <= 0:
                queue.popleft()
                continue
            head[1] -= 1
            return str(head[0])
        return None

    def _admit(self, session: _Session) -> None:
        if session.admitted:
            return
        body: dict[str, Any] = {
            "hermes_session_id": session.hermes_session_id,
            "platform": mask_text(session.platform or self._platform or "cli"),
        }
        if session.model:
            # Sent outside the event pipeline: a custom model id can carry a credential.
            body["model"] = mask_text(session.model)
        if session.parent:
            body["parent_hermes_session_id"] = session.parent
        if session.subagent_id:
            body["subagent_id"] = session.subagent_id
        if session.delegation_id:
            body["delegation_id"] = session.delegation_id
        seq = self._state_request()
        with self._lock:
            self._admissions_in_flight += 1
        resp: Optional[dict[str, Any]] = None
        # Whether the gateway may have created the session without this process learning
        # its id (a transient failure): only then is it retried and may it hold cancels.
        lost = True
        try:
            resp = self.client.create_session(body)
            info = resp.get("session")
            if not (isinstance(info, dict) and isinstance(info.get("id"), str) and info["id"]):
                # Without the gateway's id no cancel naming it could be matched: handled
                # as a failed admission, retried, instead of admitted.
                logger.warning("session admission answer without a session id")
                resp = None
        except HermesApiError as exc:
            logger.warning("session admission failed (%s)", exc.code)
            lost = exc.retryable
        finally:
            pending = self._finish_admission(session, resp, lost=lost)
        for command_id, target in pending:
            self._cancel_session(command_id, target)
        if resp is None:
            return
        self._set_state(resp.get("effective_state"), seq)

    def _retry_admissions(self) -> None:
        """Retry the admission of active sessions whose admission failed, independently of
        their tool calls: an idle session would otherwise never publish its id, and a
        cancel naming it would wait unresolved until the session ends."""
        with self._lock:
            pending = [
                self._sessions[sid]
                for sid in self._unadmitted
                if sid in self._sessions and self._sessions[sid].active
            ]
        # Bounded per heartbeat, and stopped at the first failure (the gateway is likely
        # unavailable): a backlog never delays the status file past the guard's deadline.
        for session in pending[:_MAX_ADMISSION_RETRIES_PER_TICK]:
            try:
                self._admit(session)
            except Exception as exc:  # the heartbeat never fails on a retry
                logger.debug("admission retry failed: %s", type(exc).__name__)
            self._refresh_status_if_due()
            if not session.admitted:
                with self._lock:
                    if session.hermes_session_id in self._unadmitted:
                        self._unadmitted.move_to_end(session.hermes_session_id)
                break

    def _finish_admission(
        self, session: _Session, resp: Optional[Mapping[str, Any]], *, lost: bool = True
    ) -> list[tuple[str, str]]:
        """Publish the session's Agenomic id and hand back the cancels to decide now: those
        naming that id, and, once no admission is in flight, every one still unresolved
        (they then name no session and are refused)."""
        info = resp.get("session") if resp is not None else None
        with self._lock:
            self._admissions_in_flight -= 1
            if resp is not None:
                # Under the lock: a concurrent admission of the same session that fails
                # afterwards sees it admitted and never records it as unadmitted.
                session.admitted = True
            if resp is None and lost and session.active and not session.admitted:
                self._unadmitted[session.hermes_session_id] = None
            else:
                # Admitted (by this request or a concurrent one), refused for good (no
                # session was created), or failed after the session already ended (its
                # terminal callback ran while this request was in flight): no id will be
                # published for it, so it holds no cancel unresolved.
                self._unadmitted.pop(session.hermes_session_id, None)
            if isinstance(info, dict) and isinstance(info.get("id"), str):
                session.agenomic_id = cast(str, info["id"])
                self._agenomic_sessions[session.agenomic_id] = session.hermes_session_id
            ready: list[str] = []
            if session.agenomic_id:
                ready.append(session.agenomic_id)
            if not self._admissions_in_flight and not self._unadmitted:
                ready.extend(t for t in self._unresolved_cancels if t not in ready)
            return [
                (command_id, target)
                for target in ready
                for command_id in self._unresolved_cancels.pop(target, [])
            ]

    def _forget_unadmitted(self, sid: str) -> None:
        """A session ended without being admitted: no id will be published for it, so once
        nothing else may still name the unresolved cancels they are decided (refused)."""
        with self._lock:
            if sid not in self._unadmitted:
                return
            self._unadmitted.pop(sid, None)
            if self._admissions_in_flight or self._unadmitted:
                return
            pending = [
                (command_id, target)
                for target in list(self._unresolved_cancels)
                for command_id in self._unresolved_cancels.pop(target, [])
            ]
        for command_id, target in pending:
            self._cancel_session(command_id, target)

    def on_session_start(self, **kwargs: object) -> None:
        """Admit the session (idempotent server side) and emit ``session.started``.

        Example:
            >>> _demo_adapter().on_session_start(session_id="s1", platform="cli", model="demo-model") is None
            True
        """
        try:
            sid = _str(kwargs.get("session_id"))
            platform = _str(kwargs.get("platform"))
            self._ensure_started(platform)
            if not sid:
                return
            session = self._session(sid, platform, _str(kwargs.get("model")) or None)
            session.active = True
            self._admit(session)
            self._emit(
                "session.started",
                sid,
                span_id=sid,
                parent_span_id=session.parent,
                model={"model": session.model} if session.model else None,
                extra={"platform": platform, "subagent_id": session.subagent_id},
            )
        except Exception as exc:
            logger.debug("on_session_start failed: %s", type(exc).__name__)

    def _end(self, sid: str, final: bool, status: str, reason: str = "") -> str:
        """Report a session end: ``delivered``, ``transient`` (worth retrying) or
        ``refused`` (for good). Callers hold :meth:`_end_lock` for ``sid``."""
        body: dict[str, Any] = {"final": final, "status": status}
        if reason:
            # Sent to the gateway directly, not through the event pipeline: masked here.
            body["reason"] = mask_text(reason)[:200]
        try:
            self.client.end_session(sid, body)
        except HermesApiError as exc:
            logger.warning("session end not reported (%s)", exc.code)
            return "transient" if exc.retryable else "refused"
        return "delivered"

    def _end_lock(self, sid: str) -> threading.Lock:
        """Serializes the end reports of one session (striped): a retried older end and a
        newer one are never in flight together, so their delivery order is never reversed."""
        return self._end_locks[hash(sid) % len(self._end_locks)]

    def _end_turn(self, sid: str, status: str, reason: str) -> None:
        """A non-terminal turn end. Once delivered it supersedes a pending older end, which
        is then never replayed after it (its cancels wait for the next terminal end)."""
        with self._end_lock(sid):
            if self._end(sid, False, status, reason) == "delivered":
                with self._lock:
                    self._pending_ends.pop(sid, None)

    def _end_terminal(
        self,
        sid: str,
        final: bool,
        status: str,
        reason: str,
        subagent_id: Optional[str],
        how: str,
    ) -> None:
        """Report a terminal end, then settle the cancels waiting for it. The gateway
        applies a cancel only once the session is terminal in the control plane: while the
        report fails transiently it is retried every heartbeat and the cancels wait."""
        with self._end_lock(sid):
            outcome = self._end(sid, final, status, reason)
            with self._lock:
                if outcome != "transient":
                    self._pending_ends.pop(sid, None)  # superseded by this newer end
                else:
                    self._pending_ends[sid] = _PendingEnd(final, status, reason, subagent_id, how)
                    self._bound_pending_ends()
        if outcome != "transient":
            self._observe_terminal(sid, subagent_id, how, delivered=outcome == "delivered")

    def _bound_pending_ends(self) -> None:
        """Called under ``_lock``. Bounded, but never at the expense of a cancel: an end
        some cancel waits for is kept (those are bounded by the gateway's commands), the
        oldest other one is dropped."""
        while len(self._pending_ends) > _MAX_AUTH:
            victim = next(
                (
                    key
                    for key, end in self._pending_ends.items()
                    if key not in self._cancel_sessions
                    and not (end.subagent_id and end.subagent_id in self._cancel_subagents)
                ),
                None,
            )
            if victim is None:
                break
            del self._pending_ends[victim]

    def _retry_ends(self) -> None:
        with self._lock:
            pending = list(self._pending_ends.items())
        for sid, end in pending:
            with self._end_lock(sid):
                with self._lock:
                    if self._pending_ends.get(sid) is not end:
                        continue  # superseded by a newer end meanwhile: never replayed
                outcome = self._end(sid, end.final, end.status, end.reason)
                if outcome == "transient":
                    # The end stays pending (and its cancels waiting) for as long as it
                    # takes: acknowledging them now would claim an end the control plane
                    # has not recorded.
                    self._refresh_status_if_due()
                    continue
                with self._lock:
                    if self._pending_ends.get(sid) is end:
                        del self._pending_ends[sid]
            self._observe_terminal(sid, end.subagent_id, end.how, delivered=outcome == "delivered")
            self._refresh_status_if_due()

    def on_session_end(self, **kwargs: object) -> None:
        """Per TURN end (not final). ``interrupted`` applies a pending cancel.

        Example:
            >>> _demo_adapter().on_session_end(session_id="s1", interrupted=False) is None
            True
        """
        try:
            sid = _str(kwargs.get("session_id"))
            if not sid:
                return
            interrupted = bool(kwargs.get("interrupted"))
            status = (
                "interrupted"
                if interrupted
                else ("failed" if kwargs.get("failed") else "completed")
            )
            if interrupted and self._cancel_pending(sid):
                # The interrupt is the cancel Agenomic asked for: report the session as
                # cancelled, a terminal state the gateway records before applying the command.
                status = "cancelled"
            reason = _str(kwargs.get("turn_exit_reason") or kwargs.get("reason"))
            if not interrupted:
                self._end_turn(sid, status, reason)
            self._emit(
                "session.turn_ended",
                sid,
                turn_id=_str(kwargs.get("turn_id")) or None,
                status=status,
                reason=_str(kwargs.get("turn_exit_reason"))[:200] or None,
            )
            if interrupted:
                session = self._sessions.get(sid)
                if status == "cancelled":
                    # The cancel's interrupt ends the session: it is no longer advertised
                    # as active, retried for admission, nor a target for later cancels.
                    with self._lock:
                        if session is not None:
                            session.active = False
                    self._forget_unadmitted(sid)
                self._end_terminal(
                    sid,
                    False,
                    status,
                    reason,
                    session.subagent_id if session else None,
                    "on_session_end interrupted",
                )
        except Exception as exc:
            logger.debug("on_session_end failed: %s", type(exc).__name__)

    def on_session_finalize(self, **kwargs: object) -> None:
        """Final end of a session.

        Example:
            >>> _demo_adapter().on_session_finalize(session_id="s1", reason="exit") is None
            True
        """
        try:
            sid = _str(kwargs.get("session_id"))
            reason = _str(kwargs.get("reason"))
            self._emit("session.finalized", sid or None, reason=reason[:200] or None)
            if not sid:
                return
            with self._lock:
                session = self._sessions.get(sid)
                subagent_id = (
                    session.subagent_id
                    if session is not None
                    else self._children.get(sid, (None, None))[1]
                )
            # A pending cancel (of the session or of its subagent) ends here: report it as
            # cancelled, the terminal state the gateway checks before applying the command.
            status = "cancelled" if self._cancel_pending(sid) else "completed"
            with self._lock:
                if session is not None:
                    session.active = False
            self._forget_unadmitted(sid)
            self._end_terminal(sid, True, status, reason, subagent_id, "on_session_finalize")
        except Exception as exc:
            logger.debug("on_session_finalize failed: %s", type(exc).__name__)

    def on_session_reset(self, **kwargs: object) -> None:
        """``/new`` or reset: observed only.

        Example:
            >>> _demo_adapter().on_session_reset(session_id="s2", old_session_id="s1") is None
            True
        """
        sid = _str(kwargs.get("session_id"))
        self._emit(
            "session.reset",
            sid or None,
            reason=_str(kwargs.get("reason"))[:200] or None,
            extra={"old_session_id": _str(kwargs.get("old_session_id")) or None},
        )

    def subagent_start(self, **kwargs: object) -> None:
        """Fires BEFORE the child's ``on_session_start``: record the parent link.

        Example:
            >>> _demo_adapter().subagent_start(parent_session_id="s1", child_session_id="s1.1") is None
            True
        """
        try:
            child = _str(kwargs.get("child_session_id"))
            parent = _str(kwargs.get("parent_session_id"))
            subagent_id = _str(kwargs.get("child_subagent_id")) or None
            if child:
                reservation = getattr(self._invocation, "delegation", None)
                with self._lock:
                    self._children[child] = (parent, subagent_id)
                    if reservation is not None:
                        self._child_delegations[child] = reservation
            self._emit(
                "subagent.started",
                parent or None,
                span_id=child or None,
                parent_span_id=parent or None,
                turn_id=_str(kwargs.get("parent_turn_id")) or None,
                content={"goal": kwargs.get("child_goal")},
                extra={
                    "child_session_id": child,
                    "child_subagent_id": subagent_id,
                    "child_role": _str(kwargs.get("child_role")) or None,
                },
            )
        except Exception as exc:
            logger.debug("subagent_start failed: %s", type(exc).__name__)

    def subagent_stop(self, **kwargs: object) -> None:
        """Child finished: end its session and apply a pending cancel.

        Example:
            >>> _demo_adapter().subagent_stop(parent_session_id="s1", child_session_id="s1.1", child_status="completed") is None
            True
        """
        try:
            child = _str(kwargs.get("child_session_id"))
            parent = _str(kwargs.get("parent_session_id"))
            raw_status = _str(kwargs.get("child_status")).lower()
            status = (
                raw_status
                if raw_status in ("completed", "failed", "interrupted", "cancelled")
                else ("failed" if raw_status in ("error", "timeout") else "completed")
            )
            subagent_id = self._children.get(child, (None, None))[1] if child else None
            self._emit(
                "subagent.stopped",
                parent or None,
                span_id=child or None,
                parent_span_id=parent or None,
                status=status,
                latency_ms=kwargs.get("duration_ms")
                if isinstance(kwargs.get("duration_ms"), int)
                else None,
                content={"summary": kwargs.get("child_summary")},
                extra={"child_session_id": child, "child_subagent_id": subagent_id},
            )
            if child:
                with self._lock:
                    session = self._sessions.get(child)
                    if session is not None:
                        session.active = False
                self._forget_unadmitted(child)
                self._end_terminal(
                    child, True, status, "subagent_stop", subagent_id, "subagent_stop"
                )
        except Exception as exc:
            logger.debug("subagent_stop failed: %s", type(exc).__name__)

    # ------------------------------------------------------------------
    # model calls (observed only)
    # ------------------------------------------------------------------
    def pre_api_request(self, **kwargs: object) -> None:
        """Main loop model call started.

        Example:
            >>> _demo_adapter().pre_api_request(session_id="s1", api_request_id="r1", model="demo-model") is None
            True
        """
        sid = _str(kwargs.get("session_id"))
        self._ensure_started(_str(kwargs.get("platform")))
        self._emit(
            "model.call.started",
            sid or None,
            span_id=_str(kwargs.get("api_request_id")) or None,
            parent_span_id=_str(kwargs.get("turn_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            model={"provider": _str(kwargs.get("provider")), "model": _str(kwargs.get("model"))},
            content={"input": kwargs.get("request_messages")},
            extra={
                "base_url": _str(kwargs.get("base_url")),
                "message_count": kwargs.get("message_count"),
                "tool_count": kwargs.get("tool_count"),
                "approx_input_tokens": kwargs.get("approx_input_tokens"),
                "retry_count": kwargs.get("retry_count"),
            },
        )

    @staticmethod
    def _usage(raw: object) -> Optional[dict[str, Any]]:
        if raw is None:
            return None

        def get(key: str) -> object:
            return raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)

        inp = get("input_tokens") or get("prompt_tokens")
        out = get("output_tokens") or get("completion_tokens")
        if not isinstance(inp, int) and not isinstance(out, int):
            return {"known": False}
        return {
            "input_tokens": inp if isinstance(inp, int) else None,
            "output_tokens": out if isinstance(out, int) else None,
            "known": True,
        }

    def post_api_request(self, **kwargs: object) -> None:
        """Main loop model call completed.

        Example:
            >>> _demo_adapter().post_api_request(session_id="s1", api_request_id="r1", api_duration=0.4) is None
            True
        """
        duration = kwargs.get("api_duration")
        self._emit(
            "model.call.completed",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("api_request_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            model={"provider": _str(kwargs.get("provider")), "model": _str(kwargs.get("model"))},
            status="ok",
            latency_ms=int(duration * 1000) if isinstance(duration, (int, float)) else None,
            usage=self._usage(kwargs.get("usage")),
            content={"output": kwargs.get("assistant_message")},
            extra={
                "finish_reason": _str(kwargs.get("finish_reason")) or None,
                "tool_call_count": kwargs.get("assistant_tool_call_count"),
            },
        )

    def api_request_error(self, **kwargs: object) -> None:
        """Main loop model call failed.

        Example:
            >>> _demo_adapter().api_request_error(session_id="s1", api_request_id="r1", status_code=429) is None
            True
        """
        raw_error = kwargs.get("error")
        error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
        self._emit(
            "model.call.failed",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("api_request_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            model={"provider": _str(kwargs.get("provider")), "model": _str(kwargs.get("model"))},
            status="error",
            reason=_str(kwargs.get("reason"))[:200] or None,
            extra={
                "status_code": kwargs.get("status_code"),
                "retryable": kwargs.get("retryable"),
                "error_type": _str(error.get("type")) or None,
            },
        )

    def pre_auxiliary_call(self, **kwargs: object) -> None:
        """Auxiliary model call (titles, compression, ...): only controllable at the Model Gateway.

        Example:
            >>> _demo_adapter().pre_auxiliary_call(session_id="s1", aux_task="title") is None
            True
        """
        self._emit(
            "model.aux.started",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("api_request_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            model={"provider": _str(kwargs.get("provider")), "model": _str(kwargs.get("model"))},
            extra={"aux_task": _str(kwargs.get("aux_task")), "controlled": False},
        )

    def post_auxiliary_call(self, **kwargs: object) -> None:
        """Auxiliary model call completed.

        Example:
            >>> _demo_adapter().post_auxiliary_call(session_id="s1", aux_task="title", api_duration=0.1) is None
            True
        """
        duration = kwargs.get("api_duration")
        self._emit(
            "model.aux.completed",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("api_request_id")) or None,
            model={"provider": _str(kwargs.get("provider")), "model": _str(kwargs.get("model"))},
            status="error" if kwargs.get("error") else "ok",
            latency_ms=int(duration * 1000) if isinstance(duration, (int, float)) else None,
            usage=self._usage(kwargs.get("usage")),
            extra={"aux_task": _str(kwargs.get("aux_task")), "controlled": False},
        )

    def pre_approval_request(self, **kwargs: object) -> None:
        """Hermes approval requested. Observed only: the adapter never answers approvals.

        Example:
            >>> _demo_adapter().pre_approval_request(session_id="s1", tool_call_id="c1", command="rm -rf build") is None
            True
        """
        self._emit(
            "approval.requested",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("tool_call_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            content={"command": kwargs.get("command")},
            extra={
                "surface": _str(kwargs.get("surface")),
                "pattern_key": _str(kwargs.get("pattern_key")),
            },
        )

    def post_approval_response(self, **kwargs: object) -> None:
        """Hermes approval answered (by a human or Hermes smart approvals). Observed only.

        Example:
            >>> _demo_adapter().post_approval_response(session_id="s1", tool_call_id="c1", choice="once") is None
            True
        """
        self._emit(
            "approval.responded",
            _str(kwargs.get("session_id")) or None,
            span_id=_str(kwargs.get("tool_call_id")) or None,
            turn_id=_str(kwargs.get("turn_id")) or None,
            extra={
                "surface": _str(kwargs.get("surface")),
                "choice": _str(kwargs.get("choice")),
                "decided_by": _str(kwargs.get("decided_by")) or None,
            },
        )

    def agent_loop_stopped(self, **kwargs: object) -> None:
        """Hermes Messaging Gateway ``/stop``.

        Example:
            >>> _demo_adapter().agent_loop_stopped(reason="/stop", platform="telegram") is None
            True
        """
        self._emit(
            "agent.loop_stopped",
            None,
            reason=_str(kwargs.get("reason"))[:200] or None,
            extra={"platform": _str(kwargs.get("platform"))},
        )

    def llm_request(self, **kwargs: object) -> Optional[dict[str, object]]:
        """Add ``X-Agenomic-Hermes-Session`` on Model Gateway requests; nothing else changes.

        Example:
            >>> a = _demo_adapter()
            >>> out = a.llm_request(session_id="s1", base_url=a.config.endpoint + "/v1", request={"model": "m"})
            >>> out["request"]
            {'model': 'm', 'extra_headers': {'X-Agenomic-Hermes-Session': 's1'}}
        """
        try:
            request = kwargs.get("request")
            sid = _str(kwargs.get("session_id"))
            base_url = _str(kwargs.get("base_url"))
            if not isinstance(request, dict) or not sid:
                return None
            if not base_url.startswith(self.config.endpoint + "/"):
                return None
            headers = request.get("extra_headers")
            new_headers = dict(headers) if isinstance(headers, dict) else {}
            # Header names are case-insensitive: any other spelling of the session header
            # is replaced, never sent next to the authoritative one.
            variants = [
                k
                for k in new_headers
                if isinstance(k, str)
                and k.lower() == _SESSION_HEADER.lower()
                and k != _SESSION_HEADER
            ]
            if not variants and new_headers.get(_SESSION_HEADER) == sid:
                return None
            for key in variants:
                del new_headers[key]
            new_headers[_SESSION_HEADER] = sid
            new_request = dict(request)
            new_request["extra_headers"] = new_headers
            return {"request": new_request, "source": "agenomic", "reason": "session correlation"}
        except Exception as exc:
            logger.debug("llm_request middleware failed: %s", type(exc).__name__)
            return None

    # ------------------------------------------------------------------
    # authorization
    # ------------------------------------------------------------------
    def _protected_roots(self) -> list[str]:
        roots = [
            self.home / "skills",
            self.home / "plugins",
            self.home / "config.yaml",
            self.home / ".env",
            Path("/etc/hermes"),
        ]
        extra = self._profile.get("protected_paths")
        if isinstance(extra, list):
            roots += [Path(p).expanduser() for p in extra if isinstance(p, str) and p]
        return [os.path.realpath(r) for r in roots]

    def protected_targets(self, tool: str, args: Mapping[str, object]) -> list[str]:
        """Paths a write tool would touch inside protected Hermes locations.

        Example:
            >>> a = _demo_adapter()
            >>> config_file = str(a.home / "config.yaml")
            >>> a.protected_targets("write_file", {"path": config_file}) == [config_file]
            True
        """
        if tool not in _WRITE_TOOLS:
            return []
        candidates: list[str] = []
        path = args.get("path")
        if isinstance(path, str) and path:
            candidates.append(path)
        patch = args.get("patch")
        if isinstance(patch, str):
            for line in patch.splitlines():
                for marker in (
                    "*** Update File:",
                    "*** Add File:",
                    "*** Delete File:",
                    "*** Move File:",
                ):
                    if line.startswith(marker):
                        rest = line[len(marker) :].strip()
                        candidates += [p.strip() for p in rest.split("->") if p.strip()]
        roots = self._protected_roots()
        hits: list[str] = []
        for candidate in candidates:
            resolved = os.path.realpath(os.path.expanduser(candidate))
            for root in roots:
                if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
                    hits.append(candidate)
                    break
        return hits

    def _reserve_delegation(
        self, sid: str, args: Mapping[str, Any], tool_call_id: str
    ) -> tuple[Optional[str], Optional[list[Any]]]:
        """``(block, reservation)``. The reservation is not queued for a child yet."""
        action = _str(args.get("action")).lower()
        if action in _DELEGATE_CONTROL_ACTIONS:
            return None, None
        tasks = args.get("tasks")
        count = len(tasks) if isinstance(tasks, list) and tasks else 1
        status, resp = self.client.reserve_delegation(
            sid, {"count": count, "tool_call_id": tool_call_id}
        )
        decision = _str(resp.get("decision"))
        if decision in _DELEGATION_STATUS and status != _DELEGATION_STATUS[decision]:
            # allow and observe come with 200, deny with 403: a decision contradicting its
            # status is not trusted (enforce blocks on the invalid answer).
            raise HermesApiError(
                "invalid_response", "delegation answer contradicting its HTTP status", status
            )
        if decision == "observe":
            return None, None
        delegation_id = resp.get("delegation_id")
        if decision == "allow" and isinstance(delegation_id, str) and delegation_id:
            # An empty id would be dropped from the child's admission: never a reservation.
            return None, [delegation_id, count]
        if decision == "deny":
            codes = resp.get("reason_codes")
            explanation = _str(resp.get("explanation")) or ", ".join(
                str(c) for c in (codes if isinstance(codes, list) else [])
            )
            return f"Agenomic denied delegate_task: {explanation or 'delegation limit'}", None
        raise HermesApiError(
            "invalid_response", "delegation answer without a valid decision", status
        )

    def _claim_delegation(
        self,
        *,
        slot: tuple[str, str, str, str],
        claim: str,
        sid: str,
        tool: str,
        args: Mapping[str, Any],
        tool_call_id: str,
        local_hash: str,
        mode: LocalMode,
    ) -> tuple[Optional[str], Optional[_Provisional]]:
        """``(block, provisional)`` for this invocation of ``delegate_task``.

        An unclaimed provisional entry waiting in ``slot`` (a retry of the same action, or
        of the same approval) is claimed and reused; otherwise this invocation reserves its
        own, so two concurrent identical invocations never share one reservation.
        """
        with self._lock:
            waiting = [
                e
                for e in self._provisional_delegations.get(slot, ())
                if e.claimed_by in (None, claim)
            ]
            if waiting:
                # The one reserved for this tool call, else the oldest one waiting.
                entry = next((e for e in waiting if e.tool_call_id == tool_call_id), waiting[0])
                entry.claimed_by = claim
                return None, entry
        denied, reservation = self._reserve_delegation(sid, args, tool_call_id)
        if denied and mode == "shadow":
            # Shadow never changes execution: the refused reservation is recorded.
            self._emit_decision(
                sid,
                tool,
                tool_call_id,
                "deny",
                denied,
                local_hash,
                extra={"local": True, "shadow": True},
            )
        elif denied:
            self._emit_decision(sid, tool, tool_call_id, "deny", denied, local_hash)
            return denied, None
        if reservation is None:
            return None, None
        provisional = _Provisional(
            reservation, claimed_by=claim, slot=slot, tool_call_id=tool_call_id
        )
        with self._lock:
            # Published claimed: a retry of this action finds it once it is released.
            self._provisional_delegations.setdefault(slot, []).append(provisional)
        return None, provisional

    def _unlist_delegation(self, provisional: _Provisional) -> None:
        """Remove ``provisional`` from its slot (the lock is held)."""
        entries = self._provisional_delegations.get(provisional.slot)
        if entries is None:
            return
        for i, entry in enumerate(entries):
            if entry is provisional:
                del entries[i]
                break
        if not entries:
            del self._provisional_delegations[provisional.slot]

    def _rebind_delegation(self, provisional: Optional[_Provisional], approval_id: str) -> None:
        """The action of this reservation now waits for ``approval_id``: the reservation
        moves with it, so the retry resumed under that approval is the one reusing it."""
        if provisional is None or provisional.settled:
            return
        sid, tool, local_hash, _ = provisional.slot
        target = (sid, tool, local_hash, approval_id)
        with self._lock:
            self._unlist_delegation(provisional)
            provisional.slot = target
            if self._provisional_delegations.get(target):
                # That approval already has its reservation: this one is forgotten.
                provisional.settled = True
                return
            self._provisional_delegations[target] = [provisional]

    def _release_delegation(self, provisional: Optional[_Provisional]) -> None:
        """This invocation stops deciding: an unsettled reservation waits for a retry."""
        if provisional is None:
            return
        with self._lock:
            if provisional.settled:
                return
            entries = self._provisional_delegations.setdefault(provisional.slot, [])
            if not any(e is provisional for e in entries):
                entries.append(provisional)
            provisional.claimed_by = None

    def _settle_delegation(
        self, provisional: Optional[_Provisional], commit: bool
    ) -> Optional[list[Any]]:
        """Queue this invocation's reservation for the child once the action is allowed,
        or forget it; another invocation's reservation is never touched."""
        if provisional is None:
            return None
        with self._lock:
            provisional.settled = True
            key = provisional.slot
            self._unlist_delegation(provisional)
            if commit:
                self._delegations.setdefault(key[0], deque()).append(provisional.reservation)
                return provisional.reservation
            return None

    def _drop_delegation(self, auth: _Authorization) -> None:
        """Forget the unconsumed part of a reservation whose call did not run its children."""
        with self._lock:
            reservation, auth.delegation = auth.delegation, None
            queue = self._delegations.get(auth.session_id)
            if reservation is None or not queue:
                return
            for i, queued in enumerate(queue):
                if queued is reservation:
                    del queue[i]
                    break

    def _drop_pending(self, key: tuple[str, str, str], pending: Optional[_Pending]) -> None:
        """Forget the approval this invocation retried under, never one bound to another."""
        if pending is None:
            return
        with self._lock:
            entries = self._pending.get(key)
            if entries is None:
                return
            for i, entry in enumerate(entries):
                if entry is pending:
                    del entries[i]
                    break
            if not entries:
                del self._pending[key]

    def _forget_approval(self, key: tuple[str, str, str], pending: _Pending) -> None:
        """A rejected or expired approval: its identity and the reservation waiting for it
        are both forgotten."""
        self._drop_pending(key, pending)
        with self._lock:
            waiting = [
                e
                for e in self._provisional_delegations.get((*key, pending.approval_id), ())
                if e.claimed_by is None
            ]
        for provisional in waiting:
            self._settle_delegation(provisional, commit=False)

    def _unclaim(self, pending: _Pending, claim: str) -> None:
        with self._lock:
            if pending.claimed_by == claim:
                pending.claimed_by = None

    def _select_pending(
        self,
        *,
        key: tuple[str, str, str],
        claim: str,
        sid: str,
        tool: str,
        tool_call_id: str,
        local_hash: str,
    ) -> tuple[Optional[_Pending], Optional[str]]:
        """``(pending, block)``: the approval this invocation retries under, if any.

        Every unclaimed approval issued for this action is tried in issue order: the first
        one granted (or whose status cannot be read; the gateway then decides) is claimed
        and returned. A still pending one is released and the next one tried; a rejected or
        expired one is forgotten. Identities are never merged: the invocation resumes the
        identity of the one approval it claimed. With no approval at all the action is
        asked afresh; when none can be used the call is blocked, preferring the message of
        an approval still pending, then that of one refused, then "in use".
        """
        tried: set[int] = set()
        waiting: Optional[str] = None
        refused: Optional[str] = None
        in_use: Optional[_Pending] = None
        while True:
            candidate: Optional[_Pending] = None
            with self._lock:
                for entry in self._pending.get(key, ()):
                    if id(entry) in tried:
                        continue
                    if entry.claimed_by in (None, claim):
                        entry.claimed_by = claim
                        candidate = entry
                        break
                    if in_use is None:
                        in_use = entry
            if candidate is None:
                break
            tried.add(id(candidate))
            try:
                gate, keep = self._approval_gate(candidate)
            except BaseException:
                self._unclaim(candidate, claim)
                raise
            if gate is None:
                return candidate, None
            self._unclaim(candidate, claim)
            if keep:
                waiting = waiting or gate
            else:
                self._forget_approval(key, candidate)
                refused = refused or gate
        if waiting or refused:
            return None, waiting or refused
        if in_use is not None:
            # Another invocation is retrying under this approval right now; it is the one
            # the approval authorizes. This one is not resumed under the same identity.
            message = APPROVAL_IN_USE_MESSAGE.format(approval_id=in_use.approval_id)
            self._emit_decision(
                sid,
                tool,
                tool_call_id,
                "require_approval",
                message,
                local_hash,
                extra={"local": True, "approval_id": in_use.approval_id},
            )
            return None, message
        return None, None

    def _approval_gate(self, pending: _Pending) -> tuple[Optional[str], bool]:
        """Before a controlled retry: ``(block, keep_identity)``.

        ``block`` is ``None`` to proceed with the reused identity. A rejected or
        expired approval blocks and forgets the identity; a pending one keeps it.
        When the status cannot be read the gateway decides on the retry.
        """
        try:
            status = _str(self.client.approval(pending.approval_id).get("status"))
        except HermesApiError:
            return None, True
        if status == "pending":
            return (
                f"Agenomic approval {pending.approval_id} is still pending; the action was not "
                "executed. Retry the same call after approval."
            ), True
        if status in ("approved", "consumed"):
            return None, True
        return (
            f"Agenomic approval {pending.approval_id} was {status or 'not granted'}; "
            "the action was not executed."
        ), False

    def authorize(
        self,
        *,
        tool: str,
        args: dict[str, JsonValue],
        sid: str,
        tool_call_id: str,
        turn_id: str = "",
        api_request_id: str = "",
        local_hash: str,
    ) -> _Verdict:
        """Ask the gateway; return a block message or the authorization. Raises on transport errors.

        Example:
            >>> a = _demo_adapter({"decision": "deny", "effective_mode": "enforce", "explanation": "not allowed"})
            >>> a.authorize(tool="terminal", args={"command": "ls"}, sid="s1", tool_call_id="c1", local_hash="h").block
            'Agenomic denied terminal: not allowed (decision unknown)'
        """
        if mask_text(tool) != tool:
            # A credential-shaped tool name never leaves the process (discovery withholds it
            # too): handled as an authorization that could not be asked, i.e. blocked in
            # enforce, unchanged execution in shadow and observe.
            raise HermesApiError("credential_shaped_tool", "tool name not sent to the gateway", 0)
        session = self._session(sid)
        if not session.admitted:
            self._admit(session)
        mode = self.local_mode()
        unconfirmed = self._confirm_foreign_mutators(sid, tool, tool_call_id, local_hash, mode)
        if unconfirmed is not None:
            return _Verdict(block=unconfirmed)
        # The hello may have changed the effective state: observe never asks the gateway,
        # and the snapshot taken before it must not decide what happens next.
        mode = self.local_mode()
        if mode == "observe":
            return _Verdict()
        # One authorization in flight per call identity: two invocations racing through the
        # first gate would otherwise both obtain a permit and both execute.
        ident = (sid, tool, tool_call_id) if tool_call_id else None
        if ident is not None:
            with self._lock:
                busy = ident in self._authorizing
                if not busy:
                    self._authorizing.add(ident)
            if busy:
                if mode == "shadow":
                    return _Verdict()  # shadow never blocks; this invocation holds no permit
                return _Verdict(
                    block=f"Agenomic: another invocation of {tool} with the same call id is "
                    "being authorized; the action was not executed."
                )
        try:
            return self._authorize_once(
                tool=tool,
                args=args,
                sid=sid,
                tool_call_id=tool_call_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                local_hash=local_hash,
                mode=mode,
            )
        finally:
            if ident is not None:
                with self._lock:
                    self._authorizing.discard(ident)

    def _authorize_once(
        self,
        *,
        tool: str,
        args: dict[str, JsonValue],
        sid: str,
        tool_call_id: str,
        turn_id: str,
        api_request_id: str,
        local_hash: str,
        mode: LocalMode,
    ) -> _Verdict:
        key = (sid, tool, local_hash)
        claim = tool_call_id or f"hermes-{uuid.uuid4().hex}"
        with self._lock:
            previous = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
        pending: Optional[_Pending] = None
        if mode == "shadow":
            self._shadow_pending_approval(key, sid, tool, tool_call_id, local_hash)
        else:
            # An approval required while enforcing stays for a later enforce; in shadow the
            # call is asked afresh, so a pending or refused approval never blocks it.
            pending, block = self._select_pending(
                key=key,
                claim=claim,
                sid=sid,
                tool=tool,
                tool_call_id=tool_call_id,
                local_hash=local_hash,
            )
            if block is not None:
                return _Verdict(block=block)
        provisional: Optional[_Provisional] = None
        try:
            if tool == _DELEGATE_TOOL:
                # After the approval claim: the invocation retrying under an approval is the
                # one that reuses the reservation associated with that approval.
                denied, provisional = self._claim_delegation(
                    slot=(*key, pending.approval_id if pending is not None else ""),
                    claim=claim,
                    sid=sid,
                    tool=tool,
                    args=args,
                    tool_call_id=tool_call_id,
                    local_hash=local_hash,
                    mode=mode,
                )
                if denied is not None:
                    return _Verdict(block=denied)
            return self._authorize_claimed(
                tool=tool,
                args=args,
                sid=sid,
                tool_call_id=tool_call_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                local_hash=local_hash,
                key=key,
                pending=pending,
                previous=previous,
                provisional=provisional,
            )
        finally:
            self._release_delegation(provisional)
            if pending is not None:
                # Still waiting (pending approval, transport error, observe): the next
                # retry may claim it. An allowed or denied call already removed it.
                self._unclaim(pending, claim)

    def _authorize_claimed(
        self,
        *,
        tool: str,
        args: dict[str, JsonValue],
        sid: str,
        tool_call_id: str,
        turn_id: str,
        api_request_id: str,
        local_hash: str,
        key: tuple[str, str, str],
        pending: Optional[_Pending],
        previous: Optional[_Authorization],
        provisional: Optional[_Provisional],
    ) -> _Verdict:
        if pending is not None:
            # Granted (checked by ``_select_pending``): resume that approval's identity.
            logical_call_id, attempt = pending.logical_call_id, pending.attempt
        else:
            logical_call_id = tool_call_id or f"hermes-{uuid.uuid4().hex}"
            attempt = (
                previous.attempt + 1 if previous is not None and previous.state == "done" else 1
            )
        # The gateway decides on, hashes and later verifies a redacted copy: credential
        # values never leave the process. The binding to the arguments that execute stays
        # local (``local_hash``, over the original arguments, checked at every gate).
        sent_args = cast(dict[str, JsonValue], redact(args))
        body: dict[str, Any] = {
            "tool_call_id": logical_call_id,
            "tool": tool,
            "arguments": sent_args,
            "attempt": attempt,
            "schema_hash": self._schema_hash_for(tool),
            "turn_id": turn_id or None,
            "api_request_id": api_request_id or None,
        }
        seq = self._state_request()
        status, resp = self.client.authorize(sid, body)
        decision = _str(resp.get("decision"))
        effective_mode = _str(resp.get("effective_mode"))
        if decision not in (
            "allow",
            "deny",
            "require_approval",
            "observe",
        ) or effective_mode not in (
            "enforce",
            "shadow",
            "observe",
        ):
            raise HermesApiError(
                "invalid_response", "authorize answer without a valid decision", status
            )
        if status != _DECISION_STATUS[decision]:
            # The gateway answers allow and observe with 200, require_approval with 202 and
            # deny with 403: a decision contradicting its status is not trusted.
            raise HermesApiError(
                "invalid_response", "authorize answer contradicting its HTTP status", status
            )
        if (decision == "observe") != (effective_mode == "observe"):
            # The gateway answers observe exactly when its effective mode is observe: an
            # observe decision under a stricter mode (or the reverse) is not trusted to
            # downgrade enforcement.
            raise HermesApiError(
                "invalid_response", "authorize answer with an inconsistent mode", status
            )
        if decision == "observe":
            current = self._effective_state
            if _is_blocking_state(current):
                # A newer heartbeat already set a blocking state (enforce_blocked, paused,
                # quarantined, revoked): this answer is stale, it never downgrades the state
                # and the call does not run under it.
                self._settle_delegation(provisional, commit=False)
                return _Verdict(
                    block=f"Agenomic: the instance is {current}; the action was not executed."
                )
            if not self._set_state("observe", seq) and self.local_mode() == "enforce":
                # A state from a request sent after this one (a heartbeat selecting enforce)
                # is already applied: the observe answer is stale. Under a newer shadow the
                # call proceeds: shadow never changes execution.
                self._settle_delegation(provisional, commit=False)
                return _Verdict(
                    block="Agenomic: the mode changed while this call was decided; "
                    "the action was not executed."
                )
            self._settle_delegation(provisional, commit=False)
            return _Verdict()
        applied = True
        blocking = self._effective_state
        if _is_blocking_state(blocking):
            # A blocking state (enforce_blocked, paused, quarantined, revoked) is applied:
            # no answer, allow included, lets the call run under it. The newer request
            # still orders later answers.
            self._note_state_seq(seq)
            self._settle_delegation(provisional, commit=False)
            return _Verdict(
                block=f"Agenomic: the instance is {blocking}; the action was not executed."
            )
        if effective_mode in ("shadow", "enforce"):
            # Also when the mode is unchanged: an older heartbeat must not override it.
            applied = self._set_state(effective_mode, seq)
        if not applied and _is_blocking_state(self._effective_state):
            # A newer request applied a blocking state after the check above (it maps to
            # the enforce local mode, so the comparison below would not see it).
            blocked_now = self._effective_state
            self._settle_delegation(provisional, commit=False)
            return _Verdict(
                block=f"Agenomic: the instance is {blocked_now}; the action was not executed."
            )
        if not applied and self.local_mode() != effective_mode:
            # A request sent after this one already applied another mode: the verdict is
            # read in that mode, never in the stale one.
            current = self.local_mode()
            if current == "observe":
                self._settle_delegation(provisional, commit=False)
                return _Verdict()
            if current == "enforce":
                self._settle_delegation(provisional, commit=False)
                return _Verdict(
                    block="Agenomic: the mode changed while this call was decided; "
                    "the action was not executed."
                )
            effective_mode = current
        explanation = _str(resp.get("explanation"))[:300]
        decision_id = _str(resp.get("decision_id")) or None
        self._emit_decision(
            sid,
            tool,
            tool_call_id,
            decision,
            explanation,
            local_hash,
            extra={
                "effective_mode": effective_mode,
                "decision_id": decision_id,
                "record_id": resp.get("record_id"),
                "approval_id": resp.get("approval_id"),
                "counterfactual": resp.get("counterfactual"),
                "reason_codes": resp.get("reason_codes"),
            },
            action_id=_str(resp.get("action_id")) or None,
            attempt=attempt,
        )
        shadow = effective_mode == "shadow"
        if decision == "deny" and not shadow:
            self._drop_pending(key, pending)
            self._settle_delegation(provisional, commit=False)
            return _Verdict(
                block=f"Agenomic denied {tool}: {explanation or 'policy'} (decision {decision_id or 'unknown'})"
            )
        if decision == "require_approval" and not shadow:
            approval_id = _str(resp.get("approval_id"))
            if not approval_id:
                raise HermesApiError(
                    "invalid_response", "require_approval without approval_id", status
                )
            issued = _Pending(
                _str(resp.get("logical_call_id")) or logical_call_id,
                int(cast(Any, resp.get("attempt")) or attempt),
                approval_id,
            )
            with self._lock:
                entries = self._pending.setdefault(key, [])
                # Every concurrently issued approval keeps its own identity; the entry this
                # invocation retried under is replaced in place, never another one.
                if pending is not None and any(e is pending for e in entries):
                    entries[next(i for i, e in enumerate(entries) if e is pending)] = issued
                elif not any(e.approval_id == approval_id for e in entries):
                    entries.append(issued)
            # A reservation of this action waits for this approval's retry.
            self._rebind_delegation(provisional, approval_id)
            return _Verdict(block=APPROVAL_MESSAGE.format(approval_id=approval_id))
        permit = resp.get("permit")
        record_id = _str(resp.get("record_id")) or None
        if not shadow and (record_id is None or not isinstance(permit, dict)):
            raise HermesApiError("invalid_response", "allow without record_id and permit", status)
        if (
            not shadow
            and tool == _DELEGATE_TOOL
            and provisional is None
            and _str(args.get("action")).lower() not in _DELEGATE_CONTROL_ACTIONS
        ):
            # The reservation was decided under observe (none made) and enforce answered
            # this authorization: no reservation covers the children, so the call is not
            # executed (the model's retry reserves under enforce).
            self._drop_pending(key, pending)
            return _Verdict(
                block="Agenomic: the mode changed while delegate_task was decided; "
                "the action was not executed."
            )
        auth = _Authorization(
            tool_call_id=tool_call_id or logical_call_id,
            session_id=sid,
            tool=tool,
            logical_call_id=_str(resp.get("logical_call_id")) or logical_call_id,
            attempt=int(cast(Any, resp.get("attempt")) or attempt),
            local_hash=local_hash,
            # Reported with the permit: the copy the gateway hashed, never the original.
            arguments=sent_args,
            effective_mode=effective_mode,
            record_id=record_id,
            permit=permit if isinstance(permit, dict) else None,
            server_hash=_str(resp.get("arguments_hash")) or None,
            decision_id=decision_id,
        )
        # This invocation consumed the approval: the next identical call needs a new one.
        self._drop_pending(key, pending)
        if not shadow:
            local_block = self._local_checks(tool, args)
            if local_block is not None:
                self._emit_decision(
                    sid, tool, tool_call_id, "deny", local_block, local_hash, extra={"local": True}
                )
                self._settle_delegation(provisional, commit=False)
                return _Verdict(block=local_block)
        else:
            local_block = self._local_checks(tool, args)
            if local_block is not None:
                self._emit_decision(
                    sid,
                    tool,
                    tool_call_id,
                    "deny",
                    local_block,
                    local_hash,
                    extra={"local": True, "shadow": True},
                )
        auth.delegation = self._settle_delegation(provisional, commit=True)
        with self._lock:
            auth_key = (auth.session_id, auth.tool, auth.tool_call_id)
            self._auth[auth_key] = auth
            self._auth.move_to_end(auth_key)
            while len(self._auth) > _MAX_AUTH:
                self._auth.popitem(last=False)
        return _Verdict(authorization=auth)

    def _confirm_foreign_mutators(
        self, sid: str, tool: str, tool_call_id: str, local_hash: str, mode: LocalMode
    ) -> Optional[str]:
        """Re-send hello before authorizing when the argument mutators changed since the server
        last confirmed them; a block message (enforce) when that hello is not delivered."""
        if self.foreign_mutators() == self._foreign or self._hello():
            return None
        # The hello may have overlapped a heartbeat that changed the mode: the current one
        # decides whether the unconfirmed callbacks block, not the snapshot taken before.
        mode = self.local_mode()
        reason = self._mutators_message(tool)
        self._emit_decision(
            sid,
            tool,
            tool_call_id,
            "deny",
            reason,
            local_hash,
            extra={"local": True, "foreign_mutators_unconfirmed": True, "local_mode": mode},
        )
        return None if mode in ("observe", "shadow") else reason

    @staticmethod
    def _mutators_message(tool: str) -> str:
        return (
            f"Agenomic: Hermes callbacks that can change the arguments of {tool} changed and "
            "the gateway has not confirmed them; the action was not executed."
        )

    def _local_finding(self, tool: str, args: Mapping[str, Any]) -> Optional[tuple[str, str]]:
        """The message and reason code of the first local check the call fails."""
        if not self.hermes_compatible:
            return (
                f"Agenomic: Hermes {self._identity.get('version')} is not in the adapter "
                "compatibility table; protected actions are blocked in enforce",
                HERMES_INCOMPATIBLE_REASON,
            )
        hits = self.protected_targets(tool, args)
        if hits:
            return (
                f"Agenomic denied {tool}: writes to protected Hermes paths need a reviewed "
                "proposal (local check)",
                PROTECTED_PATH_REASON,
            )
        return None

    def _local_checks(self, tool: str, args: Mapping[str, Any]) -> Optional[str]:
        """Defence in depth applied in enforce after the gateway allowed."""
        finding = self._local_finding(tool, args)
        return finding[0] if finding is not None else None

    def _first_gate(
        self,
        store: OrderedDict[tuple[str, str, str], str],
        key: tuple[str, str, str],
        gate: Literal["pre", "execution"],
    ) -> bool:
        """Whether ``gate`` records this invocation: the other gate of the same invocation
        consumes the entry the first one left; the same gate seeing the call id again is a
        later invocation reusing it, which records on its own."""
        if not key[2]:
            return True
        with self._lock:
            first = store.get(key)
            if first is not None and first != gate:
                del store[key]
                return False
            store[key] = gate
            store.move_to_end(key)
            while len(store) > _MAX_AUTH:
                store.popitem(last=False)
        return True

    def _shadow_pending_approval(
        self,
        key: tuple[str, str, str],
        sid: str,
        tool: str,
        tool_call_id: str,
        local_hash: str,
    ) -> None:
        """Shadow never blocks on an approval required in enforce for the same action, but
        records it: a local counterfactual ``require_approval`` naming the approval still
        held. The entry is only read, never claimed or dropped, so it stays for a later
        enforce retry."""
        with self._lock:
            entries = self._pending.get(key)
            approval_id = entries[0].approval_id if entries else None
        if approval_id is None:
            return
        self._emit_decision(
            sid,
            tool,
            tool_call_id,
            "require_approval",
            APPROVAL_MESSAGE.format(approval_id=approval_id),
            local_hash,
            extra={
                "local": True,
                "local_mode": "shadow",
                "shadow": True,
                "reason_codes": [APPROVAL_PENDING_REASON],
                "counterfactual": {
                    "outcome": "require_approval",
                    "reason_codes": [APPROVAL_PENDING_REASON],
                },
                "approval_id": approval_id,
            },
        )

    def _outstanding_approval(
        self, sid: str, tool: str, args: Mapping[str, object]
    ) -> Optional[str]:
        """The id of the first approval held for this action, read without claiming it."""
        if not self._pending:
            return None
        try:
            local_hash = arguments_hash(args)
        except CanonicalError:
            return None
        with self._lock:
            entries = self._pending.get((sid, tool, local_hash))
            return entries[0].approval_id if entries else None

    def _observe_local_checks(
        self,
        sid: str,
        tool: str,
        tool_call_id: str,
        args: Mapping[str, object],
        gate: Literal["pre", "execution"],
    ) -> None:
        """Observe never blocks and never asks the gateway, but the local checks enforce
        would apply (Hermes compatibility, protected paths, argument mutators the gateway
        has not confirmed) are recorded as local ``tool.call.decision`` events with the
        counterfactual deny, once per invocation.

        An approval required in enforce for the same action (session, tool, arguments
        hash) and still held locally is recorded as a counterfactual ``require_approval``
        naming it: enforce would have retried under it rather than run. The entry is only
        read, never claimed, released or dropped, and its status is not asked, so it stays
        for a later enforce retry. Delegation reservations are gateway state: observe
        neither reserves nor records one, since a refusal is only known by asking."""
        try:
            findings: list[tuple[str, str, str, dict[str, object]]] = []
            if self.foreign_mutators() != self._foreign:
                findings.append(
                    (
                        self._mutators_message(tool),
                        FOREIGN_MUTATORS_REASON,
                        "deny",
                        {"foreign_mutators_unconfirmed": True},
                    )
                )
            local = self._local_finding(tool, args)
            if local is not None:
                findings.append((local[0], local[1], "deny", {}))
            approval_id = self._outstanding_approval(sid, tool, args)
            if approval_id is not None:
                findings.append(
                    (
                        APPROVAL_MESSAGE.format(approval_id=approval_id),
                        APPROVAL_PENDING_REASON,
                        "require_approval",
                        {"approval_id": approval_id},
                    )
                )
            if not findings or not self._first_gate(
                self._observed_local, (sid, tool, tool_call_id), gate
            ):
                return
            input_hash = content_hash(args)
            for message, code, outcome, details in findings:
                extra: dict[str, object] = {
                    "local": True,
                    "local_mode": "observe",
                    "reason_codes": [code],
                    "counterfactual": {"outcome": outcome, "reason_codes": [code]},
                    **details,
                }
                self._emit_decision(
                    sid, tool, tool_call_id, outcome, message, input_hash, extra=extra
                )
        except Exception as exc:
            # Recording only: observe never changes execution.
            logger.warning("observe local checks failed: %s", type(exc).__name__)

    def _emit_decision(
        self,
        sid: str,
        tool: str,
        tool_call_id: str,
        decision: str,
        reason: str,
        local_hash: str,
        *,
        extra: Optional[Mapping[str, object]] = None,
        action_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> None:
        self._emit(
            "tool.call.decision",
            sid,
            span_id=tool_call_id or None,
            tool={"name": tool},
            decision=decision,
            reason=reason[:300] or None,
            input_hash=local_hash,
            action_id=action_id,
            attempt_id=attempt,
            extra=extra,
        )

    def _unavailable(
        self, sid: str, tool: str, tool_call_id: str, exc: BaseException
    ) -> Optional[str]:
        code = getattr(exc, "code", type(exc).__name__)
        mode = self.local_mode()
        self._emit(
            "authorization.unavailable",
            sid or None,
            span_id=tool_call_id or None,
            tool={"name": tool},
            reason=str(code)[:100],
            extra={"local_mode": mode},
        )
        if mode in ("shadow", "observe"):
            # Shadow and observe never change execution (observe can be reached after the
            # authorization started, e.g. through the hello confirming argument mutators).
            return None
        return f"Agenomic authorization unavailable ({code}); the action was not executed."

    def _arguments_not_canonical(
        self,
        sid: str,
        tool: str,
        tool_call_id: str,
        args: Mapping[str, object],
        gate: Literal["pre", "execution"],
    ) -> Optional[str]:
        """Arguments without a canonical form (NaN, a set, an object another plugin put
        there) cannot be authorized: blocked in enforce; in shadow and observe the call
        proceeds and a local ``tool.call.decision`` records the counterfactual deny.

        Recorded once per invocation, whichever gate sees it first: the other gate of the
        same invocation consumes the entry; the same gate seeing the call id again is a
        later invocation reusing it, which is recorded on its own."""
        mode = self.local_mode()
        if self._first_gate(self._not_canonical, (sid, tool, tool_call_id), gate):
            extra: dict[str, object] = {
                "local": True,
                "local_mode": mode,
                "reason_codes": [NOT_CANONICAL_REASON],
            }
            if mode != "enforce":
                extra["counterfactual"] = {
                    "outcome": "deny",
                    "reason_codes": [NOT_CANONICAL_REASON],
                }
            self._emit_decision(
                sid,
                tool,
                tool_call_id,
                "deny",
                NOT_CANONICAL_REASON,
                content_hash(args),
                extra=extra,
            )
        if mode != "enforce":
            return None
        return f"Agenomic: arguments of {tool} have no canonical form; the action was not executed."

    def _blocked_session(
        self, sid: str, tool: str, tool_call_id: str, args: Mapping[str, object]
    ) -> Optional[str]:
        """A locally applied pause, quarantine or revoke, or a pending cancel of this
        session or of its subagent, blocks in every mode, an authorization cached by the
        other gate included (the command is asynchronous: a tool call racing with it must
        not run). The block is recorded as a local ``tool.call.decision``."""
        blocker = self._local_blocker(sid)
        if blocker is None:
            return None
        self._emit_local_block(sid, tool, tool_call_id, args, blocker)
        return blocker[0]

    def _local_blocker(self, sid: str) -> Optional[tuple[str, str]]:
        """``(message, reason code)`` of a local pause, quarantine or revoke, or of a pending
        cancel of this session or of its subagent; ``None`` when nothing blocks. Both are
        written under ``_lock``: a caller holding it sees a consistent answer."""
        status = self._local_status
        if status in _BLOCKING_STATUS:
            return (
                f"Agenomic: this instance is {status}; the action was not executed.",
                INSTANCE_STOPPED_REASON,
            )
        kind = self._cancel_kind(sid)
        if kind is None:
            return None
        return (
            f"Agenomic cancelled this {kind}; the action was not executed.",
            CANCEL_PENDING_REASON,
        )

    def _emit_local_block(
        self,
        sid: str,
        tool: str,
        tool_call_id: str,
        args: Mapping[str, object],
        blocker: tuple[str, str],
    ) -> None:
        message, reason = blocker
        self._emit_decision(
            sid,
            tool,
            tool_call_id,
            "deny",
            message,
            content_hash(args),
            extra={
                "local": True,
                "local_mode": self.local_mode(),
                "reason_codes": [reason],
            },
        )

    def pre_tool_call(self, **kwargs: object) -> Optional[dict[str, str]]:
        """Authorization gate. ``None`` lets Hermes proceed; a block directive vetoes the call.

        Example:
            >>> a = _demo_adapter({"decision": "deny", "effective_mode": "enforce", "explanation": "not allowed"})
            >>> a.pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", tool_call_id="c1")
            {'action': 'block', 'message': 'Agenomic denied terminal: not allowed (decision unknown)'}
        """
        tool = _str(kwargs.get("tool_name"))
        sid = _str(kwargs.get("session_id"))
        tool_call_id = _str(kwargs.get("tool_call_id"))
        try:
            self._ensure_started()
            raw_args = kwargs.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            cancelled = self._blocked_session(sid, tool, tool_call_id, args)
            if cancelled:
                return _block(cancelled)
            if self.local_mode() == "observe":
                self._observe_local_checks(sid, tool, tool_call_id, args, "pre")
            try:
                local_hash = arguments_hash(args)
            except CanonicalError:
                message = self._arguments_not_canonical(sid, tool, tool_call_id, args, "pre")
                return _block(message) if message else None
            self._emit(
                "tool.call.requested",
                sid or None,
                span_id=tool_call_id or None,
                parent_span_id=_str(kwargs.get("api_request_id")) or None,
                turn_id=_str(kwargs.get("turn_id")) or None,
                tool={"name": tool},
                input_hash=local_hash,
                content={"input": args},
            )
            # In the agent loop order the middleware's admission is already done: a local
            # command or a mode change applied since the first check is caught here, read
            # atomically with the mode this gate then decides under.
            with self._lock, self._state_lock:
                local_block = self._local_blocker(sid)
                mode_now = self.local_mode()
                observe_now = mode_now == "observe"
                # local_mode() folds enforce_blocked into enforce: the raw state is read too,
                # so a permit cached before a blocking heartbeat is never reused.
                raw_state = self._effective_state
            if local_block is not None:
                self._emit_local_block(sid, tool, tool_call_id, args, local_block)
                return _block(local_block[0])
            if _is_blocking_state(raw_state):
                return _block(
                    f"Agenomic: the instance is {raw_state}; the action was not executed."
                )
            admitted = getattr(self._invocation, "admitted", None)
            if admitted is not None and admitted[0] == (sid, tool, tool_call_id):
                # The middleware already admitted this very call (agent loop order). An
                # authorization made here would belong to no execution plan: never
                # reported, left reusable. A stricter mode since then blocks instead.
                if admitted[1] in ("observe", "shadow") and mode_now == "enforce":
                    return _block(
                        f"Agenomic: enforce became active after {tool} was decided in "
                        f"{admitted[1]}; the action was not executed."
                    )
                if admitted[1] == "observe" and mode_now == "shadow":
                    return None  # shadow never changes execution
                if not admitted[2] and mode_now != "enforce":
                    # Admitted without an authorization (shadow fail-open): none is made
                    # here either, since it would be owned by no plan; shadow never blocks.
                    return None
            if observe_now:
                if (
                    admitted is not None
                    and admitted[0] == (sid, tool, tool_call_id)
                    and admitted[2]
                ):
                    # The middleware runs this call under an authorization: arguments
                    # changed since are recorded and detach it (observe never blocks).
                    with self._lock:
                        running = self._auth.get((sid, tool, tool_call_id))
                    if (
                        running is not None
                        and running.state == "executing"
                        and running.local_hash != local_hash
                    ):
                        self._mismatch(running, local_hash, "pre_tool_call")
                return None
            with self._lock:
                existing = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
            if existing is not None and self._retire_stale(existing):
                existing = None
            if (
                existing is not None
                and existing.state == "executing"
                and existing.effective_mode == "shadow"
                and self.local_mode() == "enforce"
            ):
                # Agent loop order: the middleware ran under a shadow decision and enforce
                # became active before this gate; no enforce permit covers the call.
                return _block(
                    f"Agenomic: enforce became active after {tool} was decided in shadow; "
                    "the action was not executed."
                )
            if existing is not None and existing.state in ("authorized", "executing"):
                if existing.local_hash == local_hash:
                    return None
                return self._mismatch(existing, local_hash, "pre_tool_call")
            try:
                verdict = self.authorize(
                    tool=tool,
                    args=args,
                    sid=sid,
                    tool_call_id=tool_call_id,
                    turn_id=_str(kwargs.get("turn_id")),
                    api_request_id=_str(kwargs.get("api_request_id")),
                    local_hash=local_hash,
                )
            except HermesApiError as exc:
                message = self._unavailable(sid, tool, tool_call_id, exc)
                return _block(message) if message else None
            return _block(verdict.block) if verdict.block else None
        except Exception as exc:
            logger.warning("pre_tool_call failed: %s", type(exc).__name__)
            message = self._unavailable(sid, tool, tool_call_id, exc)
            if message and self.local_mode() != "observe":
                return _block(message)
            # Failing open (observe, shadow) goes through the same locked recheck as any
            # admission without an authorization: a command or enforce applied since blocks.
            raw_args = kwargs.get("args")
            fallback_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            try:
                plan = self._admit_without_authorization(sid, tool, tool_call_id, fallback_args, {})
            except Exception:
                return _block(NO_AUTH_MESSAGE)
            return None if plan.proceed else _block(plan.error or NO_AUTH_MESSAGE)

    def _mismatch(
        self, auth: _Authorization, local_hash: str, where: str
    ) -> Optional[dict[str, str]]:
        self._emit(
            "authorization.argument_mismatch",
            auth.session_id or None,
            span_id=auth.tool_call_id or None,
            tool={"name": auth.tool},
            input_hash=local_hash,
            reason=f"arguments changed after authorization ({where})",
            extra={"authorized_hash": auth.local_hash, "server_hash": auth.server_hash},
        )
        if auth.effective_mode == "shadow" or self.local_mode() in ("shadow", "observe"):
            # Shadow and observe never change execution, also when they became active
            # after enforce authorized the call: the mismatch is recorded, and the
            # authorization is detached so no permit-backed report describes a call
            # that ran with other arguments.
            with self._lock:
                auth.detached = True
            return None
        return _block(
            f"Agenomic: arguments of {auth.tool} changed after authorization; the action was not executed."
        )

    def _retire_stale(self, auth: _Authorization) -> bool:
        """Retire ``auth`` when it was decided under another mode than the current one and
        has not started executing (the state changed between the gates): the call is then
        authorized again under the current mode. Returns whether it was retired."""
        with self._lock:
            if auth.state != "authorized":
                return False
            # A blocking raw state (enforce_blocked collapses to the enforce local mode)
            # also retires it: the gateway decides again under that state.
            blocked = _is_blocking_state(self._effective_state)
            if not blocked and auth.effective_mode == self.local_mode():
                return False
            auth.state = "done"
        self._drop_delegation(auth)
        return True

    # ------------------------------------------------------------------
    # execution and reporting
    # ------------------------------------------------------------------
    def _observe_cached(self, meta: dict[str, Any]) -> Optional[_Authorization]:
        """Observe after ``pre_tool_call`` authorized this call in enforce or shadow (the
        state changed between the gates): the authorization is never left reusable. Same
        arguments: it is this execution's, retired and reported after it like any other,
        and its children take its reservation. Other arguments: it is retired unused."""
        tool_call_id = _str(meta.get("tool_call_id"))
        if not tool_call_id:
            return None
        key = (_str(meta.get("sid")), _str(meta.get("tool")), tool_call_id)
        try:
            local_hash: Optional[str] = arguments_hash(meta.get("args") or {})
        except CanonicalError:
            local_hash = None
        with self._lock:
            auth = self._auth.get(key)
            if auth is None or auth.state != "authorized":
                return None
            if auth.local_hash == local_hash:
                auth.state = "executing"
                meta["local_hash"] = local_hash
                return auth
            auth.state = "done"
        self._drop_delegation(auth)
        return None

    def _admit_without_authorization(
        self,
        sid: str,
        tool: str,
        tool_call_id: str,
        args: Mapping[str, object],
        meta: dict[str, Any],
        *,
        observe: bool = False,
    ) -> _ExecutionPlan:
        """The final admission of a call that proceeds without an authorization (observe,
        a shadow fail-open). In the agent loop order this middleware is the last gate, so
        a local command, a blocking state or enforce applied since the mode was read is
        caught here, atomically, as the admission of an authorized call does."""
        with self._lock, self._state_lock:
            local_block = self._local_blocker(sid)
            raw_state = self._effective_state
            mode = self.local_mode()
        if local_block is not None:
            self._emit_local_block(sid, tool, tool_call_id, args, local_block)
            return _ExecutionPlan(False, error=local_block[0], meta=meta)
        if _is_blocking_state(raw_state):
            return _ExecutionPlan(
                False,
                error=f"Agenomic: the instance is {raw_state}; the action was not executed.",
                meta=meta,
            )
        if mode == "enforce":
            return _ExecutionPlan(
                False,
                error="Agenomic: the mode changed while this call was decided; "
                "the action was not executed.",
                meta=meta,
            )
        return _ExecutionPlan(True, observe=observe, meta=meta)

    def _execution_gate(self, kwargs: Mapping[str, Any]) -> _ExecutionPlan:
        tool = _str(kwargs.get("tool_name"))
        sid = _str(kwargs.get("session_id"))
        tool_call_id = _str(kwargs.get("tool_call_id"))
        raw_args = kwargs.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        self._ensure_started()
        meta = {"tool": tool, "sid": sid, "tool_call_id": tool_call_id, "args": args}
        cancelled = self._blocked_session(sid, tool, tool_call_id, args)
        if cancelled:
            return _ExecutionPlan(False, error=cancelled, meta=meta)
        if self.local_mode() == "observe":
            self._observe_local_checks(sid, tool, tool_call_id, args, "execution")
            observed = self._observe_cached(meta)
            # Same atomic recheck as the admission below: a local command, or enforce (or a
            # blocking state), applied while the observe checks ran stops the call.
            local_block: Optional[tuple[str, str]] = None
            changed: Optional[str] = None
            with self._lock, self._state_lock:
                local_block = self._local_blocker(sid)
                if local_block is None and self.local_mode() == "enforce":
                    changed = self._effective_state or "enforce"
                if (local_block or changed) and observed is not None:
                    observed.state = "done"
            if observed is not None and (local_block or changed):
                self._drop_delegation(observed)
            if local_block is not None:
                self._emit_local_block(sid, tool, tool_call_id, args, local_block)
                return _ExecutionPlan(False, error=local_block[0], meta=meta)
            if changed is not None:
                if _is_blocking_state(changed):
                    error = f"Agenomic: the instance is {changed}; the action was not executed."
                else:
                    error = (
                        "Agenomic: the mode changed while this call was decided; "
                        "the action was not executed."
                    )
                return _ExecutionPlan(False, error=error, meta=meta)
            return _ExecutionPlan(True, observe=True, auth=observed, meta=meta)
        try:
            local_hash = arguments_hash(args)
        except CanonicalError:
            message = self._arguments_not_canonical(sid, tool, tool_call_id, args, "execution")
            if message:
                return _ExecutionPlan(False, error=message, meta=meta)
            return self._admit_without_authorization(sid, tool, tool_call_id, args, meta)
        meta["local_hash"] = local_hash
        with self._lock:
            auth = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
            if auth is not None and auth.state == "done":
                auth = None
        if auth is not None and self._retire_stale(auth):
            auth = None  # decided under another mode: asked again under the current one
        if auth is None:
            # Agent loop order: the middleware runs before pre_tool_call, so it asks first.
            try:
                verdict = self.authorize(
                    tool=tool,
                    args=args,
                    sid=sid,
                    tool_call_id=tool_call_id,
                    turn_id=_str(kwargs.get("turn_id")),
                    api_request_id=_str(kwargs.get("api_request_id")),
                    local_hash=local_hash,
                )
            except HermesApiError as exc:
                message = self._unavailable(sid, tool, tool_call_id, exc)
                if message:
                    return _ExecutionPlan(False, error=message, meta=meta)
                return self._admit_without_authorization(sid, tool, tool_call_id, args, meta)
            if verdict.block:
                return _ExecutionPlan(False, error=verdict.block, meta=meta)
            if self.local_mode() == "observe":
                return self._admit_without_authorization(
                    sid, tool, tool_call_id, args, meta, observe=True
                )
            auth = verdict.authorization
            if auth is None:
                if self.local_mode() == "shadow":
                    return self._admit_without_authorization(sid, tool, tool_call_id, args, meta)
                return _ExecutionPlan(False, error=NO_AUTH_MESSAGE, meta=meta)
        elif auth.local_hash != local_hash:
            blocked = self._mismatch(auth, local_hash, "tool_execution")
            if blocked is not None:
                return _ExecutionPlan(False, error=blocked["message"], meta=meta)
        blocked_by: Optional[str] = None
        local_block = None
        # Admission is atomic with server state updates (lock order: _lock, then
        # _state_lock; _set_state never takes _lock): a blocking state applied after the
        # earlier checks is seen here, before the call is marked executing.
        rerun: Optional[_ExecutionPlan] = None
        with self._lock, self._state_lock:
            if auth.state == "executing":
                # A second chain run for the same tool_call_id while the first is executing:
                # a shadow decision is reused only if nothing applied since forbids it.
                if auth.effective_mode != "shadow":
                    return _ExecutionPlan(False, error=NO_AUTH_MESSAGE, meta=meta)
                local_block = self._local_blocker(sid)
                raw_state = self._effective_state
                if local_block is None:
                    if _is_blocking_state(raw_state):
                        rerun = _ExecutionPlan(
                            False,
                            error=f"Agenomic: the instance is {raw_state}; "
                            "the action was not executed.",
                            meta=meta,
                        )
                    elif self.local_mode() == "enforce":
                        rerun = _ExecutionPlan(
                            False,
                            error="Agenomic: the mode changed while this call was decided; "
                            "the action was not executed.",
                            meta=meta,
                        )
                    else:
                        rerun = _ExecutionPlan(True, meta=meta)
                else:
                    rerun = _ExecutionPlan(False, error=local_block[0], meta=meta)
            else:
                # A local pause, quarantine, revoke or cancel applied after the first check
                # (while the authorization was in flight) is seen here: written under _lock.
                local_block = self._local_blocker(sid)
                if local_block is not None:
                    auth.state = "done"
                elif _is_blocking_state(self._effective_state):
                    blocked_by = self._effective_state
                    auth.state = "done"
                elif auth.effective_mode != "enforce" and self.local_mode() == "enforce":
                    # An observe or shadow decision is never executed once enforcement
                    # applies.
                    blocked_by = "enforce"
                    auth.state = "done"
                else:
                    auth.state = "executing"
        if rerun is not None:
            # The shadow rerun shares the first chain's authorization: nothing to drop.
            if local_block is not None:
                self._emit_local_block(sid, tool, tool_call_id, args, local_block)
            return rerun
        if local_block is not None:
            self._drop_delegation(auth)
            self._emit_local_block(sid, tool, tool_call_id, args, local_block)
            return _ExecutionPlan(False, error=local_block[0], meta=meta)
        if blocked_by == "enforce":
            self._drop_delegation(auth)
            return _ExecutionPlan(
                False,
                error="Agenomic: the mode changed while this call was decided; "
                "the action was not executed.",
                meta=meta,
            )
        if blocked_by is not None:
            self._drop_delegation(auth)
            return _ExecutionPlan(
                False,
                error=f"Agenomic: the instance is {blocked_by}; the action was not executed.",
                meta=meta,
            )
        self._emit(
            "tool.call.started",
            sid or None,
            span_id=tool_call_id or None,
            tool={"name": tool},
            input_hash=local_hash,
            action_id=auth.logical_call_id,
            attempt_id=auth.attempt,
        )
        return _ExecutionPlan(True, auth=auth, meta=meta)

    def tool_execution(self, **kwargs: object) -> object:
        """Execution middleware. Never raises before ``next_call``; reports after it.

        Example:
            >>> a = _demo_adapter()  # the gateway answers observe: the call runs
            >>> a.tool_execution(tool_name="terminal", args={}, session_id="s1", tool_call_id="c1", next_call=lambda: "ok")
            'ok'
        """
        next_call = cast(Optional[Callable[..., object]], kwargs.get("next_call"))
        plan: _ExecutionPlan
        try:
            plan = self._execution_gate(kwargs)
        except Exception as exc:
            logger.warning("tool_execution gate failed: %s", type(exc).__name__)
            try:
                mode = self.local_mode()
            except Exception:
                mode = "enforce"
            if mode == "enforce":
                plan = _ExecutionPlan(False, error=NO_AUTH_MESSAGE)
            else:
                # Failing open goes through the locked recheck of commands, blocking
                # states and the mode, as every admission without an authorization does.
                raw_args = kwargs.get("args")
                try:
                    plan = self._admit_without_authorization(
                        _str(kwargs.get("session_id")),
                        _str(kwargs.get("tool_name")),
                        _str(kwargs.get("tool_call_id")),
                        raw_args if isinstance(raw_args, dict) else {},
                        {},
                        observe=mode == "observe",
                    )
                except Exception:
                    plan = _ExecutionPlan(False, error=NO_AUTH_MESSAGE)
        if not plan.proceed:
            try:
                self._emit(
                    "tool.call.blocked",
                    _str(kwargs.get("session_id")) or None,
                    span_id=_str(kwargs.get("tool_call_id")) or None,
                    tool={"name": _str(kwargs.get("tool_name"))},
                    reason=(plan.error or "")[:300],
                )
            except Exception as exc:
                logger.debug("blocked event failed: %s", type(exc).__name__)
            return _error_result(plan.error or NO_AUTH_MESSAGE)
        if next_call is None:
            return _error_result(NO_AUTH_MESSAGE)
        # A status left by an earlier invocation with the same identity must not describe
        # this one: only the post_tool_call of this execution may mark it blocked.
        meta_call = _str(plan.meta.get("tool_call_id"))
        if meta_call:
            with self._lock:
                self._post_status.pop(
                    (_str(plan.meta.get("sid")), _str(plan.meta.get("tool")), meta_call), None
                )
        started = time.monotonic()
        outer = getattr(self._invocation, "delegation", None)
        outer_admitted = getattr(self._invocation, "admitted", None)
        self._invocation.delegation = plan.auth.delegation if plan.auth is not None else None
        # The mode this call was admitted under, for the inner pre_tool_call (agent loop
        # order): a stricter mode applied since blocks there instead of authorizing anew.
        # A plan that proceeds without an authorization is never an enforce admission:
        # it is a non-enforce fail-open (an outage, arguments without a canonical form,
        # no permit returned in shadow), recorded as shadow.
        admitted_mode = (
            "observe"
            if plan.observe
            else (plan.auth.effective_mode if plan.auth is not None else "shadow")
        )
        self._invocation.admitted = (
            (_str(plan.meta.get("sid")), _str(plan.meta.get("tool")), meta_call),
            admitted_mode,
            plan.auth is not None,
        )
        try:
            result = next_call()
        except BaseException:
            self._invocation.delegation = outer
            self._invocation.admitted = outer_admitted
            self._after_execution(plan, None, started, raised=True)
            raise
        self._invocation.delegation = outer
        self._invocation.admitted = outer_admitted
        self._after_execution(plan, result, started, raised=False)
        return result

    def _after_execution(
        self, plan: _ExecutionPlan, result: object, started: float, *, raised: bool
    ) -> None:
        try:
            duration_ms = int((time.monotonic() - started) * 1000)
            tool_call_id = _str(plan.meta.get("tool_call_id"))
            post_key = (_str(plan.meta.get("sid")), _str(plan.meta.get("tool")), tool_call_id)
            with self._lock:
                post = self._post_status.get(post_key) if tool_call_id else None
            auth = plan.auth
            if auth is not None:
                with self._lock:
                    auth.state = "done"
            if post == "blocked":
                # Hermes (scope, another plugin, guardrails) blocked it inside next_call: not executed.
                if auth is not None:
                    self._drop_delegation(auth)
                self._emit(
                    "tool.call.not_executed",
                    _str(plan.meta.get("sid")) or None,
                    span_id=tool_call_id or None,
                    tool={"name": _str(plan.meta.get("tool"))},
                    reason="blocked by Hermes after authorization",
                )
                return
            is_error = raised or _result_is_error(result)
            if is_error and auth is not None:
                # Children not started by a failed delegate_task never claim its reservation.
                self._drop_delegation(auth)
            output_hash = None if raised else content_hash(result)
            self._emit(
                "tool.call.executed",
                _str(plan.meta.get("sid")) or None,
                span_id=tool_call_id or None,
                tool={"name": _str(plan.meta.get("tool"))},
                status="error" if is_error else "ok",
                latency_ms=duration_ms,
                output_hash=output_hash,
                content=None if raised else {"output": result},
            )
            if auth is None or auth.record_id is None:
                return
            with self._lock:
                detached = auth.detached
            if detached:
                return
            body: dict[str, Any] = {
                "logical_call_id": auth.logical_call_id,
                "attempt": auth.attempt,
                "tool": auth.tool,
                "arguments": auth.arguments,
                "permit": auth.permit,
                "is_error": is_error,
                "duration_ms": duration_ms,
                "result_hash": output_hash,
                "result_preview": (
                    redacted_preview(result, self.config.capture.preview_chars)
                    if self.config.capture.content == "redacted_preview" and not raised
                    else None
                ),
            }
            self._report(_ReportRetry(auth.session_id, body))
        except Exception as exc:  # reporting never changes the tool result
            logger.warning("post execution handling failed: %s", type(exc).__name__)

    def _report(self, item: _ReportRetry) -> bool:
        try:
            self.client.report(item.session_id, item.body)
            return True
        except HermesApiError as exc:
            item.attempts += 1
            self._emit(
                "action.report_failed",
                item.session_id or None,
                action_id=_str(item.body.get("logical_call_id")) or None,
                attempt_id=item.body.get("attempt")
                if isinstance(item.body.get("attempt"), int)
                else None,
                tool={"name": _str(item.body.get("tool"))},
                reason=str(exc.code)[:100],
                extra={
                    "external_state": "unknown",
                    "http_status": exc.status,
                    "attempts": item.attempts,
                },
            )
            if exc.code == "permit_invalid" or not exc.retryable:
                # Permanent (any 4xx but 408, 425, 429): resending can only fail the same way.
                return False
            if item.attempts < _MAX_REPORT_RETRIES:
                evicted: Optional[_ReportRetry] = None
                with self._report_lock:  # tool threads and the heartbeat share the queue
                    retries = self._report_retries
                    if retries.maxlen is not None and len(retries) == retries.maxlen:
                        # Full: the oldest report is evicted explicitly, never silently. The
                        # event (spooled by the exporter) tells the gateway it is unknown.
                        evicted = retries.popleft()
                    retries.append(item)
                if evicted is not None:
                    self._report_dropped(evicted)
            return False

    def _report_dropped(self, item: _ReportRetry) -> None:
        logger.error("action report dropped after %d attempt(s): retry queue full", item.attempts)
        self._emit(
            "action.report_dropped",
            item.session_id or None,
            action_id=_str(item.body.get("logical_call_id")) or None,
            attempt_id=item.body.get("attempt")
            if isinstance(item.body.get("attempt"), int)
            else None,
            tool={"name": _str(item.body.get("tool"))},
            reason="report retry queue full",
            extra={"external_state": "unknown", "attempts": item.attempts},
        )

    def _retry_reports(self) -> None:
        with self._report_lock:
            pending = len(self._report_retries)
        for _ in range(pending):
            with self._report_lock:
                try:
                    item = self._report_retries.popleft()
                except IndexError:
                    return
            self._report(item)
            self._refresh_status_if_due()

    def post_tool_call(self, **kwargs: object) -> None:
        """Compare executed arguments with the authorized ones; record the terminal status.

        Example:
            >>> _demo_adapter().post_tool_call(tool_name="terminal", args={}, session_id="s1", tool_call_id="c1", status="ok") is None
            True
        """
        try:
            tool_call_id = _str(kwargs.get("tool_call_id"))
            status = _str(kwargs.get("status"))
            if tool_call_id:
                with self._lock:
                    self._post_status[
                        (
                            _str(kwargs.get("session_id")),
                            _str(kwargs.get("tool_name")),
                            tool_call_id,
                        )
                    ] = status
                    while len(self._post_status) > _MAX_AUTH:
                        self._post_status.popitem(last=False)
                    auth = self._auth.get(
                        (
                            _str(kwargs.get("session_id")),
                            _str(kwargs.get("tool_name")),
                            tool_call_id,
                        )
                    )
                    # Direct dispatch: a later pre_tool_call callback blocked the call after
                    # this adapter authorized it, before tool_execution was entered, so no
                    # middleware frame will retire the authorization.
                    unused = False
                    if auth is not None and status == "blocked" and auth.state == "authorized":
                        auth.state = "done"
                        unused = True
            else:
                auth = None
                unused = False
            if unused and auth is not None:
                self._drop_delegation(auth)
                self._emit(
                    "tool.call.not_executed",
                    _str(kwargs.get("session_id")) or None,
                    span_id=tool_call_id or None,
                    tool={"name": _str(kwargs.get("tool_name"))},
                    reason="blocked by Hermes after authorization",
                )
            raw_args = kwargs.get("args")
            args = raw_args if isinstance(raw_args, dict) else {}
            try:
                executed_hash: Optional[str] = arguments_hash(args)
            except CanonicalError:
                executed_hash = None
            if auth is not None and status != "blocked" and executed_hash != auth.local_hash:
                self._mismatch(auth, executed_hash or "", "post_tool_call")
            if _str(kwargs.get("tool_name")) == "skill_manage":
                self._propose_staged_skill(kwargs.get("result"))
            duration = kwargs.get("duration_ms")
            self._emit(
                "tool.call.completed",
                _str(kwargs.get("session_id")) or None,
                span_id=tool_call_id or None,
                parent_span_id=_str(kwargs.get("api_request_id")) or None,
                turn_id=_str(kwargs.get("turn_id")) or None,
                tool={"name": _str(kwargs.get("tool_name"))},
                status=status or None,
                latency_ms=duration if isinstance(duration, int) else None,
                input_hash=executed_hash,
                output_hash=content_hash(kwargs.get("result")),
                extra={"error_type": _str(kwargs.get("error_type")) or None},
            )
        except Exception as exc:
            logger.debug("post_tool_call failed: %s", type(exc).__name__)

    def _propose_staged_skill(self, result: Any) -> None:
        """Submit a skill write that ``skills.write_approval`` staged as an Agenomic change
        proposal. The runtime only proposes: review, approval by a distinct human and
        publication happen in Agenomic, and the supervisor writes published skills."""
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            return
        if not isinstance(parsed, dict) or not parsed.get("staged"):
            return
        pending_id = _str(parsed.get("pending_id"))
        if not pending_id:
            return
        try:
            from tools.write_approval import list_pending, skill_pending_diff
        except ImportError:
            return
        record = next((r for r in list_pending("skills") if r.get("id") == pending_id), None)
        if not isinstance(record, dict):
            return
        raw_payload = record.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        action = _str(payload.get("action"))
        name = _str(payload.get("name"))
        if not name:
            return
        file_path = (
            "SKILL.md"
            if action in ("create", "edit")
            else _str(payload.get("file_path")) or "SKILL.md"
        )
        content_key = "file_content" if action == "write_file" else "content"
        raw_content = payload.get(content_key)
        # An empty string is real content (an empty file): only an absent field falls back
        # to the diff, so the proposal always describes what was staged.
        has_content = isinstance(raw_content, str)
        content = raw_content if isinstance(raw_content, str) else ""
        diff = _str(skill_pending_diff(record))
        target = f"skills/{name}/{file_path}"
        if len(target) > 500:
            # Truncating would propose another path than the one staged: refused instead.
            logger.warning("staged skill write %s not proposed: target too long", pending_id)
            self._emit(
                "skill.proposal.refused",
                None,
                extra={"pending_id": pending_id, "reason_codes": ["target_too_long"]},
            )
            return
        if mask_text(content) != content or mask_text(diff) != diff or mask_text(target) != target:
            # The proposal body leaves the process without the event redaction pipeline,
            # and a masked skill is not what was staged: a reviewer would approve content
            # Hermes never wrote. A credential-bearing proposal is not sent at all.
            logger.warning(
                "staged skill write %s not proposed: it carries a credential", pending_id
            )
            self._emit(
                "skill.proposal.refused",
                None,
                extra={"pending_id": pending_id, "reason_codes": ["credential_detected"]},
            )
            return
        body = {
            "kind": "skill",
            "target": target,
            "content": content if has_content else diff,
            "diff": diff,
            "rationale": mask_text(
                _str(record.get("summary")) or f"Hermes staged {action} {pending_id}"
            )[:4000],
        }
        try:
            proposal = self.client.propose(body)
        except HermesApiError as exc:
            logger.warning("staged skill write not proposed (%s)", exc.code)
            return
        self._emit(
            "skill.proposal.submitted",
            None,
            extra={
                "pending_id": pending_id,
                "proposal_id": proposal.get("id"),
                "status": proposal.get("status"),
            },
        )


def _demo_adapter(answer: Optional[Mapping[str, JsonValue]] = None) -> HermesAdapter:
    """Offline adapter for the examples: a fake gateway answering ``answer`` to every call
    (``observe`` by default), a temporary ``HERMES_HOME``, a compatible Hermes, no threads."""
    import tempfile

    import httpx

    doc: dict[str, JsonValue] = (
        dict(answer)
        if answer is not None
        else {"effective_state": "observe", "decision": "observe", "effective_mode": "observe"}
    )
    config = AdapterConfig.model_validate({"endpoint": "https://a.example"})
    client = RuntimeClient(
        config.endpoint,
        "agmhr_x",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=doc)),
    )
    return HermesAdapter(
        config,
        SecretStr("agmhr_x"),
        client=client,
        hermes_home=Path(tempfile.mkdtemp()),
        start_threads=False,
        identity={"version": "0.21.5", "release_date": None, "commit": None},
    )


#: Acknowledgement order: a later state supersedes a queued earlier one.
_ACK_RANK = {"received": 1, "applied": 2, "refused": 2}


def _guard_max_age_s() -> float:
    try:
        value = float(os.environ.get("AGENOMIC_HERMES_GUARD_MAX_AGE_S") or DEFAULT_MAX_AGE_S)
    except ValueError:
        value = DEFAULT_MAX_AGE_S
    if not math.isfinite(value) or value <= 0:
        value = DEFAULT_MAX_AGE_S
    # The guard refuses deadlines below MIN_MAX_AGE_S, so max_age / 3 never drops below the
    # one second floor of the heartbeat interval.
    return max(value, MIN_MAX_AGE_S)


_ADAPTER: Optional[HermesAdapter] = None


def current_adapter() -> Optional[HermesAdapter]:
    """The adapter registered in this process, if any.

    Example:
        >>> current_adapter() is None or isinstance(current_adapter(), HermesAdapter)
        True
    """
    return _ADAPTER


def register(ctx: object) -> None:
    """Hermes plugin entry point.

    Reads the adapter config, registers hooks and middleware and writes the
    guard status file. A configuration error marks the status file as not
    loaded (so the guard blocks) and is raised so Hermes reports the plugin as
    failed. No network call happens here; ``/hello`` is sent on first use.

    Example:
        >>> register(ctx)  # doctest: +SKIP
        >>> current_adapter().local_mode()  # doctest: +SKIP
        'enforce'
    """
    global _ADAPTER
    home = _hermes_home()

    def failed(error: str) -> None:
        # A failed (re)load never leaves the previous adapter running: its heartbeat would
        # rewrite the status as loaded and its hooks would keep the old configuration.
        global _ADAPTER
        previous, _ADAPTER = _ADAPTER, None
        if previous is not None:
            previous.shutdown()
        with contextlib.suppress(OSError):
            write_status(
                status_path({"HERMES_HOME": str(home)}),
                loaded=False,
                instance_status="unknown",
                effective_state=None,
                error=error,
            )

    try:
        config, token = build_config(settings_from_context(ctx))
    except ConfigError as exc:
        failed("config_error")
        logger.error("Agenomic adapter not loaded: %s", exc)
        raise
    except Exception as exc:  # any unexpected configuration failure is a failed reload too
        failed("config_error")
        logger.error("Agenomic adapter not loaded: %s", type(exc).__name__)
        raise
    assert token is not None
    previous, _ADAPTER = _ADAPTER, None
    if previous is not None:
        previous.shutdown()
    try:
        adapter = HermesAdapter(config, token, ctx=ctx, hermes_home=home)
        adapter.install(ctx)
    except Exception as exc:
        failed("install_error")
        logger.error("Agenomic adapter not loaded: %s", type(exc).__name__)
        raise
    _ADAPTER = adapter
    logger.info("Agenomic adapter %s registered", ADAPTER_VERSION)
