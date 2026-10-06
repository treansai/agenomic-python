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
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

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
    now_iso,
    open_private,
    redacted_preview,
)
from agenomic.integrations.hermes.guard import (
    _SHARING_BACKOFF_S,
    _SHARING_RETRIES,
    DEFAULT_MAX_AGE_S,
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
_ENFORCE_LIKE = {"enforce", "enforce_blocked", "paused", "quarantined", "revoked"}
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
NO_AUTH_MESSAGE = "Agenomic: no valid authorization for this action"
#: Reason recorded when a call's arguments have no canonical form (``agenomic.canon/v1``).
NOT_CANONICAL_REASON = "arguments_not_canonical"


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


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


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
) -> None:
    """Atomically write the status file read by ``agenomic-hermes-guard``.

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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    # A temporary file left by an earlier crash keeps its mode on reopen; it is narrowed.
    fd = open_private(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
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
        self._children: dict[str, tuple[str, Optional[str]]] = {}
        self._delegations: dict[str, deque[list[Any]]] = {}
        # Reservations wait here, per (session, tool, arguments hash), until the action
        # itself is allowed; a retry after an approval or a transport error reuses them.
        # One invocation at a time claims an entry (see ``_Provisional``).
        self._provisional_delegations: dict[tuple[str, str, str], _Provisional] = {}
        # Keyed by (session, tool, tool_call_id): providers reuse call ids across sessions,
        # and an authorization must never serve another session's or tool's call.
        self._auth: OrderedDict[tuple[str, str, str], _Authorization] = OrderedDict()
        self._pending: dict[tuple[str, str, str], _Pending] = {}
        self._post_status: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        # Calls whose arguments had no canonical form and were already recorded, so the
        # second gate of the same call does not record it again.
        self._not_canonical: OrderedDict[tuple[str, str, str], None] = OrderedDict()
        self._report_retries: deque[_ReportRetry] = deque(maxlen=1000)
        self._commands_seen: set[str] = set()
        # Acknowledgements that failed in transport; retried on every tick until accepted.
        self._ack_retries: deque[tuple[str, str, dict[str, Any]]] = deque(maxlen=500)
        self._cancel_sessions: dict[str, str] = {}
        self._cancel_subagents: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Serializes status writes so a heartbeat in flight cannot undo shutdown's "not loaded".
        self._status_lock = threading.Lock()

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
                label = str(getattr(cb, "__qualname__", None) or getattr(cb, "__name__", repr(cb)))
                if label.startswith("shell_hook[pre_tool_call:") and GUARD_COMMAND in label:
                    continue
                found.append(
                    {
                        "kind": kind,
                        "name": name,
                        "callback": label[:200],
                        "module": str(getattr(cb, "__module__", "") or "")[:200],
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
        if self._hello():
            try:
                self.discover_tools()
            except HermesApiError as exc:
                logger.warning("tool discovery failed (%s)", exc.code)
        self._write_status()
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

    def _set_state(self, state: object) -> None:
        if isinstance(state, str) and state:
            self._effective_state = state

    def _instance_status(self) -> str:
        if self._local_status:
            return self._local_status
        state = self._effective_state
        if state is None:
            return "unknown"
        return state if state in _BLOCKING_STATUS else "active"

    def _gates_registered(self) -> bool:
        # Either gate can ask the gateway and refuse; without both the guard must keep
        # blocking, because no callback would consult Agenomic before a tool runs.
        return bool(self._contracts["pre_tool_call"] or self._contracts["tool_execution"])

    def _write_status(self) -> None:
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
                )
            except OSError as exc:
                logger.warning("status file not written: %s", type(exc).__name__)

    def _provider(self) -> dict[str, Any]:
        model = _hermes_config().get("model")
        if not isinstance(model, dict):
            return {}
        return {
            "provider": _str(model.get("provider")) or None,
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
            if isinstance(e, dict) and GUARD_COMMAND in _str(e.get("command"))
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
        return {
            "hermes": self._identity,
            "adapter": {"version": ADAPTER_VERSION, "config_schema": CONFIG_SCHEMA},
            "contracts": self._contracts,
            "foreign_mutators": foreign,
            "provider": self._provider(),
            "compat_results": self._compat_results(),
            "platform": self._platform or "cli",
        }

    def _hello(self) -> bool:
        self._hello_attempt_at = time.monotonic()
        foreign = self.foreign_mutators()
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
        self._set_state(resp.get("effective_state"))
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
        for name in sorted(registry.get_all_tool_names()):
            schema = registry.get_schema(name)
            if not isinstance(schema, dict):
                continue
            try:
                digest = schema_hash(schema)
            except CanonicalError:
                continue
            self._schema_hashes[name] = digest
            toolset = _str(registry.get_toolset_for_tool(name))
            entry: dict[str, Any] = {
                "tool_name": name,
                "source": "builtin",
                "schema_hash": digest,
                "input_schema": schema.get("parameters")
                if isinstance(schema.get("parameters"), dict)
                else {},
            }
            if toolset.startswith("mcp-"):
                entry["source"] = "mcp"
                entry["mcp_server"] = toolset[4:]
            elif name in plugin_tools:
                entry["source"] = "plugin"
            tools.append(entry)
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
            with self._lock:
                active: list[JsonValue] = [
                    s.hermes_session_id for s in self._sessions.values() if s.active
                ]
            resp = self.client.heartbeat(
                {"active_sessions": active, "exporter": dict(self.exporter.stats())}
            )
            self._set_state(resp.get("effective_state"))
            commands = resp.get("commands")
            if isinstance(commands, list):
                for command in commands:
                    if isinstance(command, dict):
                        self.handle_command(command)
        except HermesApiError as exc:
            logger.warning("Agenomic heartbeat failed (%s)", exc.code)
        finally:
            self._retry_acks()
            self._retry_reports()
            self._write_status()

    def _ack(self, command_id: str, status: str, detail: dict[str, Any]) -> bool:
        try:
            self.client.ack_command(command_id, status, detail)
        except HermesApiError as exc:
            logger.warning("command %s ack %s failed (%s)", command_id, status, exc.code)
            if exc.status == 0 or exc.status >= 500:
                self._ack_retries.append((command_id, status, detail))
            return False
        self._drop_superseded_acks(command_id, status)
        self._emit("command." + status, None, extra={"command_id": command_id, "detail": detail})
        return True

    def _drop_superseded_acks(self, command_id: str, status: str) -> None:
        rank = _ACK_RANK.get(status, 0)
        kept = [
            item
            for item in self._ack_retries
            if item[0] != command_id or _ACK_RANK.get(item[1], 0) > rank
        ]
        if len(kept) != len(self._ack_retries):
            self._ack_retries.clear()
            self._ack_retries.extend(kept)

    def _retry_acks(self) -> None:
        for _ in range(len(self._ack_retries)):
            try:
                command_id, status, detail = self._ack_retries.popleft()
            except IndexError:
                return
            self._ack(command_id, status, detail)

    def handle_command(self, command: Mapping[str, JsonValue]) -> None:
        """Execute one plugin command. ``applied`` is only acknowledged once observed.

        Example:
            >>> a = _demo_adapter()
            >>> a.handle_command({"id": "c1", "kind": "pause", "target_kind": "instance"})
            >>> json.loads(a.status_file.read_text())["instance_status"]
            'paused'
        """
        command_id = _str(command.get("id"))
        if not command_id or command_id in self._commands_seen:
            return
        self._commands_seen.add(command_id)
        kind = _str(command.get("kind"))
        target_kind = _str(command.get("target_kind")) or "instance"
        target = _str(command.get("target_ref"))
        if _str(command.get("status")) in ("requested", ""):
            self._ack(command_id, "received", {"executor": "plugin"})
        if target_kind == "instance" and kind in ("pause", "revoke"):
            self._local_status = "paused" if kind == "pause" else "revoked"
            self._write_status()
            self._ack(command_id, "applied", {"local_state": self._local_status})
        elif target_kind == "instance" and kind == "resume":
            self._local_status = None
            self._write_status()
            self._ack(command_id, "applied", {"local_state": "active"})
        elif target_kind == "instance" and kind == "quarantine":
            # Quarantine is a process stop by the supervisor; the plugin only blocks locally.
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
        if self._interrupt_subagent(subagent_id):
            self._cancel_subagents[subagent_id] = command_id
        else:
            self._ack(command_id, "refused", {"reason": "subagent_not_running"})

    def _cancel_session(self, command_id: str, target: str) -> None:
        sid = self._agenomic_sessions.get(target, target)
        with self._lock:
            session = self._sessions.get(sid)
        if session is None or not session.active:
            self._ack(command_id, "refused", {"reason": "session_not_active"})
            return
        self._cancel_sessions[sid] = command_id
        if session.subagent_id and self._interrupt_subagent(session.subagent_id):
            self._cancel_subagents[session.subagent_id] = command_id
        # Root sessions expose no interrupt handle to plugins: further tool calls are blocked and
        # the command is applied once Hermes reports the session's end.

    def _cancel_pending(self, sid: str) -> bool:
        with self._lock:
            session = self._sessions.get(sid)
            subagent_id = (
                session.subagent_id if session else self._children.get(sid, (None, None))[1]
            )
        return sid in self._cancel_sessions or bool(
            subagent_id and subagent_id in self._cancel_subagents
        )

    def _observe_terminal(self, sid: str, subagent_id: Optional[str], how: str) -> None:
        command_id = self._cancel_sessions.pop(sid, None) if sid else None
        if subagent_id:
            command_id = self._cancel_subagents.pop(subagent_id, None) or command_id
        if command_id:
            self._ack(command_id, "applied", {"observed": how, "hermes_session_id": sid})

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
                    session.delegation_id = self._take_delegation(link[0])
                self._sessions[sid] = session
            if platform and not session.platform:
                session.platform = platform
            if model and not session.model:
                session.model = model
            return session

    def _take_delegation(self, parent: str) -> Optional[str]:
        queue = self._delegations.get(parent)
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
            "platform": session.platform or self._platform or "cli",
        }
        if session.model:
            body["model"] = session.model
        if session.parent:
            body["parent_hermes_session_id"] = session.parent
        if session.subagent_id:
            body["subagent_id"] = session.subagent_id
        if session.delegation_id:
            body["delegation_id"] = session.delegation_id
        try:
            resp = self.client.create_session(body)
        except HermesApiError as exc:
            logger.warning("session admission failed (%s)", exc.code)
            return
        session.admitted = True
        self._set_state(resp.get("effective_state"))
        info = resp.get("session")
        if isinstance(info, dict) and isinstance(info.get("id"), str):
            session.agenomic_id = cast(str, info["id"])
            self._agenomic_sessions[cast(str, info["id"])] = session.hermes_session_id

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

    def _end(self, sid: str, final: bool, status: str, reason: str = "") -> None:
        body: dict[str, Any] = {"final": final, "status": status}
        if reason:
            body["reason"] = reason[:200]
        try:
            self.client.end_session(sid, body)
        except HermesApiError as exc:
            logger.warning("session end not reported (%s)", exc.code)

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
            self._end(
                sid, False, status, _str(kwargs.get("turn_exit_reason") or kwargs.get("reason"))
            )
            self._emit(
                "session.turn_ended",
                sid,
                turn_id=_str(kwargs.get("turn_id")) or None,
                status=status,
                reason=_str(kwargs.get("turn_exit_reason"))[:200] or None,
            )
            if interrupted:
                session = self._sessions.get(sid)
                self._observe_terminal(
                    sid, session.subagent_id if session else None, "on_session_end interrupted"
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
            self._end(sid, True, status, reason)
            with self._lock:
                if session is not None:
                    session.active = False
            self._observe_terminal(sid, subagent_id, "on_session_finalize")
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
                with self._lock:
                    self._children[child] = (parent, subagent_id)
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
                self._end(child, True, status, "subagent_stop")
                with self._lock:
                    session = self._sessions.get(child)
                    if session is not None:
                        session.active = False
                self._observe_terminal(child, subagent_id, "subagent_stop")
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
            if new_headers.get("X-Agenomic-Hermes-Session") == sid:
                return None
            new_headers["X-Agenomic-Hermes-Session"] = sid
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
        resp = self.client.reserve_delegation(sid, {"count": count, "tool_call_id": tool_call_id})
        decision = _str(resp.get("decision"))
        if decision == "observe":
            return None, None
        if decision == "allow" and isinstance(resp.get("delegation_id"), str):
            return None, [resp["delegation_id"], count]
        if decision == "deny":
            codes = resp.get("reason_codes")
            explanation = _str(resp.get("explanation")) or ", ".join(
                str(c) for c in (codes if isinstance(codes, list) else [])
            )
            return f"Agenomic denied delegate_task: {explanation or 'delegation limit'}", None
        raise HermesApiError("invalid_response", "delegation answer without a valid decision", 200)

    def _claim_delegation(
        self,
        *,
        key: tuple[str, str, str],
        claim: str,
        sid: str,
        tool: str,
        args: Mapping[str, Any],
        tool_call_id: str,
        local_hash: str,
        mode: LocalMode,
    ) -> tuple[Optional[str], Optional[_Provisional]]:
        """``(block, provisional)`` for this invocation of ``delegate_task``.

        An unclaimed provisional entry (a retry of the same action) is claimed and reused;
        otherwise this invocation reserves its own, so two concurrent identical
        invocations never share one reservation.
        """
        with self._lock:
            entry = self._provisional_delegations.get(key)
            if entry is not None and entry.claimed_by in (None, claim):
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
        provisional = _Provisional(reservation, claimed_by=claim)
        with self._lock:
            # Published only when free: a retry of this action then finds and reuses it.
            self._provisional_delegations.setdefault(key, provisional)
        return None, provisional

    def _release_delegation(
        self, key: tuple[str, str, str], provisional: Optional[_Provisional]
    ) -> None:
        """This invocation stops deciding: an unsettled reservation waits for a retry."""
        if provisional is None:
            return
        with self._lock:
            if provisional.settled:
                return
            current = self._provisional_delegations.get(key)
            if current is None:
                self._provisional_delegations[key] = provisional
            elif current is not provisional:
                # Another invocation's reservation already waits for this action.
                provisional.settled = True
                return
            provisional.claimed_by = None

    def _settle_delegation(
        self, key: tuple[str, str, str], provisional: Optional[_Provisional], commit: bool
    ) -> Optional[list[Any]]:
        """Queue this invocation's reservation for the child once the action is allowed,
        or forget it; another invocation's reservation is never touched."""
        if provisional is None:
            return None
        with self._lock:
            provisional.settled = True
            if self._provisional_delegations.get(key) is provisional:
                del self._provisional_delegations[key]
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
            if self._pending.get(key) is pending:
                del self._pending[key]

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
        session = self._session(sid)
        if not session.admitted:
            self._admit(session)
        mode = self.local_mode()
        unconfirmed = self._confirm_foreign_mutators(sid, tool, tool_call_id, local_hash, mode)
        if unconfirmed is not None:
            return _Verdict(block=unconfirmed)
        key = (sid, tool, local_hash)
        claim = tool_call_id or f"hermes-{uuid.uuid4().hex}"
        with self._lock:
            # An approval required while enforcing stays for a later enforce; in shadow the
            # call is asked afresh, so a pending or refused approval never blocks it.
            pending = self._pending.get(key) if mode != "shadow" else None
            previous = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
            in_use = pending is not None and pending.claimed_by not in (None, claim)
            if pending is not None and not in_use:
                pending.claimed_by = claim
        if pending is not None and in_use:
            # Another invocation is retrying under this approval right now; it is the one
            # the approval authorizes. This one is not resumed under the same identity.
            message = APPROVAL_IN_USE_MESSAGE.format(approval_id=pending.approval_id)
            self._emit_decision(
                sid,
                tool,
                tool_call_id,
                "require_approval",
                message,
                local_hash,
                extra={"local": True, "approval_id": pending.approval_id},
            )
            return _Verdict(block=message)
        provisional: Optional[_Provisional] = None
        try:
            if tool == _DELEGATE_TOOL:
                # After the approval claim: the invocation retrying under an approval is the
                # one that reuses the reservation associated with it.
                denied, provisional = self._claim_delegation(
                    key=key,
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
            self._release_delegation(key, provisional)
            if pending is not None:
                with self._lock:
                    # Still waiting (pending approval, transport error, observe): the next
                    # retry may claim it. An allowed or denied call already removed it.
                    if pending.claimed_by == claim:
                        pending.claimed_by = None

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
            gate, keep = self._approval_gate(pending)
            if gate is not None:
                if not keep:
                    self._drop_pending(key, pending)
                    self._settle_delegation(key, provisional, commit=False)
                return _Verdict(block=gate)
            logical_call_id, attempt = pending.logical_call_id, pending.attempt
        else:
            logical_call_id = tool_call_id or f"hermes-{uuid.uuid4().hex}"
            attempt = (
                previous.attempt + 1 if previous is not None and previous.state == "done" else 1
            )
        body: dict[str, Any] = {
            "tool_call_id": logical_call_id,
            "tool": tool,
            "arguments": args,
            "attempt": attempt,
            "schema_hash": self._schema_hash_for(tool),
            "turn_id": turn_id or None,
            "api_request_id": api_request_id or None,
        }
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
        if decision == "observe" or effective_mode == "observe":
            self._effective_state = "observe"
            self._settle_delegation(key, provisional, commit=False)
            return _Verdict()
        if effective_mode == "shadow" and self._effective_state not in _ENFORCE_LIKE - {"enforce"}:
            self._effective_state = "shadow"
        elif effective_mode == "enforce" and self._effective_state not in _ENFORCE_LIKE:
            self._effective_state = "enforce"
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
            self._settle_delegation(key, provisional, commit=False)
            return _Verdict(
                block=f"Agenomic denied {tool}: {explanation or 'policy'} (decision {decision_id or 'unknown'})"
            )
        if decision == "require_approval" and not shadow:
            approval_id = _str(resp.get("approval_id"))
            if not approval_id:
                raise HermesApiError(
                    "invalid_response", "require_approval without approval_id", status
                )
            with self._lock:
                current = self._pending.get(key)
                # Never replace an approval another invocation is bound to.
                if current is None or current is pending:
                    self._pending[key] = _Pending(
                        _str(resp.get("logical_call_id")) or logical_call_id,
                        int(cast(Any, resp.get("attempt")) or attempt),
                        approval_id,
                    )
            return _Verdict(block=APPROVAL_MESSAGE.format(approval_id=approval_id))
        permit = resp.get("permit")
        record_id = _str(resp.get("record_id")) or None
        if not shadow and (record_id is None or not isinstance(permit, dict)):
            raise HermesApiError("invalid_response", "allow without record_id and permit", status)
        auth = _Authorization(
            tool_call_id=tool_call_id or logical_call_id,
            session_id=sid,
            tool=tool,
            logical_call_id=_str(resp.get("logical_call_id")) or logical_call_id,
            attempt=int(cast(Any, resp.get("attempt")) or attempt),
            local_hash=local_hash,
            arguments=args,
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
                self._settle_delegation(key, provisional, commit=False)
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
        auth.delegation = self._settle_delegation(key, provisional, commit=True)
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
        reason = (
            f"Agenomic: Hermes callbacks that can change the arguments of {tool} changed and "
            "the gateway has not confirmed them; the action was not executed."
        )
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

    def _local_checks(self, tool: str, args: Mapping[str, Any]) -> Optional[str]:
        """Defence in depth applied in enforce after the gateway allowed."""
        if not self.hermes_compatible:
            return (
                f"Agenomic: Hermes {self._identity.get('version')} is not in the adapter "
                "compatibility table; protected actions are blocked in enforce"
            )
        hits = self.protected_targets(tool, args)
        if hits:
            return (
                f"Agenomic denied {tool}: writes to protected Hermes paths need a reviewed "
                "proposal (local check)"
            )
        return None

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
        if mode == "shadow":
            return None
        return f"Agenomic authorization unavailable ({code}); the action was not executed."

    def _arguments_not_canonical(
        self, sid: str, tool: str, tool_call_id: str, args: Mapping[str, object]
    ) -> Optional[str]:
        """Arguments without a canonical form (NaN, a set, an object another plugin put
        there) cannot be authorized: blocked in enforce; in shadow and observe the call
        proceeds and a local ``tool.call.decision`` records the counterfactual deny.
        Recorded once per call, whichever gate sees it first."""
        mode = self.local_mode()
        key = (sid, tool, tool_call_id)
        with self._lock:
            recorded = bool(tool_call_id) and key in self._not_canonical
            if tool_call_id and not recorded:
                self._not_canonical[key] = None
                while len(self._not_canonical) > _MAX_AUTH:
                    self._not_canonical.popitem(last=False)
        if not recorded:
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

    def _blocked_session(self, sid: str) -> Optional[str]:
        if sid and sid in self._cancel_sessions:
            return "Agenomic cancelled this session; the action was not executed."
        return None

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
            cancelled = self._blocked_session(sid)
            if cancelled:
                return _block(cancelled)
            try:
                local_hash = arguments_hash(args)
            except CanonicalError:
                message = self._arguments_not_canonical(sid, tool, tool_call_id, args)
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
            if self.local_mode() == "observe":
                return None
            with self._lock:
                existing = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
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
            if self.local_mode() == "observe":
                return None
            return _block(message) if message else None

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
        if auth.effective_mode == "shadow":
            return None
        return _block(
            f"Agenomic: arguments of {auth.tool} changed after authorization; the action was not executed."
        )

    # ------------------------------------------------------------------
    # execution and reporting
    # ------------------------------------------------------------------
    def _execution_gate(self, kwargs: Mapping[str, Any]) -> _ExecutionPlan:
        tool = _str(kwargs.get("tool_name"))
        sid = _str(kwargs.get("session_id"))
        tool_call_id = _str(kwargs.get("tool_call_id"))
        raw_args = kwargs.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        self._ensure_started()
        meta = {"tool": tool, "sid": sid, "tool_call_id": tool_call_id, "args": args}
        cancelled = self._blocked_session(sid)
        if cancelled:
            return _ExecutionPlan(False, error=cancelled, meta=meta)
        if self.local_mode() == "observe":
            return _ExecutionPlan(True, observe=True, meta=meta)
        try:
            local_hash = arguments_hash(args)
        except CanonicalError:
            message = self._arguments_not_canonical(sid, tool, tool_call_id, args)
            if message:
                return _ExecutionPlan(False, error=message, meta=meta)
            return _ExecutionPlan(True, meta=meta)
        meta["local_hash"] = local_hash
        with self._lock:
            auth = self._auth.get((sid, tool, tool_call_id)) if tool_call_id else None
            if auth is not None and auth.state == "done":
                auth = None
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
                return _ExecutionPlan(True, meta=meta)
            if verdict.block:
                return _ExecutionPlan(False, error=verdict.block, meta=meta)
            if self.local_mode() == "observe":
                return _ExecutionPlan(True, observe=True, meta=meta)
            auth = verdict.authorization
            if auth is None:
                if self.local_mode() == "shadow":
                    return _ExecutionPlan(True, meta=meta)
                return _ExecutionPlan(False, error=NO_AUTH_MESSAGE, meta=meta)
        elif auth.local_hash != local_hash:
            blocked = self._mismatch(auth, local_hash, "tool_execution")
            if blocked is not None:
                return _ExecutionPlan(False, error=blocked["message"], meta=meta)
        with self._lock:
            if auth.state == "executing":
                # A second chain run for the same tool_call_id while the first is executing.
                if auth.effective_mode == "shadow":
                    return _ExecutionPlan(True, meta=meta)
                return _ExecutionPlan(False, error=NO_AUTH_MESSAGE, meta=meta)
            auth.state = "executing"
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
                plan = _ExecutionPlan(True, observe=mode == "observe")
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
        try:
            result = next_call()
        except BaseException:
            self._after_execution(plan, None, started, raised=True)
            raise
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
            if exc.code == "permit_invalid" or exc.status in (400, 403, 404, 409, 422):
                return False
            if item.attempts < _MAX_REPORT_RETRIES:
                self._report_retries.append(item)
            return False

    def _retry_reports(self) -> None:
        for _ in range(len(self._report_retries)):
            try:
                item = self._report_retries.popleft()
            except IndexError:
                return
            self._report(item)

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
            else:
                auth = None
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
        content = _str(payload.get(content_key))
        diff = skill_pending_diff(record)
        body = {
            "kind": "skill",
            "target": f"skills/{name}/{file_path}"[:500],
            "content": content or diff,
            "diff": diff,
            "rationale": (_str(record.get("summary")) or f"Hermes staged {action} {pending_id}")[
                :4000
            ],
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
    try:
        config, token = build_config(settings_from_context(ctx))
    except ConfigError as exc:
        with contextlib.suppress(OSError):
            write_status(
                status_path({"HERMES_HOME": str(home)}),
                loaded=False,
                instance_status="unknown",
                effective_state=None,
                error="config_error",
            )
        logger.error("Agenomic adapter not loaded: %s", exc)
        raise
    assert token is not None
    previous = _ADAPTER
    if previous is not None:
        previous.shutdown()
    adapter = HermesAdapter(config, token, ctx=ctx, hermes_home=home)
    adapter.install(ctx)
    _ADAPTER = adapter
    logger.info("Agenomic adapter %s registered", ADAPTER_VERSION)
