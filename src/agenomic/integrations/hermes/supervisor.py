"""``agenomic-hermes-supervisor``: the trusted process that starts and stops Hermes.

It runs outside the agent's control with its own credential
(``AGENOMIC_HERMES_SUPERVISOR_TOKEN``, ``agmhs_``) which is never passed to the
child. It starts Hermes as a child process (argv after ``--``) with an
environment built from an explicit allowlist, heartbeats
``/v1/hermes/supervisor/heartbeat`` with the process state and an isolation
self check, executes supervisor commands (``quarantine``/``revoke`` stop the
process and refuse restarts, ``resume`` allows them again) and syncs approved
skills into a directory the agent cannot write.

The isolation self check reports what this process can observe from inside
the same container and network namespace as the child; it attests a
deployment, it does not create one.

Example:
    >>> env = build_child_env(
    ...     {"PATH": "/bin", "OPENAI_API_KEY": "sk-x", "AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_x",
    ...      "AGENOMIC_HERMES_SUPERVISOR_TOKEN": "agmhs_x", "GITHUB_TOKEN": "g"},
    ... )
    >>> sorted(env)
    ['AGENOMIC_HERMES_RUNTIME_TOKEN', 'PATH']
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from agenomic.integrations.hermes.client import HermesApiError, SupervisorClient
from agenomic.integrations.hermes.config import DEFAULT_TOKEN_ENV
from agenomic.integrations.hermes.exporter import now_iso

logger = logging.getLogger("agenomic.integrations.hermes.supervisor")

SUPERVISOR_TOKEN_ENV = "AGENOMIC_HERMES_SUPERVISOR_TOKEN"
ENDPOINT_ENV = "AGENOMIC_HERMES_ENDPOINT"
DEFAULT_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "TMPDIR",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HERMES_HOME",
    "HERMES_ACCEPT_HOOKS",
    "AGENOMIC_HERMES_CONFIG",
    "SYSTEMROOT",
    "AGENOMIC_HERMES_GUARD_MAX_AGE_S",
)
DEFAULT_FORBIDDEN_HOSTS = ("api.openai.com:443", "api.anthropic.com:443", "openrouter.ai:443")
DOCKER_SOCKETS = ("/var/run/docker.sock", "/run/docker.sock")
_SECRET_NAME = re.compile(
    r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_ACCESS_KEY|_PRIVATE_KEY)$|^API_KEY$", re.I
)
_PROVIDER_KEYS = re.compile(
    r"_API_KEY$|^API_KEY$|^AWS_SECRET_ACCESS_KEY$|^GOOGLE_APPLICATION_CREDENTIALS$", re.I
)
#: Prefix of every Agenomic runtime token (the credential the Hermes plugin presents).
RUNTIME_TOKEN_PREFIX = "agmhr_"


def runtime_token_env_problem(name: str) -> Optional[str]:
    """Why ``name`` cannot carry the runtime token into the child, or ``None``.

    The runtime token variable is the only credential the child receives, so it must
    not be the supervisor token, a provider key, or any other credential shaped name
    (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET``, ``*_PASSWORD``...) unless it is an
    ``AGENOMIC_`` name.

    Example:
        >>> runtime_token_env_problem("AGENOMIC_HERMES_RUNTIME_TOKEN") is None
        True
        >>> runtime_token_env_problem("OPENAI_API_KEY")
        'OPENAI_API_KEY is a provider credential name, not the Agenomic runtime token'
    """
    if name == SUPERVISOR_TOKEN_ENV:
        return f"{name} is the supervisor credential and is never passed to the child"
    if _PROVIDER_KEYS.search(name):
        return f"{name} is a provider credential name, not the Agenomic runtime token"
    if _SECRET_NAME.search(name) and not name.upper().startswith("AGENOMIC_"):
        return f"{name} names another credential; use an AGENOMIC_ variable for the runtime token"
    return None


def runtime_token_problem(
    source: Mapping[str, str], runtime_token_env: str = DEFAULT_TOKEN_ENV
) -> Optional[str]:
    """Why the runtime token in ``source`` cannot be given to the child, or ``None``.

    An absent variable is not a problem (the child then has no runtime credential);
    a present one must be a valid name holding an ``agmhr_`` token.

    Example:
        >>> runtime_token_problem({"AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_x"}) is None
        True
        >>> runtime_token_problem({"AGENOMIC_HERMES_RUNTIME_TOKEN": "sk-x"})
        'AGENOMIC_HERMES_RUNTIME_TOKEN does not hold an Agenomic runtime token (agmhr_...)'
    """
    problem = runtime_token_env_problem(runtime_token_env)
    if problem is not None:
        return problem
    value = source.get(runtime_token_env)
    if value is not None and not value.startswith(RUNTIME_TOKEN_PREFIX):
        return (
            f"{runtime_token_env} does not hold an Agenomic runtime token "
            f"({RUNTIME_TOKEN_PREFIX}...)"
        )
    return None


def _runtime_token_exempt(name: str, source: Mapping[str, str], runtime_token_env: str) -> bool:
    """``name`` is the runtime token variable, validly named and holding an ``agmhr_`` value."""
    return (
        name == runtime_token_env
        and runtime_token_env_problem(name) is None
        and source.get(name, "").startswith(RUNTIME_TOKEN_PREFIX)
    )


def build_child_env(
    source: Mapping[str, str],
    *,
    allow: Iterable[str] = (),
    runtime_token_env: str = DEFAULT_TOKEN_ENV,
) -> dict[str, str]:
    """Child environment from an explicit allowlist.

    Only allowlisted names are copied; any ``*_API_KEY``, ``*_TOKEN``,
    ``*_SECRET``... is removed even when allowlisted, except the runtime
    token variable when its name is valid (:func:`runtime_token_env_problem`) and it
    holds an ``agmhr_`` token. The supervisor token is never copied.

    Example:
        >>> build_child_env({"HOME": "/h", "X": "1"}, allow=["X"])
        {'HOME': '/h', 'X': '1'}
    """
    names = set(DEFAULT_ENV_ALLOWLIST) | set(allow) | {runtime_token_env}
    env: dict[str, str] = {}
    for name in sorted(names):
        if name == SUPERVISOR_TOKEN_ENV or name not in source:
            continue
        if name == runtime_token_env:
            if not _runtime_token_exempt(name, source, runtime_token_env):
                continue
        elif _SECRET_NAME.search(name):
            continue
        env[name] = source[name]
    return env


def provider_secrets_absent(
    env: Mapping[str, str], *, runtime_token_env: str = DEFAULT_TOKEN_ENV
) -> bool:
    """No provider key and no supervisor credential in ``env``.

    Only a validly named runtime token variable holding an ``agmhr_`` token is exempt,
    so a provider key configured as the runtime token variable is still reported.

    Example:
        >>> provider_secrets_absent({"OPENAI_API_KEY": "x"})
        False
        >>> provider_secrets_absent({"OPENAI_API_KEY": "sk-x"}, runtime_token_env="OPENAI_API_KEY")
        False
    """
    for name in env:
        if name == SUPERVISOR_TOKEN_ENV:
            return False
        if _runtime_token_exempt(name, env, runtime_token_env):
            continue
        if _PROVIDER_KEYS.search(name) or _SECRET_NAME.search(name):
            return False
    return True


class _Closable(Protocol):
    def close(self) -> None: ...


#: ``socket.create_connection`` shape: ``connect((host, port), timeout_s)`` returns a socket.
_Connect = Callable[[tuple[str, int], float], _Closable]


def _parse_host(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    if not host:
        return spec, 443
    return host.strip("[]"), int(port)


def egress_restricted(
    hosts: Sequence[str], *, timeout_s: float = 2.0, connect: _Connect = socket.create_connection
) -> bool:
    """``True`` only when a TCP connect to every forbidden host fails.

    An empty list cannot attest anything and returns ``False``.

    Example:
        >>> def refuse(addr, timeout):
        ...     raise OSError("unreachable")
        >>> egress_restricted(["api.openai.com:443"], connect=refuse)
        True
        >>> egress_restricted([])
        False
    """
    if not hosts:
        return False
    for spec in hosts:
        try:
            host, port = _parse_host(spec)
        except ValueError:
            return False
        try:
            sock = connect((host, port), timeout_s)
        except OSError:
            continue
        with contextlib.suppress(OSError):
            sock.close()
        logger.warning("egress check: connection to %s succeeded", spec)
        return False
    return True


# The supervisor targets POSIX hosts (process groups, uids, read only mounts); these
# fallbacks only keep the module importable and type checked on Windows.
if sys.platform == "win32":
    _KILL_SIGNAL = signal.SIGTERM
else:
    _KILL_SIGNAL = signal.SIGKILL


def _current_uid() -> int:
    if sys.platform == "win32":
        return -1
    return os.getuid()


def _readonly_fs(path: Path) -> bool:
    if sys.platform == "win32":
        return False
    try:
        return bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError:
        return False


def _mode_allows_write(st: os.stat_result, uid: int, groups: set[int]) -> bool:
    mode = st.st_mode
    if st.st_uid == uid:
        return bool(mode & stat.S_IWUSR)
    if st.st_gid in groups:
        return bool(mode & stat.S_IWGRP)
    return bool(mode & stat.S_IWOTH)


def writable_by(path: Path, uid: int, gids: Iterable[int]) -> bool:
    """Whether ``uid`` (with ``gids``) could modify ``path`` or replace it in its directory.

    Mode bits and ownership are checked for the file, then for its parent
    directory (a writable directory lets the file be renamed or removed), with
    POSIX sticky bit semantics: in a sticky directory only the owner of the
    entry or of the directory may remove or rename it. A missing path is
    writable when its nearest existing ancestor is (the child can create the
    missing directories). A read only mount wins. Root can write anything that
    is not on a read only mount.

    Example:
        >>> import tempfile
        >>> d = Path(tempfile.mkdtemp())
        >>> writable_by(d, os.getuid(), [os.getgid()])
        True
    """
    groups = set(gids)
    parent = path.parent
    exists = path.exists()
    if (
        exists
        and not _readonly_fs(path)
        and (uid == 0 or _mode_allows_write(path.stat(), uid, groups))
    ):
        return True
    if not exists:
        ancestor = parent
        while not ancestor.exists():
            if ancestor.parent == ancestor:
                return False
            ancestor = ancestor.parent
        if _readonly_fs(ancestor):
            return False
        return uid == 0 or _mode_allows_write(ancestor.stat(), uid, groups)
    if parent == path or _readonly_fs(parent):
        return False
    if uid == 0:
        return True
    pst = parent.stat()
    if not _mode_allows_write(pst, uid, groups):
        return False
    sticky_protected = (
        exists
        and bool(pst.st_mode & stat.S_ISVTX)
        and path.stat().st_uid != uid
        and pst.st_uid != uid
    )
    return not sticky_protected


def isolation_report(
    child_env: Mapping[str, str],
    *,
    child_uid: int,
    child_gids: Iterable[int],
    config_paths: Sequence[Path],
    skills_paths: Sequence[Path],
    forbidden_hosts: Sequence[str],
    runtime_token_env: str = DEFAULT_TOKEN_ENV,
    connect: _Connect = socket.create_connection,
    docker_sockets: Sequence[str] = DOCKER_SOCKETS,
) -> dict[str, JsonValue]:
    """Isolation attestation sent with every supervisor heartbeat.

    Example:
        >>> r = isolation_report({}, child_uid=1000, child_gids=[1000], config_paths=[],
        ...     skills_paths=[], forbidden_hosts=[], docker_sockets=[])
        >>> r["provider_secrets_absent"], r["egress_restricted"], r["runs_as_non_root"]
        (True, False, True)
    """
    gids = list(child_gids)
    return {
        "provider_secrets_absent": provider_secrets_absent(
            child_env, runtime_token_env=runtime_token_env
        ),
        "egress_restricted": egress_restricted(forbidden_hosts, connect=connect),
        "config_readonly": bool(config_paths)
        and not any(writable_by(p, child_uid, gids) for p in config_paths),
        "skills_readonly": bool(skills_paths)
        and not any(writable_by(p, child_uid, gids) for p in skills_paths),
        "docker_socket_absent": not any(os.path.exists(s) for s in docker_sockets),
        "runs_as_non_root": child_uid != 0,
        "checked_at": now_iso(),
    }


class _SupervisorApi(Protocol):
    def heartbeat(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]: ...

    def ack_command(
        self, command_id: str, status: str, detail: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]: ...

    def approved_skills(self) -> dict[str, JsonValue]: ...


#: Acknowledgement order: a later state supersedes a queued earlier one.
_ACK_RANK = {"received": 1, "applied": 2, "refused": 2}


class SupervisorSettings(BaseModel):
    """Supervisor configuration (from the command line), validated on construction and
    on assignment.

    Example:
        >>> SupervisorSettings(argv=["hermes", "chat"], hermes_home=Path("/srv/hermes")).interval_s
        15.0
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    argv: list[str] = Field(min_length=1)
    hermes_home: Path
    skills_dir: Optional[Path] = None
    config_paths: list[Path] = Field(default_factory=list)
    readonly_paths: list[Path] = Field(default_factory=list)
    forbidden_hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_HOSTS))
    env_allow: list[str] = Field(default_factory=list)
    child_uid: Optional[int] = Field(default=None, ge=0)
    child_gid: Optional[int] = Field(default=None, ge=0)
    grace_s: float = Field(default=10.0, gt=0)
    interval_s: float = Field(default=15.0, gt=0)
    skills_every: int = Field(default=4, ge=1)
    restart: bool = True
    max_restarts: int = Field(default=5, ge=0)
    runtime_token_env: str = Field(default=DEFAULT_TOKEN_ENV, min_length=1)

    @field_validator("runtime_token_env")
    @classmethod
    def _runtime_token_env_is_not_a_provider_credential(cls, value: str) -> str:
        problem = runtime_token_env_problem(value)
        if problem is not None:
            raise ValueError(problem)
        return value


def _digest_matches(content: bytes, digest: str) -> bool:
    if digest.startswith("sha256:"):
        return hashlib.sha256(content).hexdigest() == digest[7:]
    if digest.startswith("blake3:"):
        from agenomic.crypto.hashing import blake3_hex

        return blake3_hex(content) == digest[7:]
    return False


_MANIFEST = ".agenomic_manifest.json"


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    """Replace ``path`` with ``data`` so a reader (or the next sync after a crash or a full
    disk) sees the old content or the new one, never a truncated file.

    Example:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "m.json"
        >>> _write_atomic(p, b"{}", 0o644)
        >>> p.read_bytes()
        b'{}'
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if sys.platform != "win32":
        # Make the rename itself durable; a directory that cannot be opened only loses that.
        with contextlib.suppress(OSError):
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


def _managed_files(root: Path) -> set[str]:
    """Every regular file under ``root`` (symbolic links and the manifest excluded), relative
    and with ``/`` separators on every platform, so manifest entries compare equal on Windows."""
    found: set[str] = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = Path(dirpath) / name
            if name == _MANIFEST or full.is_symlink() or not full.is_file():
                continue
            found.add(full.relative_to(root).as_posix())
    return found


def _previous_files(root: Path, manifest_path: Path) -> set[str]:
    """Files the previous sync wrote. Without a manifest (first sync) nothing; with an
    unreadable or malformed one, every regular file in the directory, so a skill that is no
    longer approved is removed rather than kept forever."""
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        logger.error(
            "skills manifest unreadable (%s); reconciling the whole directory", type(exc).__name__
        )
        return _managed_files(root)
    try:
        files = json.loads(raw).get("files")
        if isinstance(files, list) and all(isinstance(f, str) for f in files):
            return set(files)
    except (ValueError, AttributeError):
        pass
    logger.error("skills manifest malformed; reconciling the whole directory")
    return _managed_files(root)


def sync_skills(skills: Sequence[Mapping[str, JsonValue]], skills_dir: Path) -> dict[str, int]:
    """Write approved skills into ``skills_dir``; remove files a previous sync wrote and
    that are no longer approved. Targets escaping the directory or failing their digest
    are skipped. Files and the manifest are replaced atomically; when the manifest is
    unreadable or malformed, every regular file of the directory that is not approved now
    is removed (the directory belongs to the supervisor) and the fact is logged as an error.

    Example:
        >>> import tempfile, hashlib
        >>> d = Path(tempfile.mkdtemp())
        >>> body = "# skill"
        >>> sha = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
        >>> sync_skills([{"target": "skills/demo/SKILL.md", "digest": sha, "content": body}], d)
        {'written': 1, 'unchanged': 0, 'removed': 0, 'rejected': 0}
    """
    skills_dir.mkdir(parents=True, exist_ok=True)
    root = skills_dir.resolve()
    manifest_path = root / _MANIFEST
    previous = _previous_files(root, manifest_path)
    counts = {"written": 0, "unchanged": 0, "removed": 0, "rejected": 0}
    current: set[str] = set()
    writes: list[tuple[Path, bytes]] = []
    for skill in skills:
        target = str(skill.get("target") or "")
        content = skill.get("content")
        digest = str(skill.get("digest") or "")
        rel = target[len("skills/") :] if target.startswith("skills/") else target
        dest = (root / rel).resolve()
        if (
            not rel
            or os.path.isabs(rel)
            or not isinstance(content, str)
            or not str(dest).startswith(str(root) + os.sep)
            or dest == manifest_path
        ):
            counts["rejected"] += 1
            continue
        data = content.encode("utf-8")
        if not _digest_matches(data, digest):
            logger.warning("approved skill %s skipped: digest mismatch", rel)
            counts["rejected"] += 1
            continue
        relative = dest.relative_to(root).as_posix()
        current.add(relative)
        if dest.exists() and dest.read_bytes() == data:
            if sys.platform != "win32" and stat.S_IMODE(dest.stat().st_mode) != 0o644:
                os.chmod(dest, 0o644)  # a wider mode set since the last sync is narrowed
            counts["unchanged"] += 1
            continue
        writes.append((dest, data))
    if any(dest.relative_to(root).as_posix() not in previous for dest, _ in writes):
        # Record the new files before writing them: a sync interrupted after a write still
        # leaves a manifest that names it, so a later sync removes it once unapproved.
        _write_atomic(
            manifest_path, json.dumps({"files": sorted(previous | current)}).encode("utf-8"), 0o644
        )
    for dest, data in writes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(dest, data, 0o644)
        counts["written"] += 1
    for stale in sorted(previous - current):
        path = (root / stale).resolve()
        if str(path).startswith(str(root) + os.sep) and path.is_file():
            path.unlink()
            counts["removed"] += 1
    _write_atomic(manifest_path, json.dumps({"files": sorted(current)}).encode("utf-8"), 0o644)
    return counts


class Supervisor:
    """Runs one Hermes child under Agenomic control.

    Example:
        >>> sup = _demo_supervisor()
        >>> sup.state, sup.refuse_restart, "AGENOMIC_HERMES_SUPERVISOR_TOKEN" in sup.child_env
        ('stopped', False, False)
    """

    def __init__(
        self,
        settings: SupervisorSettings,
        client: _SupervisorApi,
        *,
        environ: Optional[Mapping[str, str]] = None,
        connect: _Connect = socket.create_connection,
    ) -> None:
        self.settings = settings
        self.client = client
        self._environ = dict(os.environ if environ is None else environ)
        self._connect = connect
        self.child_env = build_child_env(
            self._environ, allow=settings.env_allow, runtime_token_env=settings.runtime_token_env
        )
        # The attested home and the home Hermes loads must be the same directory.
        self.child_env["HERMES_HOME"] = str(settings.hermes_home)
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.state = "stopped"
        self.exit_code: Optional[int] = None
        self.restarts = 0
        self.refuse_restart = False
        self._stopping = threading.Event()
        self._seen_commands: set[str] = set()
        # Acknowledgements that failed in transport, retried on every tick until accepted.
        self._ack_retries: deque[tuple[str, str, dict[str, JsonValue]]] = deque(maxlen=500)
        self._ticks = 0
        self._launch_failed = False
        self.gave_up = False

    # -- child process -----------------------------------------------
    def start_child(self) -> bool:
        """Start Hermes unless restarts are refused.

        Example:
            >>> sup = _demo_supervisor(["sleep", "5"])
            >>> sup.start_child(), sup.state
            (True, 'running')
            >>> sup.stop_child()
            -15
        """
        if self.refuse_restart or not self.settings.argv:
            return False
        self.state = "starting"
        kwargs: dict[str, Any] = {"env": self.child_env, "start_new_session": True}
        if self.settings.child_uid is not None:
            kwargs["user"] = self.settings.child_uid
        if self.settings.child_gid is not None:
            kwargs["group"] = self.settings.child_gid
        if self.settings.child_uid is not None or self.settings.child_gid is not None:
            # Popen keeps the supervisor's supplementary groups unless told otherwise; the
            # isolation report models the child with its primary group only.
            kwargs["extra_groups"] = []
        try:
            self.proc = subprocess.Popen(self.settings.argv, **kwargs)
        except OSError as e:
            logger.error("Hermes did not start: %s", type(e).__name__)
            self.proc = None
            self.state = "exited"
            self.exit_code = None
            self._launch_failed = True
            return False
        self._launch_failed = False
        self.state = "running"
        self.exit_code = None
        logger.info("Hermes started (pid %d)", self.proc.pid)
        return True

    def poll(self) -> None:
        """Refresh the child state; restart on failure when allowed.

        Example:
            >>> sup = _demo_supervisor()
            >>> sup.poll()
            >>> sup.state
            'stopped'
        """
        if self.proc is None:
            if self._launch_failed and not self._stopping.is_set():
                self._restart_or_give_up("launch failed")
            return
        code = self.proc.poll()
        if code is None:
            self.state = "running"
            return
        if self.state == "running":
            self.state = "exited"
            self.exit_code = code
            logger.warning("Hermes exited with code %d", code)
            if code != 0 and not self._stopping.is_set():
                self._restart_or_give_up(f"exit code {code}")

    def _restart_or_give_up(self, why: str) -> None:
        if not self.settings.restart or self.refuse_restart:
            return
        if self.restarts >= self.settings.max_restarts:
            if not self.gave_up:
                logger.error("Hermes not restarted after %d attempts (%s)", self.restarts, why)
            self.gave_up = True
            return
        self.restarts += 1
        time.sleep(min(60.0, 2.0**self.restarts))
        self.start_child()

    def stop_child(self) -> Optional[int]:
        """SIGTERM the child's process group, SIGKILL after ``grace_s``. Returns the exit code.

        Example:
            >>> sup = _demo_supervisor(["sleep", "5"])
            >>> sup.start_child()
            True
            >>> sup.stop_child(), sup.state
            (-15, 'stopped')
        """
        proc = self.proc
        if proc is None:
            self.state = "stopped"
            return self.exit_code
        if proc.poll() is None:
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(self.settings.grace_s)
            except subprocess.TimeoutExpired:
                self._signal(proc, _KILL_SIGNAL)
                proc.wait()
        self.exit_code = proc.returncode
        self.state = "stopped"
        return self.exit_code

    @staticmethod
    def _signal(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        if sys.platform == "win32":
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(sig)
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(sig)

    # -- control plane -----------------------------------------------
    def _gids(self) -> list[int]:
        if self.settings.child_gid is not None:
            return [self.settings.child_gid]
        if sys.platform == "win32":
            return []
        if self.settings.child_uid is not None:
            return [os.getgid()]
        return list({os.getgid(), *os.getgroups()})

    def isolation(self) -> dict[str, JsonValue]:
        """Self check of the child's isolation.

        Example:
            >>> iso = _demo_supervisor().isolation()
            >>> iso["provider_secrets_absent"], iso["egress_restricted"]
            (True, False)
        """
        home = self.settings.hermes_home
        config_paths = self.settings.config_paths or [home / "config.yaml", home / ".env"]
        skills_paths = [
            p for p in (self.settings.skills_dir, home / "skills", home / "plugins") if p
        ]
        skills_paths += self.settings.readonly_paths
        return isolation_report(
            self.child_env,
            child_uid=_current_uid()
            if self.settings.child_uid is None
            else self.settings.child_uid,
            child_gids=self._gids(),
            config_paths=config_paths,
            skills_paths=skills_paths,
            forbidden_hosts=self.settings.forbidden_hosts,
            runtime_token_env=self.settings.runtime_token_env,
            connect=self._connect,
        )

    def heartbeat(self, *, execute: bool = True) -> None:
        """Report process state and isolation; execute returned commands unless ``execute``
        is false (the final report after supervision ended: commands are neither executed
        nor acknowledged, so the gateway delivers them again to the next supervisor).

        Example:
            >>> sup = _demo_supervisor(commands=[{"id": "c1", "kind": "quarantine"}])
            >>> sup.heartbeat(execute=False)
            >>> sup.refuse_restart
            False
            >>> sup.heartbeat()
            >>> sup.refuse_restart
            True
        """
        body: dict[str, JsonValue] = {
            "process": {
                "state": self.state,
                "pid": self.proc.pid if self.proc is not None and self.state == "running" else None,
                "exit_code": self.exit_code,
                "restarts": self.restarts,
            },
            "isolation": self.isolation(),
        }
        try:
            resp = self.client.heartbeat(body)
        except HermesApiError as e:
            logger.warning("supervisor heartbeat failed (%s)", e.code)
            return
        commands = resp.get("commands")
        if execute and isinstance(commands, list):
            for command in commands:
                if isinstance(command, dict):
                    self.handle_command(command)

    def _ack(self, command_id: str, status: str, detail: dict[str, JsonValue]) -> None:
        try:
            self.client.ack_command(command_id, status, detail)
        except HermesApiError as e:
            logger.warning("supervisor ack %s failed (%s)", status, e.code)
            if e.status == 0 or e.status >= 500:
                self._ack_retries.append((command_id, status, detail))
            return
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
        """``quarantine``/``revoke`` stop and refuse restarts; ``resume`` allows them.

        Example:
            >>> sup = _demo_supervisor()
            >>> sup.handle_command({"id": "c1", "kind": "revoke"})
            >>> sup.refuse_restart, sup.state
            (True, 'stopped')
        """
        command_id = str(command.get("id") or "")
        if not command_id or command_id in self._seen_commands:
            return
        self._seen_commands.add(command_id)
        kind = str(command.get("kind") or "")
        if str(command.get("status") or "requested") == "requested":
            self._ack(command_id, "received", {"executor": "supervisor"})
        if kind in ("quarantine", "revoke"):
            self.refuse_restart = True
            code = self.stop_child()
            self._ack(command_id, "applied", {"process_state": self.state, "exit_code": code})
        elif kind == "resume":
            self.refuse_restart = False
            restarted = False
            if self.proc is None or self.proc.poll() is not None:
                restarted = self.start_child()
            self._ack(command_id, "applied", {"restarted": restarted, "process_state": self.state})
        else:
            self._ack(command_id, "refused", {"reason": "unsupported_command", "kind": kind})

    def sync_skills(self) -> Optional[dict[str, int]]:
        """Pull approved skills into the supervisor owned skills directory.

        Example:
            >>> import tempfile
            >>> sup = _demo_supervisor()
            >>> sup.settings.skills_dir = Path(tempfile.mkdtemp())
            >>> sup.sync_skills()
            {'written': 0, 'unchanged': 0, 'removed': 0, 'rejected': 0}
        """
        if self.settings.skills_dir is None:
            return None
        try:
            resp = self.client.approved_skills()
        except HermesApiError as e:
            logger.warning("approved skills not fetched (%s)", e.code)
            return None
        skills = resp.get("skills")
        if not isinstance(skills, list):
            return None
        return sync_skills([s for s in skills if isinstance(s, dict)], self.settings.skills_dir)

    def tick(self) -> None:
        """One supervision step.

        Example:
            >>> sup = _demo_supervisor(commands=[{"id": "c1", "kind": "quarantine"}])
            >>> sup.tick()
            >>> sup.refuse_restart
            True
        """
        self.poll()
        if self._ticks % max(1, self.settings.skills_every) == 0:
            self.sync_skills()
        self._ticks += 1
        self.heartbeat()
        self._retry_acks()

    def request_stop(self, *_: object) -> None:
        """Signal handler: stop at the next loop iteration.

        Example:
            >>> sup = _demo_supervisor()
            >>> sup.request_stop(15, None)
            >>> sup._stopping.is_set()
            True
        """
        self._stopping.set()

    def run(self) -> int:
        """Start Hermes and supervise until SIGTERM/SIGINT, or until the child ended when
        restarts are disabled. Returns the child's exit code, or 1 when supervision itself
        failed; the child is stopped whenever the supervisor leaves this loop.

        Example:
            >>> _demo_supervisor(["true"]).run()  # doctest: +SKIP
            0
        """
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.sync_skills()
        self.start_child()
        failed = False
        try:
            while not self._stopping.is_set():
                self.tick()
                if not self.settings.restart and self.state in ("exited", "stopped"):
                    break
                if self.gave_up:
                    break
                self._stopping.wait(self.settings.interval_s)
        except Exception:
            # The child runs in its own session, so nothing else would stop it: a supervisor
            # that cannot supervise stops Hermes and exits with a failure.
            logger.exception("supervision failed; stopping Hermes")
            failed = True
        finally:
            code = self.stop_child()
        try:
            # Report only: a command executed now (a ``resume`` would start Hermes again)
            # would act on a process nothing supervises once ``run`` returns.
            self.heartbeat(execute=False)
        except Exception:
            logger.exception("final supervisor heartbeat failed")
            failed = True
        if failed:
            return 1
        if code is None:
            return 1 if self.gave_up or self._launch_failed else 0
        return code


def _demo_supervisor(
    argv: Sequence[str] = ("true",), *, commands: Sequence[Mapping[str, JsonValue]] = ()
) -> Supervisor:
    """Offline supervisor for the examples: a fake API, a temporary home, no egress probe."""
    import tempfile

    import httpx

    answer: dict[str, JsonValue] = {"commands": [dict(c) for c in commands], "skills": []}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer))
    settings = SupervisorSettings(
        argv=list(argv),
        hermes_home=Path(tempfile.mkdtemp()),
        forbidden_hosts=[],
        restart=False,
    )
    client = SupervisorClient("https://a.example", "agmhs_x", transport=transport)
    return Supervisor(settings, client, environ={"PATH": os.environ.get("PATH", "")})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agenomic-hermes-supervisor",
        description="Start Hermes under Agenomic supervision: agenomic-hermes-supervisor [options] -- hermes ...",
    )
    p.add_argument("--endpoint", default=None, help=f"Agenomic base URL (default ${ENDPOINT_ENV})")
    p.add_argument(
        "--hermes-home",
        default=None,
        help="HERMES_HOME of the child (default $HERMES_HOME or ~/.hermes)",
    )
    p.add_argument(
        "--skills-dir", default=None, help="supervisor owned directory for approved skills"
    )
    p.add_argument(
        "--config-path",
        action="append",
        default=[],
        help="path that must be read only for the child",
    )
    p.add_argument(
        "--readonly-path", action="append", default=[], help="extra skills/plugins path to attest"
    )
    p.add_argument(
        "--forbidden-host", action="append", default=None, help="host:port the child must not reach"
    )
    p.add_argument(
        "--allow-env", action="append", default=[], help="extra environment variable for the child"
    )
    p.add_argument("--child-uid", type=int, default=None)
    p.add_argument("--child-gid", type=int, default=None)
    p.add_argument("--grace-s", type=float, default=10.0)
    p.add_argument("--interval-s", type=float, default=15.0)
    p.add_argument("--no-restart", action="store_true")
    p.add_argument("--runtime-token-env", default=DEFAULT_TOKEN_ENV)
    return p


def _settings_from_args(
    args: argparse.Namespace, child: list[str], home: Path
) -> SupervisorSettings:
    return SupervisorSettings(
        argv=child,
        hermes_home=home,
        skills_dir=Path(args.skills_dir).expanduser() if args.skills_dir else None,
        config_paths=[Path(p).expanduser() for p in args.config_path],
        readonly_paths=[Path(p).expanduser() for p in args.readonly_path],
        forbidden_hosts=list(DEFAULT_FORBIDDEN_HOSTS)
        if args.forbidden_host is None
        else args.forbidden_host,
        env_allow=args.allow_env,
        child_uid=args.child_uid,
        child_gid=args.child_gid,
        grace_s=args.grace_s,
        interval_s=args.interval_s,
        restart=not args.no_restart,
        runtime_token_env=args.runtime_token_env,
    )


def main(
    argv: Optional[Sequence[str]] = None, *, environ: Optional[Mapping[str, str]] = None
) -> int:
    """Parse ``[options] -- hermes ...`` and run the supervisor.

    Example:
        >>> main(["--endpoint", "https://a.example"], environ={})  # no command after --
        2
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    child: list[str] = []
    if "--" in args_list:
        idx = args_list.index("--")
        child = args_list[idx + 1 :]
        args_list = args_list[:idx]
    args = _parser().parse_args(args_list)
    if sys.platform == "win32":
        logger.error("agenomic-hermes-supervisor requires a POSIX host")
        return 2
    env = dict(os.environ if environ is None else environ)
    if not child:
        logger.error("no Hermes command given after --")
        return 2
    endpoint = args.endpoint or env.get(ENDPOINT_ENV)
    token = env.get(SUPERVISOR_TOKEN_ENV)
    if not endpoint:
        logger.error("endpoint missing: pass --endpoint or set %s", ENDPOINT_ENV)
        return 2
    if not token:
        logger.error("supervisor credential missing: set %s", SUPERVISOR_TOKEN_ENV)
        return 2
    home = Path(
        args.hermes_home or env.get("HERMES_HOME") or str(Path.home() / ".hermes")
    ).expanduser()
    try:
        settings = _settings_from_args(args, child, home)
    except ValidationError as exc:
        logger.error("invalid supervisor settings: %s", exc.errors(include_url=False))
        return 2
    problem = runtime_token_problem(env, settings.runtime_token_env)
    if problem is not None:
        logger.error("refusing to start Hermes: %s", problem)
        return 2
    client = SupervisorClient(endpoint, token)
    try:
        return Supervisor(settings, client, environ=env).run()
    finally:
        client.close()


def cli() -> None:
    """Console script entry point.

    Example:
        >>> cli()  # doctest: +SKIP
        Traceback (most recent call last):
        SystemExit: 0
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    raise SystemExit(main())


__all__ = [
    "Supervisor",
    "SupervisorSettings",
    "build_child_env",
    "egress_restricted",
    "isolation_report",
    "provider_secrets_absent",
    "runtime_token_env_problem",
    "runtime_token_problem",
    "sync_skills",
    "writable_by",
]
