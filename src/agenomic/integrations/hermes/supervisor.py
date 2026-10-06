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
import errno
import hashlib
import json
import logging
import os
import posixpath
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from agenomic.integrations.hermes.client import HermesApiError, SupervisorClient
from agenomic.integrations.hermes.config import (
    _PROVIDER_KEYS,
    _SECRET_NAME,
    DEFAULT_TOKEN_ENV,
    RUNTIME_TOKEN_PREFIX,
    SUPERVISOR_TOKEN_ENV,
    SUPERVISOR_TOKEN_PREFIX,
    runtime_token_env_problem,
)
from agenomic.integrations.hermes.exporter import now_iso

logger = logging.getLogger("agenomic.integrations.hermes.supervisor")

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


def _replaceable(entry: Path, uid: int, groups: set[int]) -> bool:
    """Whether ``uid`` could rename or remove the existing ``entry`` from its directory:
    the directory is writable, not on a read only mount, and its sticky bit (only the
    owner of the entry or of the directory may then remove or rename it) does not apply.
    The root directory has no parent and is never replaceable."""
    parent = entry.parent
    if parent == entry or _readonly_fs(parent):
        return False
    if uid == 0:
        return True
    pst = parent.stat()
    if not _mode_allows_write(pst, uid, groups):
        return False
    sticky_protected = (
        bool(pst.st_mode & stat.S_ISVTX) and entry.lstat().st_uid != uid and pst.st_uid != uid
    )
    return not sticky_protected


def writable_by(path: Path, uid: int, gids: Iterable[int]) -> bool:
    """Whether ``uid`` (with ``gids``) could modify ``path`` or replace it.

    Mode bits and ownership are checked for the file, then for every directory on its
    path: a writable directory lets the entry below it be renamed or removed, and with
    it the whole subtree that holds ``path``, which can then be recreated. POSIX sticky
    bit semantics apply: in a sticky directory only the owner of the entry or of the
    directory may remove or rename it. A missing path is writable when its nearest
    existing ancestor is (the child can create the missing directories). A read only
    mount wins. Root can write anything that is not on a read only mount. A path
    through symbolic links is checked twice: as written (each link can be replaced in
    its directory) and resolved (its target, and every directory above the target).

    Example:
        >>> import tempfile
        >>> d = Path(tempfile.mkdtemp())
        >>> writable_by(d, os.getuid(), [os.getgid()])
        True
    """
    groups = set(gids)
    written = Path(os.path.abspath(path))  # every ancestor, links kept
    if _writable_chain(written, uid, groups):
        return True
    resolved = Path(os.path.realpath(written))
    return resolved != written and _writable_chain(resolved, uid, groups)


def _writable_chain(path: Path, uid: int, groups: set[int]) -> bool:
    """:func:`writable_by` for one absolute path, walked lexically up to the root."""
    entry = path
    if path.exists():
        if not _readonly_fs(path) and (uid == 0 or _mode_allows_write(path.stat(), uid, groups)):
            return True
    else:
        entry = path.parent
        while not entry.exists():
            if entry.parent == entry:
                return False
            entry = entry.parent
        if not _readonly_fs(entry) and (uid == 0 or _mode_allows_write(entry.stat(), uid, groups)):
            return True
    # The entry itself, then every ancestor directory up to the root.
    while entry.parent != entry:
        if _replaceable(entry, uid, groups):
            return True
        entry = entry.parent
    return False


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
#: Terminal results kept for redelivered commands whose acknowledgement was dropped.
_MAX_TERMINAL_ACKS = 10_000


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
#: Opens a directory without following a final symbolic link (POSIX).
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
#: ``open`` errors meaning a symbolic link (``ELOOP``, ``EMLINK`` on some BSDs) or not a directory.
_LINK_ERRNOS = (errno.ELOOP, errno.EMLINK, errno.ENOTDIR)


class UnsafeSkillsPathError(OSError):
    """A skills path the supervisor refuses to write through: the directory is a symbolic
    link or belongs to another user, or the destination is a symbolic link."""


def skills_dir_problem(path: Path) -> Optional[str]:
    """Why the supervisor must not write into ``path``, or ``None``.

    On POSIX the directory must not be a symbolic link and must belong to the
    supervisor's effective uid, and its ancestors must pass :func:`_open_trusted_dir`;
    Windows has no such checks. A missing directory has no problem (the supervisor
    creates it).

    Example:
        >>> import tempfile
        >>> skills_dir_problem(Path(tempfile.mkdtemp())) is None
        True
    """
    if sys.platform != "win32":
        problem = _ancestors_problem(path)
        if problem is not None:
            return problem
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        return "is a symbolic link"
    if not stat.S_ISDIR(st.st_mode):
        return "is not a directory"
    if sys.platform != "win32" and st.st_uid != os.geteuid():
        return "belongs to another user"
    return None


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    """Replace ``path`` with ``data`` so a reader (or the next sync after a crash or a full
    disk) sees the old content or the new one, never a truncated file.

    The temporary file has an unpredictable name and is created exclusively, never
    through a symbolic link, with mode ``0600`` until it is complete. On POSIX every
    step is relative to the parent directory, opened without following a link and
    required to belong to the supervisor; a destination that is a symbolic link is
    refused (:class:`UnsafeSkillsPathError`) rather than replaced or followed.

    Example:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "m.json"
        >>> _write_atomic(p, b"{}", 0o644)
        >>> p.read_bytes()
        b'{}'
    """
    if sys.platform == "win32":
        _write_atomic_portable(path, data, mode)
        return
    try:
        dir_fd = os.open(path.parent, _DIR_FLAGS)
    except OSError as e:
        if e.errno in _LINK_ERRNOS and path.parent.is_symlink():
            raise UnsafeSkillsPathError(f"{path.parent} is a symbolic link") from e
        raise
    try:
        _write_atomic_at(dir_fd, path.name, data, mode)
    finally:
        os.close(dir_fd)


def _write_atomic_at(dir_fd: int, name: str, data: bytes, mode: int) -> None:  # pragma: posix-only
    """:func:`_write_atomic` for the entry ``name`` of the directory open as ``dir_fd``
    (POSIX): the directory must belong to the supervisor, ``name`` must not be a link."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    if os.fstat(dir_fd).st_uid != os.geteuid():
        raise UnsafeSkillsPathError(f"the directory of {name} belongs to another user")
    try:
        if stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode):
            raise UnsafeSkillsPathError(f"{name} is a symbolic link")
    except FileNotFoundError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    while True:
        tmp = f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(tmp, flags, 0o600, dir_fd=dir_fd)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fchmod(fh.fileno(), mode)
            os.fsync(fh.fileno())
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=dir_fd)
        raise
    # Make the rename itself durable; a directory that cannot be synced only loses that.
    with contextlib.suppress(OSError):
        os.fsync(dir_fd)


def _write_atomic_portable(path: Path, data: bytes, mode: int) -> None:
    """:func:`_write_atomic` without directory descriptors (Windows)."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(name, mode)
        if path.is_symlink():
            raise UnsafeSkillsPathError(f"{path} is a symbolic link")
        os.replace(name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise


def _skill_parts(rel: str) -> Optional[tuple[str, ...]]:
    """The components of a skill path relative to the skills directory, normalized
    lexically, or ``None`` when it is empty, absolute or leaves the directory.

    Example:
        >>> _skill_parts("demo/SKILL.md"), _skill_parts("a/../b.md"), _skill_parts("../x")
        (('demo', 'SKILL.md'), ('b.md',), None)
    """
    if not rel or os.path.isabs(rel) or rel.startswith("/"):
        return None
    norm = posixpath.normpath(rel.replace(os.sep, "/"))
    if norm in (".", "..") or norm.startswith(("../", "/")):
        return None
    return tuple(norm.split("/"))


#: Symbolic links followed while opening the skills directory's ancestors (trusted ones only).
_MAX_TRUSTED_LINKS = 40


def _ancestor_problem(st: os.stat_result) -> Optional[str]:  # pragma: posix-only
    """Why an ancestor directory with status ``st`` is not trusted, or ``None``."""
    if sys.platform == "win32":
        raise NotImplementedError("ownership checks are POSIX only")
    if st.st_uid not in (os.geteuid(), 0):
        return "belongs to another user"
    if st.st_mode & 0o022 and not st.st_mode & stat.S_ISVTX:
        return "is writable by others without the sticky bit"
    return None


def _trusted_link(root_fd: int, name: str) -> Optional[str]:  # pragma: posix-only
    """The target of the link ``name`` directly under ``/`` (open as ``root_fd``) when
    root owns it and ``/`` (system layout links such as ``/var`` -> ``private/var`` on
    macOS), else ``None``."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    try:
        link = os.lstat(name, dir_fd=root_fd)
    except OSError:
        return None
    parent = os.fstat(root_fd)
    if (
        not stat.S_ISLNK(link.st_mode)
        or link.st_uid != 0
        or parent.st_uid != 0
        or parent.st_mode & 0o022
    ):
        return None
    return os.readlink(name, dir_fd=root_fd)


def _open_trusted_dir(path: Path, *, create: bool) -> Optional[int]:  # pragma: posix-only
    """A descriptor of the absolute directory ``path``, opened from ``/`` one component at
    a time with ``O_NOFOLLOW`` (POSIX); missing components are created (mode ``0755``)
    when ``create`` is true, else ``None`` is returned.

    Every directory on the way must belong to root or the supervisor's euid and, when
    group or other may write it, carry the sticky bit (like ``/tmp``), so nobody else can
    rename or replace a component. A symbolic link is followed only directly under
    ``/``, owned by root, with ``/`` owned by root and writable by nobody else (system
    layout links such as macOS ``/var`` and ``/tmp``); any other link, a ``..``
    component, or a component that is not a directory raises
    :class:`UnsafeSkillsPathError`.
    """
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    if not path.is_absolute():
        raise UnsafeSkillsPathError("is not an absolute path")
    pending = list(path.parts[1:])
    links = 0
    at_root = True
    fd = os.open("/", _DIR_FLAGS)
    try:
        while pending:
            part = pending.pop(0)
            if part in ("", "."):
                continue
            if part == "..":
                raise UnsafeSkillsPathError("has a '..' component")
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    os.close(fd)
                    return None
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o755, dir_fd=fd)
                pending.insert(0, part)
                continue
            except OSError as e:
                if e.errno not in _LINK_ERRNOS:
                    raise
                target = _trusted_link(fd, part) if at_root else None
                links += 1
                if target is None or links > _MAX_TRUSTED_LINKS:
                    raise UnsafeSkillsPathError(
                        f"has an ancestor {part} that is a symbolic link or not a directory"
                    ) from e
                if target.startswith("/"):
                    os.close(fd)
                    fd = os.open("/", _DIR_FLAGS)
                    at_root = True
                pending[:0] = [p for p in target.split("/") if p]
                continue
            os.close(fd)
            fd = child
            at_root = False
            problem = _ancestor_problem(os.fstat(fd))
            if problem is not None:
                raise UnsafeSkillsPathError(f"has an ancestor {part} that {problem}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _ancestors_problem(skills_dir: Path) -> Optional[str]:  # pragma: posix-only
    """Why the ancestors of ``skills_dir`` are not trusted (:func:`_open_trusted_dir`,
    nothing created), or ``None``."""
    if ".." in skills_dir.parts:
        return "has a '..' component"
    try:
        fd = _open_trusted_dir(Path(os.path.abspath(skills_dir)).parent, create=False)
    except UnsafeSkillsPathError as e:
        return str(e)
    if fd is not None:
        os.close(fd)
    return None


def _open_skills_root(skills_dir: Path) -> int:  # pragma: posix-only
    """A descriptor of ``skills_dir`` created or opened without following a link (POSIX).

    The parent directory is opened by :func:`_open_trusted_dir` (from ``/``, component
    by component, no untrusted link, every ancestor owned by the supervisor's euid or
    root and not writable by others unless sticky); the directory itself is created
    relative to it, then opened with ``O_NOFOLLOW`` and must belong to the supervisor.
    A link or a directory of another user swapped in between the checks is refused
    (:class:`UnsafeSkillsPathError`), never followed.
    """
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    if ".." in skills_dir.parts:
        raise UnsafeSkillsPathError("has a '..' component")
    path = Path(os.path.abspath(skills_dir))
    name = path.name
    if not name:
        raise UnsafeSkillsPathError("is a filesystem root")
    parent_fd = _open_trusted_dir(path.parent, create=True)
    assert parent_fd is not None
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, 0o755, dir_fd=parent_fd)
        try:
            fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        except OSError as e:
            if e.errno in _LINK_ERRNOS:
                raise UnsafeSkillsPathError("is a symbolic link or not a directory") from e
            raise
    finally:
        os.close(parent_fd)
    try:
        if os.fstat(fd).st_uid != os.geteuid():
            raise UnsafeSkillsPathError("belongs to another user")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_dir_at(
    root_fd: int, parts: Sequence[str], *, create: bool
) -> Optional[int]:  # pragma: posix-only
    """A descriptor of the subdirectory ``parts`` of ``root_fd``, every component opened
    with ``O_NOFOLLOW`` and required to belong to the supervisor (POSIX). ``None`` when a
    component is missing and ``create`` is false."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    fd = os.dup(root_fd)
    try:
        for i, part in enumerate(parts):
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o755, dir_fd=fd)
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if create:
                    raise
                os.close(fd)
                return None
            except OSError as e:
                if e.errno in _LINK_ERRNOS:
                    where = "/".join(parts[: i + 1])
                    raise UnsafeSkillsPathError(
                        f"{where} is a symbolic link or not a directory"
                    ) from e
                raise
            os.close(fd)
            fd = child
            if os.fstat(fd).st_uid != os.geteuid():
                raise UnsafeSkillsPathError(f"{'/'.join(parts[: i + 1])} belongs to another user")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unchanged_at(root_fd: int, parts: Sequence[str], data: bytes) -> bool:  # pragma: posix-only
    """Whether the skill file ``parts`` already holds ``data`` (its mode is then narrowed
    to ``0644``). A link or a non regular file on the way raises
    :class:`UnsafeSkillsPathError`."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    dir_fd = _open_dir_at(root_fd, parts[:-1], create=False)
    if dir_fd is None:
        return False
    try:
        try:
            st = os.lstat(parts[-1], dir_fd=dir_fd)
        except FileNotFoundError:
            return False
        mode = st.st_mode
        if stat.S_ISLNK(mode):
            raise UnsafeSkillsPathError(f"{'/'.join(parts)} is a symbolic link")
        if not stat.S_ISREG(mode):
            raise UnsafeSkillsPathError(f"{'/'.join(parts)} is not a regular file")
        if st.st_uid != _supervisor_uid():
            return False  # replaced by someone else: rewritten, so the supervisor owns it
        fd = _open_regular_at(parts[-1], dir_fd)
        with os.fdopen(fd, "rb") as fh:
            if fh.read(len(data) + 1) != data:
                return False
            if stat.S_IMODE(os.fstat(fh.fileno()).st_mode) != 0o644:
                os.fchmod(fh.fileno(), 0o644)  # a wider mode set since the last sync
        return True
    finally:
        os.close(dir_fd)


#: A manifest larger than this is treated as malformed rather than read into memory.
_MANIFEST_MAX_BYTES = 16 * 1024 * 1024


def _supervisor_uid() -> int:  # pragma: posix-only
    """The effective uid that owns every file this supervisor writes."""
    if sys.platform == "win32":
        raise NotImplementedError("file ownership is POSIX only")
    return os.geteuid()


def _open_regular_at(name: str, dir_fd: int) -> int:  # pragma: posix-only
    """Open ``name`` read only in ``dir_fd``, never through a symbolic link and never
    blocking: a FIFO or device planted there is opened nonblocking, then refused with
    :class:`UnsafeSkillsPathError` because ``fstat`` does not report a regular file. A
    file the supervisor does not own (planted or replaced by the child, which can write
    the directory) is refused the same way: only what the supervisor wrote is trusted."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafeSkillsPathError(f"{name} is not a regular file")
        if st.st_uid != _supervisor_uid():
            raise UnsafeSkillsPathError(f"{name} is not owned by the supervisor")
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _managed_files(root: Path) -> set[str]:
    """Every regular file and every symbolic link (to a file or a directory, never followed)
    under ``root``, the manifest excluded, relative and with ``/`` separators on every
    platform, so manifest entries compare equal on Windows. Links are listed so a full
    reconciliation removes them, or keeps the sync failed, instead of forgetting them."""
    found: set[str] = set()
    for dirpath, dirs, files in os.walk(root):
        for name in dirs:
            full = Path(dirpath) / name
            if full.is_symlink():
                found.add(full.relative_to(root).as_posix())
        for name in files:
            full = Path(dirpath) / name
            if name == _MANIFEST or not (full.is_symlink() or full.is_file()):
                continue
            found.add(full.relative_to(root).as_posix())
    return found


def _managed_files_at(root_fd: int) -> set[str]:  # pragma: posix-only
    """:func:`_managed_files` through the descriptor of the skills directory (POSIX);
    symbolic links, to files or directories, are never followed."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    found: set[str] = set()
    for dirpath, dirs, files, dfd in os.fwalk(".", dir_fd=root_fd):
        for name in [*dirs, *files]:
            if name == _MANIFEST and name in files:
                continue
            try:
                mode = os.lstat(name, dir_fd=dfd).st_mode
            except OSError:
                continue
            if stat.S_ISLNK(mode) or (stat.S_ISREG(mode) and name in files):
                found.add(posixpath.normpath(posixpath.join(dirpath, name)))
    return found


def _manifest_files(raw: str) -> Optional[set[str]]:
    try:
        files = json.loads(raw).get("files")
        if isinstance(files, list) and all(isinstance(f, str) for f in files):
            return set(files)
    except (ValueError, AttributeError):
        pass
    return None


def _previous_files(root: Path, manifest_path: Path) -> set[str]:
    """Files the previous sync wrote. Without a manifest (first sync, or a manifest the agent
    deleted) or with an unreadable or malformed one, every regular file in the directory, so
    a skill that is no longer approved is removed rather than kept forever (on a genuine
    first sync the directory is empty)."""
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _managed_files(root)
    except (OSError, ValueError) as exc:
        logger.error(
            "skills manifest unreadable (%s); reconciling the whole directory", type(exc).__name__
        )
        return _managed_files(root)
    files = _manifest_files(raw)
    if files is None:
        logger.error("skills manifest malformed; reconciling the whole directory")
        return _managed_files(root)
    return files


def _previous_files_at(root_fd: int) -> set[str]:  # pragma: posix-only
    """:func:`_previous_files` through the descriptor of the skills directory (POSIX)."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    try:
        fd = _open_regular_at(_MANIFEST, root_fd)
        with os.fdopen(fd, "rb") as fh:
            data = fh.read(_MANIFEST_MAX_BYTES + 1)
        if len(data) > _MANIFEST_MAX_BYTES:
            raise ValueError("skills manifest too large")
        raw = data.decode("utf-8")
    except FileNotFoundError:
        # Deleted (or first sync): never trusted as "nothing was written before".
        return _managed_files_at(root_fd)
    except (OSError, ValueError) as exc:
        logger.error(
            "skills manifest unreadable (%s); reconciling the whole directory", type(exc).__name__
        )
        return _managed_files_at(root_fd)
    files = _manifest_files(raw)
    if files is None:
        logger.error("skills manifest malformed; reconciling the whole directory")
        return _managed_files_at(root_fd)
    return files


def sync_skills(skills: Sequence[Mapping[str, JsonValue]], skills_dir: Path) -> dict[str, int]:
    """Write approved skills into ``skills_dir``; remove files a previous sync wrote and
    that are no longer approved. Targets escaping the directory or failing their digest
    are skipped. Files and the manifest are replaced atomically; when the manifest is
    unreadable or malformed, every regular file of the directory that is not approved now
    is removed (the directory belongs to the supervisor) and the fact is logged as an error.

    On POSIX the directory is created and opened once without following a link (see
    :func:`_open_skills_root`) and everything else (subdirectories, reads, writes,
    removals, the manifest) goes through that descriptor, never through a path resolved
    again, so a link swapped in by the agent at any point is never followed.

    Example:
        >>> import tempfile, hashlib
        >>> d = Path(tempfile.mkdtemp())
        >>> body = "# skill"
        >>> sha = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
        >>> sync_skills([{"target": "skills/demo/SKILL.md", "digest": sha, "content": body}], d)
        {'written': 1, 'unchanged': 0, 'removed': 0, 'rejected': 0}
    """
    try:
        return _sync_skills_checked(skills, skills_dir)
    except _UnsafeSkillsDirError as e:
        logger.error("skills directory %s %s; approved skills not written", skills_dir, e)
        return {"written": 0, "unchanged": 0, "removed": 0, "rejected": len(skills)}


class _UnsafeSkillsDirError(UnsafeSkillsPathError):
    """The skills directory itself cannot be used: nothing was reconciled."""


def _sync_skills_checked(
    skills: Sequence[Mapping[str, JsonValue]], skills_dir: Path
) -> dict[str, int]:
    """:func:`sync_skills`, raising :class:`_UnsafeSkillsDirError` when the directory itself
    is unsafe instead of reporting every skill rejected (the supervisor then keeps Hermes
    stopped: nothing was reconciled)."""
    counts = {"written": 0, "unchanged": 0, "removed": 0, "rejected": 0}
    problem = skills_dir_problem(skills_dir)
    if problem is not None:
        raise _UnsafeSkillsDirError(problem)
    if sys.platform == "win32":
        return _sync_skills_portable(skills, skills_dir, counts)
    try:
        root_fd = _open_skills_root(skills_dir)
    except UnsafeSkillsPathError as e:
        raise _UnsafeSkillsDirError(str(e)) from e
    try:
        return _sync_skills_at(skills, root_fd, counts)
    finally:
        os.close(root_fd)


def _skill_entry(skill: Mapping[str, JsonValue]) -> tuple[str, Optional[tuple[str, ...]], object]:
    target = str(skill.get("target") or "")
    rel = target[len("skills/") :] if target.startswith("skills/") else target
    return rel, _skill_parts(rel), skill.get("content")


def _sync_skills_at(
    skills: Sequence[Mapping[str, JsonValue]], root_fd: int, counts: dict[str, int]
) -> dict[str, int]:  # pragma: posix-only
    """:func:`sync_skills` relative to the trusted descriptor of the directory (POSIX)."""
    if sys.platform == "win32":
        raise NotImplementedError("directory descriptors are POSIX only")
    previous = _previous_files_at(root_fd)
    current: set[str] = set()
    writes: list[tuple[tuple[str, ...], bytes]] = []
    for skill in skills:
        rel, parts, content = _skill_entry(skill)
        if parts is None or not isinstance(content, str) or parts == (_MANIFEST,):
            counts["rejected"] += 1
            continue
        data = content.encode("utf-8")
        if not _digest_matches(data, str(skill.get("digest") or "")):
            logger.warning("approved skill %s skipped: digest mismatch", rel)
            counts["rejected"] += 1
            continue
        try:
            unchanged = _unchanged_at(root_fd, parts, data)
        except UnsafeSkillsPathError as e:
            # A symbolic link on the way (the file itself or a directory): never followed.
            logger.warning("approved skill %s skipped: %s", rel, e)
            counts["rejected"] += 1
            continue
        current.add("/".join(parts))
        if unchanged:
            counts["unchanged"] += 1
            continue
        writes.append((parts, data))
    if any("/".join(parts) not in previous for parts, _ in writes):
        # Record the new files before writing them: a sync interrupted after a write still
        # leaves a manifest that names it, so a later sync removes it once unapproved.
        _write_atomic_at(
            root_fd,
            _MANIFEST,
            json.dumps({"files": sorted(previous | current)}).encode("utf-8"),
            0o644,
        )
    for parts, data in writes:
        try:
            dir_fd = _open_dir_at(root_fd, parts[:-1], create=True)
            assert dir_fd is not None  # created when missing
            try:
                _write_atomic_at(dir_fd, parts[-1], data, 0o644)
            finally:
                os.close(dir_fd)
        except UnsafeSkillsPathError as e:
            logger.error("approved skill %s not written: %s", "/".join(parts), e)
            counts["rejected"] += 1
            continue
        counts["written"] += 1
    # Stale paths that could not be removed safely stay in the manifest (a later sync keeps
    # trying) and count as rejected, so Hermes is not started on an unreconciled tree.
    unreconciled: set[str] = set()
    for stale in sorted(previous - current):
        stale_parts = _skill_parts(stale)
        if stale_parts is None or stale_parts == (_MANIFEST,):
            continue
        try:
            dir_fd = _open_dir_at(root_fd, stale_parts[:-1], create=False)
        except UnsafeSkillsPathError:
            logger.warning("stale skill %s kept: its path goes through a symbolic link", stale)
            unreconciled.add(stale)
            continue
        if dir_fd is None:
            continue
        try:
            try:
                mode = os.lstat(stale_parts[-1], dir_fd=dir_fd).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
                # unlink relative to the trusted descriptor removes a link itself, never
                # what it points to.
                os.unlink(stale_parts[-1], dir_fd=dir_fd)
                counts["removed"] += 1
            else:
                logger.warning("stale skill %s kept: it is not a regular file", stale)
                unreconciled.add(stale)
        finally:
            os.close(dir_fd)
    counts["rejected"] += len(unreconciled)
    _write_atomic_at(
        root_fd,
        _MANIFEST,
        json.dumps({"files": sorted(current | unreconciled)}).encode("utf-8"),
        0o644,
    )
    return counts


def _sync_skills_portable(
    skills: Sequence[Mapping[str, JsonValue]], skills_dir: Path, counts: dict[str, int]
) -> dict[str, int]:
    """:func:`sync_skills` without directory descriptors (Windows): explicit link checks."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    if skills_dir.is_symlink():
        logger.error(
            "skills directory %s is a symbolic link; approved skills not written", skills_dir
        )
        counts["rejected"] = len(skills)
        return counts
    root = skills_dir.resolve()
    manifest_path = root / _MANIFEST
    previous = _previous_files(root, manifest_path)
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
        if dest != Path(os.path.normpath(root / rel)):
            # A symbolic link on the way (the file itself or a directory): never followed.
            logger.warning("approved skill %s skipped: its path goes through a symbolic link", rel)
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
        try:
            _write_atomic(dest, data, 0o644)
        except UnsafeSkillsPathError as e:
            logger.error("approved skill %s not written: %s", dest.relative_to(root).as_posix(), e)
            counts["rejected"] += 1
            continue
        counts["written"] += 1
    unreconciled: set[str] = set()
    for stale in sorted(previous - current):
        lexical = Path(os.path.normpath(root / stale))
        path = lexical.resolve()
        if path != lexical:
            logger.warning("stale skill %s kept: its path goes through a symbolic link", stale)
            unreconciled.add(stale)
            continue
        if str(path).startswith(str(root) + os.sep) and path.is_file():
            path.unlink()
            counts["removed"] += 1
    counts["rejected"] += len(unreconciled)
    _write_atomic(
        manifest_path,
        json.dumps({"files": sorted(current | unreconciled)}).encode("utf-8"),
        0o644,
    )
    return counts


_GROUP_KILL_WAIT_S = 5.0


def _group_alive(pgid: int) -> bool:  # pragma: posix-only
    """Whether process group ``pgid`` still has a member; zombies we may reap are reaped."""
    if sys.platform == "win32":
        raise NotImplementedError("process groups are POSIX only")
    # A supervisor that is the subreaper (PID 1 in a container) inherits orphaned members:
    # their zombies would keep the group alive until reaped.
    with contextlib.suppress(ChildProcessError, OSError):
        while os.waitpid(-pgid, os.WNOHANG)[0] != 0:
            pass
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group(pgid: int, deadline: float) -> bool:  # pragma: posix-only
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _stop_group(pgid: int, deadline: float) -> bool:  # pragma: posix-only
    """Stop every remaining member of ``pgid`` once its leader is reaped; true when empty.

    Members get SIGTERM and the rest of the grace period (``deadline``), then SIGKILL.
    """
    if sys.platform == "win32":
        raise NotImplementedError("process groups are POSIX only")
    if not _group_alive(pgid):
        return True
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    if _wait_group(pgid, deadline):
        return True
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, _KILL_SIGNAL)
    return _wait_group(pgid, time.monotonic() + _GROUP_KILL_WAIT_S)


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
        # The child's process group (POSIX: ``start_new_session`` makes it the leader's pid),
        # tracked apart from the leader so descendants outliving it are still stopped.
        self._pgid: Optional[int] = None
        self.state = "stopped"
        self.exit_code: Optional[int] = None
        self.restarts = 0
        self.refuse_restart = False
        self._stopping = threading.Event()
        # Hermes waits for a successful first skills sync: until then it is not started, so
        # it never loads skills left over from before (possibly revoked since).
        self._start_after_sync = False
        self._seen_commands: set[str] = set()
        # Acknowledgements that failed in transport, retried on every tick until accepted.
        self._ack_retries: deque[tuple[str, str, dict[str, JsonValue]]] = deque(maxlen=500)
        self._terminal_acks: OrderedDict[str, tuple[str, dict[str, JsonValue]]] = OrderedDict()
        self._ticks = 0
        self._launch_failed = False
        self.gave_up = False
        # The leader exited and what remains of its process group is not stopped yet: no
        # replacement starts (and no restart decision is made) until the group is empty.
        self._group_pending = False

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
        if self._stopping.is_set():
            # A shutdown requested while a heartbeat or skills sync was blocked: nothing
            # (startup hooks included) runs after it.
            return False
        leader_gone = self.proc is None or self.proc.poll() is not None
        if self._pgid is not None and leader_gone and not self._release_group():
            # The previous group still runs: a replacement would leave it unsupervised.
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
        self._pgid = None if sys.platform == "win32" else self.proc.pid
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
            self._group_pending = True
        if not self._group_pending:
            return
        # Descendants outliving the leader are stopped before anything else: a replacement
        # would overwrite the recorded group and leave them unsupervised.
        if not self._release_group():
            return
        self._group_pending = False
        self.state = "exited"
        if code != 0 and not self._stopping.is_set():
            self._restart_or_give_up(f"exit code {code}")

    def _release_group(self) -> bool:
        """Stop what remains of the recorded process group once its leader is reaped.

        True when the group is empty (it is then forgotten: its id may be reused);
        otherwise the state is ``stop_failed`` and the group stays recorded.
        """
        if self._pgid is None:
            return True
        if not _stop_group(self._pgid, time.monotonic() + self.settings.grace_s):
            logger.error("Hermes process group %d did not stop", self._pgid)
            self.state = "stop_failed"
            return False
        self._pgid = None
        return True

    def _restart_or_give_up(self, why: str) -> None:
        if not self.settings.restart or self.refuse_restart:
            return
        if self.restarts >= self.settings.max_restarts:
            if not self.gave_up:
                logger.error("Hermes not restarted after %d attempts (%s)", self.restarts, why)
            self.gave_up = True
            return
        self.restarts += 1
        # A shutdown requested during the backoff ends it and starts nothing.
        if self._backoff_wait(min(60.0, 2.0**self.restarts)):
            return
        # Not started here: the same tick first heartbeats (a quarantine or revoke queued
        # meanwhile applies before any replacement runs) and syncs the approved skills.
        self._start_after_sync = True

    def _backoff_wait(self, seconds: float) -> bool:
        """Wait out the restart backoff; ``True`` when a shutdown was requested meanwhile."""
        return self._stopping.wait(seconds)

    def stop_child(self) -> Optional[int]:
        """SIGTERM the child's process group, SIGKILL after ``grace_s``. Returns the exit code.

        On POSIX the whole process group is stopped, not only the leader: members that
        outlive the leader are sent SIGKILL. ``state`` is ``stopped`` only once the group
        is empty, otherwise ``stop_failed``.

        Example:
            >>> sup = _demo_supervisor(["sleep", "5"])
            >>> sup.start_child()
            True
            >>> sup.stop_child(), sup.state
            (-15, 'stopped')
        """
        proc = self.proc
        self._group_pending = False
        if proc is None:
            self.state = "stopped"
            return self.exit_code
        deadline = time.monotonic() + self.settings.grace_s
        if proc.poll() is None:
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(self.settings.grace_s)
            except subprocess.TimeoutExpired:
                self._signal(proc, _KILL_SIGNAL)
                proc.wait()
        self.exit_code = proc.returncode
        if self._pgid is not None:
            if not _stop_group(self._pgid, deadline):
                logger.error("Hermes process group %d did not stop", self._pgid)
                self.state = "stop_failed"
                return self.exit_code
            self._pgid = None
        self.state = "stopped"
        return self.exit_code

    def _signal(self, proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        if sys.platform == "win32":
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(sig)
            return
        try:
            os.killpg(self._pgid if self._pgid is not None else os.getpgid(proc.pid), sig)
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
        # Every source the child reads its configuration from, additively: Hermes's own
        # files, the paths given explicitly and the adapter config file it is pointed to.
        config_paths: list[Path] = [home / "config.yaml", home / ".env"]
        config_paths += self.settings.config_paths
        adapter_config = self.child_env.get("AGENOMIC_HERMES_CONFIG")
        if adapter_config:
            config_paths.append(Path(adapter_config).expanduser())
        config_paths = list(dict.fromkeys(config_paths))
        skills_paths = [
            p for p in (self.settings.skills_dir, home / "skills", home / "plugins") if p
        ]
        skills_paths += self.settings.readonly_paths
        report = isolation_report(
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
        skills_dir = self.settings.skills_dir
        if skills_dir is not None and skills_dir_problem(skills_dir) is not None:
            # The supervisor refuses to sync into it: approved skills are not attested.
            report["skills_readonly"] = False
        return report

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
            if e.retryable:
                evicted = (
                    self._ack_retries[0]
                    if len(self._ack_retries) == self._ack_retries.maxlen
                    else None
                )
                self._ack_retries.append((command_id, status, detail))
                if evicted is not None and all(q[0] != evicted[0] for q in self._ack_retries):
                    self._forget_dropped_ack(*evicted)
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

    def _forget_dropped_ack(
        self, command_id: str, status: str, detail: dict[str, JsonValue]
    ) -> None:
        """An acknowledgement dropped from the full retry queue. A terminal one (``applied``,
        ``refused``) is kept as a tombstone: the gateway's redelivery is answered with it and
        never executed again (an old quarantine must not stop a Hermes started by a later
        resume). A ``received`` one makes the command executable again."""
        if _ACK_RANK.get(status, 0) >= _ACK_RANK["applied"]:
            self._terminal_acks[command_id] = (status, detail)
            while len(self._terminal_acks) > _MAX_TERMINAL_ACKS:
                oldest, _ = self._terminal_acks.popitem(last=False)
                self._seen_commands.discard(oldest)
        else:
            self._seen_commands.discard(command_id)

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
        if command_id in self._terminal_acks:
            # Its terminal acknowledgement was dropped: answer the redelivery with it.
            status, detail = self._terminal_acks.pop(command_id)
            self._ack(command_id, status, detail)
            return
        if not command_id or command_id in self._seen_commands:
            return
        self._seen_commands.add(command_id)
        kind = str(command.get("kind") or "")
        if str(command.get("status") or "requested") == "requested":
            self._ack(command_id, "received", {"executor": "supervisor"})
        if kind in ("quarantine", "revoke"):
            self.refuse_restart = True
            code = self.stop_child()
            if self.state != "stopped":
                # Code from the child's process group still runs: not applied. The command
                # is executed again when the gateway delivers it again.
                self._seen_commands.discard(command_id)
                return
            self._ack(command_id, "applied", {"process_state": self.state, "exit_code": code})
        elif kind == "resume":
            self.refuse_restart = False
            pending = self.proc is None or self.proc.poll() is not None
            if pending:
                # Never started here: the next skills sync must succeed first (same tick),
                # so a resumed Hermes cannot load a stale or revoked skill.
                self._start_after_sync = True
            self._ack(
                command_id,
                "applied",
                {"restarted": False, "start_pending": pending, "process_state": self.state},
            )
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
        valid = [s for s in skills if isinstance(s, dict)]
        try:
            # An unsafe directory is a failed sync (``None``), not "every skill rejected":
            # Hermes is not started on a tree that was never reconciled.
            counts = _sync_skills_checked(valid, self.settings.skills_dir)
            counts["rejected"] += len(skills) - len(valid)  # malformed entries
            return counts
        except UnsafeSkillsPathError as e:
            logger.error("approved skills not synced: %s", e)
            return None
        except OSError as e:
            # Disk full, permission or I/O errors: a failed sync (Hermes stays stopped
            # when it waits for one), retried on the next tick, never a supervisor crash.
            logger.error("approved skills not synced: %s", type(e).__name__)
            return None

    def tick(self) -> None:
        """One supervision step.

        Example:
            >>> sup = _demo_supervisor(commands=[{"id": "c1", "kind": "quarantine"}])
            >>> sup.tick()
            >>> sup.refuse_restart
            True
        """
        self.poll()
        # Commands first: a stop command is never delayed by a slow skills sync.
        self.heartbeat()
        if self._start_after_sync:
            self._start_when_synced(self.sync_skills())
        elif self._ticks % max(1, self.settings.skills_every) == 0:
            self._stop_unless_synced(self.sync_skills())
        self._ticks += 1
        self._retry_acks()

    def _stop_unless_synced(self, synced: Optional[dict[str, int]]) -> None:
        """A periodic sync that failed or rejected an entry leaves the approved skills
        unreconciled: a running Hermes is stopped and only restarted once a later sync
        reconciles them all."""
        if self.settings.skills_dir is None or (synced is not None and not synced.get("rejected")):
            return
        if self.proc is not None and self.proc.poll() is None:
            logger.error("approved skills not synced; Hermes is stopped until they are")
            self.stop_child()
            self._start_after_sync = True

    def _start_when_synced(self, synced: Optional[dict[str, int]]) -> None:
        """Start Hermes once the approved skills are fully reconciled (or no skills directory
        is configured). A failed sync, or one that rejected an entry (bad digest, unsafe or
        linked destination, write failure: that path was not reconciled), keeps it stopped
        and the next tick retries."""
        if self.settings.skills_dir is not None and (synced is None or synced.get("rejected")):
            if not self._start_after_sync:
                logger.error("approved skills not synced; Hermes is not started until they are")
            self._start_after_sync = True
            return
        self._start_after_sync = False
        self.start_child()

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
        # Commands still pending (a quarantine or revoke a previous supervisor only reported)
        # are fetched and applied while Hermes is not running: it starts only if allowed.
        # Without an answer from the gateway, Hermes starts and the next heartbeat applies them.
        self.heartbeat()
        self._start_when_synced(self.sync_skills())
        failed = False
        try:
            while not self._stopping.is_set():
                self.tick()
                if not self.settings.restart and (
                    (self.state in ("exited", "stopped") and not self._start_after_sync)
                    or self._group_pending
                ):
                    # Without restarts a group that did not stop is retried once more on
                    # the way out and the supervisor exits 1 if it still survives.
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
            if self.state == "stop_failed":
                failed = True
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
    if not token.startswith(SUPERVISOR_TOKEN_PREFIX):
        # Another credential (a provider key...) is never sent to the endpoint as a bearer.
        logger.error(
            "%s does not hold an Agenomic supervisor token (%s...)",
            SUPERVISOR_TOKEN_ENV,
            SUPERVISOR_TOKEN_PREFIX,
        )
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
