from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, NoReturn, Optional, Protocol, runtime_checkable

from agenomic.experiments.errors import SecretResolutionError
from agenomic.prompts.secrets import scrub
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule

__all__ = [
    "DEFAULT_RUNNER_REDACTION_RULES",
    "REDACTED",
    "EnvSecretResolver",
    "SecretResolver",
    "SecretValues",
    "redact_outbound",
    "redact_message",
    "replace_secrets",
]

REDACTED = "[REDACTED]"
MAX_MESSAGE_BYTES = 2048
WITHHELD = "a log record was withheld because it could not be redacted"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_RUNNER_TOKEN = re.compile(r"agr_[0-9a-f]{64}", re.ASCII)
_FORMATTER = logging.Formatter()
_SECRET_KEYS = (
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
)
DEFAULT_RUNNER_REDACTION_RULES: list[RedactionRule] = [
    RedactionRule(path=f"**.{key}", mode=RedactionMode.MASK) for key in _SECRET_KEYS
]


@runtime_checkable
class SecretResolver(Protocol):
    def names(self) -> list[str]: ...

    def resolve(self, ref: str) -> str: ...


class EnvSecretResolver:
    def __init__(
        self, allow: Iterable[str] = (), *, environ: Optional[Mapping[str, str]] = None
    ) -> None:
        names: list[str] = []
        for name in allow:
            if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
                raise ValueError("allow lists environment variable names")
            if name not in names:
                names.append(name)
        self._allow = tuple(names)
        self._environ = environ

    def names(self) -> list[str]:
        return [f"env:{name}" for name in self._allow]

    def resolve(self, ref: str) -> str:
        scheme, _, name = ref.partition(":")
        if scheme != "env" or name not in self._allow:
            raise SecretResolutionError(f"secret ref {ref} is not allowed on this runner", ref=ref)
        environ = self._environ if self._environ is not None else os.environ
        value = environ.get(name)
        if not value:
            raise SecretResolutionError(f"secret ref {ref} has no value on this runner", ref=ref)
        return value


class SecretValues(Mapping[str, str]):
    __slots__ = ("_values",)
    _values: dict[str, str]

    def __init__(self, values: Optional[Mapping[str, str]] = None) -> None:
        object.__setattr__(self, "_values", dict(values or {}))

    def __getitem__(self, ref: str) -> str:
        return self._values[ref]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"SecretValues(refs={sorted(self._values)!r}, values=hidden)"

    __str__ = __repr__

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("SecretValues is immutable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretValues cannot be serialized")

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("SecretValues cannot be serialized")

    def __copy__(self) -> SecretValues:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> SecretValues:
        return self

    def literals(self) -> list[str]:
        return sorted({value for value in self._values.values() if value}, key=len, reverse=True)


def replace_secrets(value: Any, literals: Iterable[str]) -> Any:
    ordered = sorted({item for item in literals if item}, key=len, reverse=True)
    if not ordered:
        return value
    return _replace(value, ordered)


def _replace(value: Any, literals: list[str]) -> Any:
    if isinstance(value, str):
        for literal in literals:
            if literal in value:
                value = value.replace(literal, REDACTED)
        return value
    if isinstance(value, (list, tuple)):
        return [_replace(item, literals) for item in value]
    if isinstance(value, Mapping):
        return {
            _replace(str(key), literals): _replace(item, literals) for key, item in value.items()
        }
    return value


def redact_outbound(value: Any, literals: Iterable[str]) -> Any:
    engine = RedactionEngine(DEFAULT_RUNNER_REDACTION_RULES)
    return replace_secrets(engine.apply(value), literals)


def redact_message(text: str, literals: Iterable[str]) -> str:
    cleaned = replace_secrets(scrub(text), literals)
    encoded = str(cleaned).encode("utf-8")
    if len(encoded) <= MAX_MESSAGE_BYTES:
        return str(cleaned)
    return encoded[:MAX_MESSAGE_BYTES].decode("utf-8", errors="ignore")


def _redact_log_text(text: str, literals: Iterable[str]) -> str:
    return _RUNNER_TOKEN.sub(REDACTED, scrub(str(replace_secrets(text, literals))))


class LogRedactor(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._sources: dict[object, Callable[[], Iterable[str]]] = {}

    @contextmanager
    def tracking(self, literals: Callable[[], Iterable[str]]) -> Iterator[None]:
        key = object()
        with self._lock:
            self._sources[key] = literals
        try:
            yield
        finally:
            with self._lock:
                del self._sources[key]

    def literals(self) -> list[str]:
        with self._lock:
            sources = list(self._sources.values())
        return [literal for source in sources for literal in source()]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            literals = self.literals()
            record.msg = _redact_log_text(record.getMessage(), literals)
            if record.exc_info:
                record.exc_text = _FORMATTER.formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = _redact_log_text(record.exc_text, literals)
            if record.stack_info:
                record.stack_info = _redact_log_text(record.stack_info, literals)
        except Exception:
            record.msg, record.exc_text, record.stack_info = WITHHELD, None, None
        record.args = ()
        record.exc_info = None
        return True


LOG_REDACTOR = LogRedactor()
