from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import httpx

from agenomic.exceptions import ApiError

__all__ = [
    "ERROR_RULES",
    "ErrorClass",
    "InfrastructureError",
    "IsolationViolation",
    "LeaseLost",
    "RecordedFixtureAmbiguous",
    "RecordedFixtureMiss",
    "RunnerConfigurationError",
    "SecretResolutionError",
    "TrialBudgetExceeded",
    "classify",
    "error_code",
]

ErrorClass = Literal["infrastructure", "runner_configuration", "agent"]
ErrorHook = Callable[[BaseException], Optional[ErrorClass]]

ERROR_RULES = "agenomic.experiments.error_rules/v1"
_INFRASTRUCTURE_STATUSES = frozenset({408, 409, 429})
_PROVIDER_INFRASTRUCTURE = frozenset(
    {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
)


class RunnerConfigurationError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details


class IsolationViolation(RunnerConfigurationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("isolation_violation", message, **details)


class SecretResolutionError(RunnerConfigurationError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("secret_unresolved", message, **details)


class InfrastructureError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class LeaseLost(Exception):
    def __init__(self, message: str = "the trial lease is no longer valid") -> None:
        super().__init__(message)


class RecordedFixtureMiss(Exception):
    def __init__(self, tool: str, arguments_hash: Optional[str] = None) -> None:
        super().__init__(f"no recorded response matches this call of {tool}")
        self.tool = tool
        self.arguments_hash = arguments_hash


class RecordedFixtureAmbiguous(Exception):
    def __init__(self, tool: str) -> None:
        super().__init__(f"several recorded responses match this call of {tool}")
        self.tool = tool


class TrialBudgetExceeded(Exception):
    def __init__(self, limit: str, value: int) -> None:
        super().__init__(f"the trial reached its {limit} budget of {value}")
        self.limit = limit
        self.value = value


def _status(exc: BaseException) -> Optional[int]:
    if isinstance(exc, ApiError):
        return exc.status
    for candidate in (exc, getattr(exc, "response", None)):
        value = getattr(candidate, "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _provider_infrastructure(exc: BaseException) -> bool:
    return any(kind.__name__ in _PROVIDER_INFRASTRUCTURE for kind in type(exc).__mro__)


def classify(exc: BaseException, hook: Optional[ErrorHook] = None) -> tuple[ErrorClass, str]:
    if hook is not None:
        chosen = hook(exc)
        if chosen is not None:
            if chosen not in ("infrastructure", "runner_configuration", "agent"):
                raise ValueError("classify_error must return an ErrorClass or None")
            return chosen, "user_hook"
    return _default_class(exc), "default_rules/v1"


def _default_class(exc: BaseException) -> ErrorClass:
    if isinstance(exc, RunnerConfigurationError):
        return "runner_configuration"
    if isinstance(exc, (InfrastructureError, httpx.TransportError, httpx.TimeoutException)):
        return "infrastructure"
    if _provider_infrastructure(exc):
        return "infrastructure"
    if isinstance(exc, ApiError) and exc.code == "registry_unavailable":
        return "infrastructure"
    status = _status(exc)
    if status is not None and (status in _INFRASTRUCTURE_STATUSES or status >= 500):
        return "infrastructure"
    return "agent"


def error_code(exc: BaseException) -> str:
    if isinstance(exc, (RunnerConfigurationError, InfrastructureError)):
        return exc.code
    if isinstance(exc, ApiError):
        if exc.code == "registry_unavailable":
            return "agenomic_unavailable"
        if exc.status == 0:
            return exc.code
    status = _status(exc)
    if status == 429 or (_provider_infrastructure(exc) and "RateLimit" in type(exc).__name__):
        return "provider_rate_limited"
    if status == 408 or isinstance(exc, httpx.TimeoutException) or "Timeout" in type(exc).__name__:
        return "provider_timeout"
    if isinstance(exc, httpx.TransportError) or "Connection" in type(exc).__name__:
        return "connection_error"
    if status is not None and status >= 500:
        return "provider_unavailable"
    if status == 409:
        return "provider_conflict"
    return "agent_exception"
