"""Typed errors of the Agents Vault surface, mapped from the server error codes.

Every class subclasses :class:`agenomic.tools.ToolExecutionError`, so code
that already handles tool-execution refusals keeps working: ``code`` carries
the server code and ``status`` the HTTP status (0 when no request was made).
Nothing here ever carries a request body, an argument value or a secret.

The SDK never decides entitlement: :class:`VaultNotEntitled` mirrors what the
server answered and exposes ``upgrade_hint`` so an integration can show it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import reduce
from typing import Any, Optional

from agenomic.tools.models import ToolExecutionError
from agenomic.vault.sensitive import MASK

_OUTCOME_UNKNOWN_HINT = (
    "the request was sent and its effect is unknown; the SDK never retries it. "
    "Verify the destination, then settle it with client.vault.executions.resolve()"
)


class VaultError(ToolExecutionError):
    """Base of every Agents Vault failure.

    Example:
        >>> error = VaultError("vault_fail_closed", "failed closed", 409, action_id="a-1")
        >>> (error.code, error.status, error.action_id)
        ('vault_fail_closed', 409, 'a-1')
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 0,
        *,
        request_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> None:
        super().__init__(code, message, status)
        self.request_id = request_id
        self.action_id = action_id

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (_restore, (type(self), self.args, dict(self.__dict__)))


def _restore(cls: type[VaultError], args: tuple[Any, ...], state: dict[str, Any]) -> VaultError:
    error = cls.__new__(cls)
    Exception.__init__(error, *args)
    error.__dict__.update(state)
    return error


class VaultNotConfigured(VaultError):
    """The client lacks what the call needs (API key, runtime token, cloud ``base_url``).

    Raised before any request is made.

    Example:
        >>> from agenomic import Client
        >>> Client().tools.execute(tool="t", binding="b")
        Traceback (most recent call last):
        ...
        agenomic.vault.errors.VaultNotConfigured: cloud_required: Agents Vault needs Agenomic Cloud (base_url) or a replay fixture set
    """


class VaultAuthenticationError(VaultError):
    """The credential was missing, expired or revoked (HTTP 401)."""


class VaultPermissionDenied(VaultError):
    """The caller lacks the permission or the human assurance the operation needs (HTTP 403)."""


class VaultNotEntitled(VaultError):
    """The Agents Vault add-on is locked for this workspace (HTTP 403, ``capability_*``).

    ``reason`` is the server's evaluation (``not_entitled``, ``not_in_edition``,
    ``disabled``, ``beta_restricted``), ``required_plan`` its plan hint.
    ``upgrade_hint`` is true when the server says an upgrade path exists.

    Example:
        >>> error = VaultNotEntitled("capability_not_entitled", "locked", 403,
        ...     capability="agents_vault", reason="not_entitled", required_plan="cloud")
        >>> (error.locked, error.upgrade_hint, error.required_plan)
        (True, True, 'cloud')
    """

    locked = True

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 403,
        *,
        capability: Optional[str] = None,
        reason: Optional[str] = None,
        required_plan: Optional[str] = None,
        request_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> None:
        super().__init__(code, message, status, request_id=request_id, action_id=action_id)
        self.capability = capability
        self.reason = reason
        self.required_plan = required_plan

    @property
    def upgrade_hint(self) -> bool:
        return self.reason in ("not_entitled", "not_in_edition")


class VaultApprovalRequired(VaultError):
    """A policy requires a human approval; nothing was executed (HTTP 202).

    Decide ``approval_id`` through ``client.protect.approvals``, then call
    ``execute`` again with the same ``action_id``.

    Example:
        >>> error = VaultApprovalRequired("a-1", "apr-9")
        >>> (error.code, error.status, error.approval_id, error.action_id)
        ('approval_required', 202, 'apr-9', 'a-1')
    """

    def __init__(self, action_id: str, approval_id: str, *, request_id: Optional[str] = None):
        super().__init__(
            "approval_required",
            f"action {action_id} waits for approval {approval_id}; "
            "decide it, then execute again with the same action_id",
            202,
            request_id=request_id,
            action_id=action_id,
        )
        self.approval_id = approval_id


class VaultPolicyDenied(VaultError):
    """Policy refused the action; nothing was executed (HTTP 403, ``denied``).

    Example:
        >>> error = VaultPolicyDenied("a-1", ["no_policy_bound"], "no policy is bound")
        >>> (error.code, error.reason_codes)
        ('policy_denied', ['no_policy_bound'])
    """

    def __init__(
        self,
        action_id: str,
        reason_codes: Iterable[str] = (),
        explanation: Optional[str] = None,
        *,
        request_id: Optional[str] = None,
    ) -> None:
        self.reason_codes = list(reason_codes)
        detail = explanation or ", ".join(self.reason_codes) or "denied"
        super().__init__(
            "policy_denied",
            f"action {action_id} was not admitted: {detail}",
            403,
            request_id=request_id,
            action_id=action_id,
        )
        self.explanation = explanation


class VaultGrantUnusable(VaultError):
    """No usable grant covers the action (HTTP 409, ``vault_grant_unusable``).

    ``reason`` is ``not_found`` (no grant), ``not_approved``, ``expired`` or
    ``exhausted`` when the server message names it, otherwise ``None``.

    Example:
        >>> error = VaultGrantUnusable("vault_grant_unusable", "grant is not usable: exhausted", 409,
        ...     reason="exhausted")
        >>> (error.exhausted, error.missing)
        (True, False)
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 409,
        *,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> None:
        super().__init__(code, message, status, request_id=request_id, action_id=action_id)
        self.reason = reason

    @property
    def exhausted(self) -> bool:
        return self.reason == "exhausted"

    @property
    def missing(self) -> bool:
        return self.reason == "not_found"


class VaultRevoked(VaultError):
    """The binding, secret, grant or identity is revoked or suspended (HTTP 409, ``vault_revoked``)."""


class VaultOutcomeUnknown(VaultError):
    """The request was sent and its effect is unknown. The SDK never retries it.

    A second submission with the same ``action_id`` returns this stored state
    again and never repeats the external effect. Verify the destination, then
    settle the execution with ``client.vault.executions.resolve(action_id, ...)``.

    Example:
        >>> error = VaultOutcomeUnknown("a-1", status_code=504)
        >>> (error.code, error.action_id, error.status_code)
        ('outcome_unknown', 'a-1', 504)
    """

    def __init__(
        self,
        action_id: str,
        *,
        status_code: Optional[int] = None,
        error_class: Optional[str] = None,
        receipt_id: Optional[str] = None,
        limitations: Iterable[str] = (),
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            "outcome_unknown",
            f"action {action_id}: {_OUTCOME_UNKNOWN_HINT}",
            409,
            request_id=request_id,
            action_id=action_id,
        )
        self.status_code = status_code
        self.error_class = error_class
        self.receipt_id = receipt_id
        self.limitations = list(limitations)


class VaultExecutionFailed(VaultError):
    """The destination answered with a failure; ``result`` holds the filtered body if any.

    Example:
        >>> error = VaultExecutionFailed("a-1", error_class="destination_error", status_code=404)
        >>> (error.code, error.status_code)
        ('execution_failed', 404)
    """

    def __init__(
        self,
        action_id: str,
        *,
        error_class: Optional[str] = None,
        status_code: Optional[int] = None,
        receipt_id: Optional[str] = None,
        result: object = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            "execution_failed",
            f"action {action_id} failed ({error_class or status_code or 'unknown'})",
            0,
            request_id=request_id,
            action_id=action_id,
        )
        self.error_class = error_class
        self.status_code = status_code
        self.receipt_id = receipt_id
        self.result = result


class VaultExecutionInProgress(VaultError):
    """The execution is not settled yet; poll it, do not resend a different request."""

    def __init__(self, action_id: str, state: str, *, request_id: Optional[str] = None) -> None:
        super().__init__(
            "execution_in_progress",
            f"action {action_id} is {state}; poll its status instead of resending",
            0,
            request_id=request_id,
            action_id=action_id,
        )
        self.state = state


class VaultRateLimited(VaultError):
    """Too many requests (HTTP 429); ``retry_after`` is the server hint in seconds."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 429,
        *,
        retry_after: Optional[float] = None,
        request_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> None:
        super().__init__(code, message, status, request_id=request_id, action_id=action_id)
        self.retry_after = retry_after


class VaultValidationError(VaultError):
    """The request is not valid (HTTP 400, or refused locally before any request)."""


class VaultNotFound(VaultError):
    """The resource does not exist in this workspace (HTTP 404)."""


class VaultConflict(VaultError):
    """The request conflicts with the current state (HTTP 409, ``conflict``)."""


class VaultRefused(VaultError):
    """The server refused the action with a ``vault_*`` code or a ``refused`` envelope."""


class VaultFailClosed(VaultRefused):
    """The server failed closed because a precondition could not be established."""


class VaultUnavailable(VaultRefused):
    """The vault backend or the destination is unavailable; the SDK does not retry it."""


class VaultServerError(VaultError):
    """The server failed (HTTP 5xx). Resend with the same ``action_id`` if it was an execution."""


class VaultTransportError(VaultError):
    """The request could not be completed; for an execution it may or may not have arrived.

    Reuse ``action_id`` to resend: the server de-duplicates, so the external
    effect is never repeated.
    """

    def __init__(self, message: str, *, action_id: Optional[str] = None) -> None:
        super().__init__("transport_error", message, 0, action_id=action_id)


class ReplayFixtureMissing(VaultError):
    """No replay fixture matches the call; nothing was executed and nothing fell back to live.

    Carries the code ``mock_unmatched`` of the tool mock engine.

    Example:
        >>> error = ReplayFixtureMissing("crm.get_customer", "crm-read", "blake3:abc")
        >>> (error.code, error.tool, error.binding)
        ('mock_unmatched', 'crm.get_customer', 'crm-read')
    """

    def __init__(self, tool: str, binding: str, arguments_hash: str) -> None:
        super().__init__(
            "mock_unmatched",
            f"no replay fixture for tool {tool!r} on binding {binding!r} "
            f"with arguments {arguments_hash}; replay never falls back to a live call",
            0,
        )
        self.tool = tool
        self.binding = binding
        self.arguments_hash = arguments_hash


class ReplayUnsupported(VaultError):
    """The operation has no replay counterpart and is not sent live while replaying."""


_SIMPLE: Mapping[str, type[VaultError]] = {
    "unauthorized": VaultAuthenticationError,
    "forbidden": VaultPermissionDenied,
    "vault_permission_denied": VaultPermissionDenied,
    "not_found": VaultNotFound,
    "validation_error": VaultValidationError,
    "conflict": VaultConflict,
    "vault_revoked": VaultRevoked,
    "vault_fail_closed": VaultFailClosed,
    "vault_backend_unavailable": VaultUnavailable,
    "vault_backend_rejected": VaultUnavailable,
    "vault_destination_unavailable": VaultUnavailable,
    "internal_error": VaultServerError,
}

_BY_STATUS: Mapping[int, str] = {
    400: "validation_error",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "too_many_requests",
}

_GRANT_REASONS = ("not_found", "not_approved", "expired", "exhausted")


def _scrubbed(text: str, needles: Iterable[str]) -> str:
    return reduce(lambda acc, needle: acc.replace(needle, MASK) if needle else acc, needles, text)


def _string(value: object, needles: Iterable[str]) -> Optional[str]:
    return _scrubbed(value, needles) if isinstance(value, str) else None


def _envelope(payload: object) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        inner = payload.get("error")
        return inner if isinstance(inner, Mapping) else payload
    return {}


def _grant_reason(message: str) -> Optional[str]:
    tail = message.rsplit(": ", 1)[-1].strip()
    return tail if tail in _GRANT_REASONS else None


def _retry_after(headers: Mapping[str, str]) -> Optional[float]:
    try:
        return float(headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _execution_error(
    body: Mapping[str, Any], action_id: Optional[str], needles: Iterable[str]
) -> Optional[VaultError]:
    kind, ident = body.get("status"), str(action_id or body.get("action_id") or "")
    if kind == "approval_required":
        return VaultApprovalRequired(ident, str(body.get("approval_id", "")))
    if kind == "denied":
        codes = [c for c in body.get("reason_codes") or [] if isinstance(c, str)]
        return VaultPolicyDenied(ident, codes, _string(body.get("explanation"), needles))
    if kind == "refused":
        code = _string(body.get("code"), needles) or "refused"
        return VaultRefused(
            code, _string(body.get("message"), needles) or code, 409, action_id=ident
        )
    return None


def build_error(
    status: int,
    payload: object,
    headers: Mapping[str, str],
    *,
    action_id: Optional[str] = None,
    scrub: Iterable[str] = (),
) -> VaultError:
    """Map an HTTP failure to the typed error the server code stands for.

    ``scrub`` lists strings (a secret value and its JSON-escaped form) that are
    masked out of every server-provided text, as defence in depth.

    Example:
        >>> error = build_error(403, {"error": {"code": "capability_not_entitled",
        ...     "message": "locked", "reason": "not_entitled", "capability": "agents_vault"}}, {})
        >>> type(error).__name__, error.upgrade_hint
        ('VaultNotEntitled', True)
    """
    needles = list(scrub)
    envelope = _execution_error(_envelope(payload), action_id, needles)
    if envelope is not None:
        return envelope
    body = _envelope(payload)
    code = _string(body.get("code"), needles) or _BY_STATUS.get(status, "http_error")
    message = _string(body.get("message"), needles) or f"the server answered {status}"
    return _by_code(status, code, message, body, headers, action_id, needles)


def _by_code(
    status: int,
    code: str,
    message: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    action_id: Optional[str],
    needles: list[str],
) -> VaultError:
    request_id = _string(body.get("request_id"), needles)
    common: dict[str, Any] = {"request_id": request_id, "action_id": action_id}
    if code.startswith("capability_"):
        reason = _string(body.get("reason"), needles) or code.removeprefix("capability_")
        return VaultNotEntitled(
            code,
            message,
            status,
            capability=_string(body.get("capability"), needles),
            reason=reason,
            required_plan=_string(body.get("required_plan"), needles),
            **common,
        )
    if code == "vault_grant_unusable":
        return VaultGrantUnusable(code, message, status, reason=_grant_reason(message), **common)
    if code == "vault_outcome_unknown":
        return VaultOutcomeUnknown(action_id or "", request_id=request_id)
    if code == "too_many_requests":
        return VaultRateLimited(code, message, status, retry_after=_retry_after(headers), **common)
    return _simple(status, code, message, common)


def _simple(status: int, code: str, message: str, common: dict[str, Any]) -> VaultError:
    cls = _SIMPLE.get(code)
    if cls is not None:
        return cls(code, message, status, **common)
    if status >= 500:
        return VaultServerError(code, message, status, **common)
    if code.startswith("vault_"):
        return VaultRefused(code, message, status, **common)
    return VaultError(code, message, status, **common)
