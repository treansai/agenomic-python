"""The execution path: request normalisation and the mapping of outcomes to results or errors.

A live reply and a replay fixture go through the same :func:`outcome_of`, so
the code that handles ``VaultOutcomeUnknown`` or ``VaultApprovalRequired`` in a
test is the code that handles it in production.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from pydantic import JsonValue

from agenomic.vault.errors import (
    VaultError,
    VaultExecutionFailed,
    VaultExecutionInProgress,
    VaultOutcomeUnknown,
    VaultPolicyDenied,
    VaultValidationError,
    build_error,
)
from agenomic.vault.models import ExecuteResult, ExecutionStatus
from agenomic.vault.transport import Call, Reply, validate

EXECUTIONS_PATH = "/v1/vault/runtime/executions"
_ENVELOPES = ("approval_required", "denied", "refused")


def normalize_action_id(action_id: Optional[str]) -> str:
    """A UUID string: generated when absent, validated and lower-cased otherwise.

    Example:
        >>> len(normalize_action_id(None))
        36
        >>> normalize_action_id("0A1B2C3D-0000-4000-8000-000000000001")
        '0a1b2c3d-0000-4000-8000-000000000001'
    """
    if action_id is None:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(action_id)))
    except ValueError:
        pass
    raise VaultValidationError("invalid_action_id", "action_id must be a UUID", 0)


@dataclass(frozen=True)
class ExecuteRequest:
    """One logical action. Every technical retry resends exactly this request."""

    tool: str
    binding: str
    arguments: Mapping[str, JsonValue] = field(repr=False)
    action_id: str
    deadline_ms: Optional[int] = None

    @classmethod
    def of(
        cls,
        tool: str,
        binding: str,
        arguments: Optional[Mapping[str, JsonValue]],
        action_id: Optional[str],
        deadline_ms: Optional[int],
    ) -> ExecuteRequest:
        if not tool or not binding:
            raise VaultValidationError("invalid_request", "tool and binding are required", 0)
        if deadline_ms is not None and deadline_ms <= 0:
            raise VaultValidationError("invalid_request", "deadline_ms must be positive", 0)
        return cls(
            tool, binding, dict(arguments or {}), normalize_action_id(action_id), deadline_ms
        )

    def call(self) -> Call:
        body: dict[str, object] = {
            "tool": self.tool,
            "binding": self.binding,
            "arguments": dict(self.arguments),
            "action_id": self.action_id,
        }
        if self.deadline_ms is not None:
            body["deadline_ms"] = self.deadline_ms
        return Call("POST", EXECUTIONS_PATH, "runtime", body, {}, "action", self.action_id)


def _codes(error_class: Optional[str]) -> list[str]:
    return [code for code in (error_class or "").split(",") if code]


def outcome_of(status: ExecutionStatus, action_id: str) -> ExecuteResult:
    """The result of a succeeded execution, or the typed error every other state stands for."""
    ident = action_id or status.action_id
    if status.state == "succeeded":
        return ExecuteResult(
            action_id=ident,
            state="succeeded",
            result=status.result,
            receipt_id=status.receipt_id,
            status_code=status.status_code,
            limitations=status.limitations,
            replayed=status.replayed,
        )
    if status.state == "outcome_unknown":
        raise VaultOutcomeUnknown(
            ident,
            status_code=status.status_code,
            error_class=status.error_class,
            receipt_id=status.receipt_id,
            limitations=status.limitations,
        )
    if status.state == "failed":
        raise VaultExecutionFailed(
            ident,
            error_class=status.error_class,
            status_code=status.status_code,
            receipt_id=status.receipt_id,
            result=status.result,
        )
    if status.state == "refused":
        raise VaultPolicyDenied(ident, _codes(status.error_class))
    raise VaultExecutionInProgress(ident, status.state or "unknown")


def parse_status(reply: Reply, action_id: str) -> ExecutionStatus:
    """A status read: any state is returned, only HTTP and shape failures raise."""
    if not isinstance(reply.payload, Mapping):
        raise VaultError("invalid_response", "the response is not an object", reply.status)
    status = validate(ExecutionStatus, reply.payload, reply.status)
    return status if status.action_id else status.model_copy(update={"action_id": action_id})


def settle(reply: Reply, action_id: str) -> ExecuteResult:
    """Turn the reply to an execute call into a result or the error it stands for."""
    payload = reply.payload
    if isinstance(payload, Mapping) and payload.get("status") in _ENVELOPES:
        raise build_error(reply.status, payload, reply.headers, action_id=action_id)
    return outcome_of(parse_status(reply, action_id), action_id)
