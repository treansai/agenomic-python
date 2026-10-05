"""Mapping of server error codes to typed exceptions, and the safety of the exceptions themselves."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from agenomic.exceptions import AgenomicError, CloudError
from agenomic.tools import ToolExecutionError
from agenomic.vault import (
    ReplayFixtureMissing,
    ReplayUnsupported,
    VaultApprovalRequired,
    VaultAuthenticationError,
    VaultConflict,
    VaultError,
    VaultExecutionFailed,
    VaultExecutionInProgress,
    VaultFailClosed,
    VaultGrantUnusable,
    VaultNotConfigured,
    VaultNotEntitled,
    VaultNotFound,
    VaultOutcomeUnknown,
    VaultPermissionDenied,
    VaultPolicyDenied,
    VaultRateLimited,
    VaultRefused,
    VaultRevoked,
    VaultServerError,
    VaultTransportError,
    VaultUnavailable,
    VaultValidationError,
)
from agenomic.vault.errors import build_error

CANARY = 'canary"secret\\value-0e4b'


def _err(code: str, message: str = "m", **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **extra}}


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (401, _err("unauthorized"), VaultAuthenticationError),
        (403, _err("forbidden"), VaultPermissionDenied),
        (403, _err("vault_permission_denied"), VaultPermissionDenied),
        (403, _err("capability_not_entitled", reason="not_entitled"), VaultNotEntitled),
        (403, _err("capability_not_in_edition", reason="not_in_edition"), VaultNotEntitled),
        (403, _err("capability_disabled", reason="disabled"), VaultNotEntitled),
        (403, _err("capability_beta_only"), VaultNotEntitled),
        (404, _err("not_found"), VaultNotFound),
        (400, _err("validation_error"), VaultValidationError),
        (409, _err("conflict"), VaultConflict),
        (409, _err("vault_revoked"), VaultRevoked),
        (409, _err("vault_grant_unusable", "grant is not usable: expired"), VaultGrantUnusable),
        (409, _err("vault_outcome_unknown"), VaultOutcomeUnknown),
        (
            409,
            _err("vault_fail_closed", "failed closed: entitlement could not be verified"),
            VaultFailClosed,
        ),
        (409, _err("vault_backend_unavailable"), VaultUnavailable),
        (409, _err("vault_backend_rejected"), VaultUnavailable),
        (409, _err("vault_destination_unavailable"), VaultUnavailable),
        (409, _err("vault_destination_denied"), VaultRefused),
        (409, _err("vault_result_blocked"), VaultRefused),
        (409, _err("vault_authorization_invalid"), VaultRefused),
        (409, _err("vault_unsupported"), VaultRefused),
        (429, _err("too_many_requests"), VaultRateLimited),
        (500, _err("internal_error"), VaultServerError),
        (502, None, VaultServerError),
        (503, "gateway text", VaultServerError),
        (400, None, VaultValidationError),
        (404, None, VaultNotFound),
        (418, None, VaultError),
    ],
)
def test_server_codes_and_statuses_map_to_the_typed_error(
    status: int, payload: Any, expected: type[VaultError]
) -> None:
    error = build_error(status, payload, {}, action_id="a-1")
    assert type(error) is expected
    assert error.status == status
    assert isinstance(error, ToolExecutionError)
    assert isinstance(error, CloudError)
    assert isinstance(error, AgenomicError)


def test_the_error_envelope_fields_are_kept() -> None:
    error = build_error(
        403,
        _err(
            "capability_not_entitled",
            "agents_vault is not included in the current plan",
            request_id="req-9",
            capability="agents_vault",
            reason="not_entitled",
            required_plan="enterprise",
        ),
        {},
    )
    assert isinstance(error, VaultNotEntitled)
    assert (error.request_id, error.capability, error.reason, error.required_plan) == (
        "req-9",
        "agents_vault",
        "not_entitled",
        "enterprise",
    )
    assert error.code == "capability_not_entitled"


def test_a_capability_code_without_a_reason_derives_it_from_the_code() -> None:
    error = build_error(403, _err("capability_not_in_edition"), {})
    assert isinstance(error, VaultNotEntitled)
    assert (error.reason, error.upgrade_hint) == ("not_in_edition", True)
    unavailable = build_error(403, _err("capability_unavailable"), {})
    assert isinstance(unavailable, VaultNotEntitled)
    assert unavailable.upgrade_hint is False


def test_a_locked_add_on_is_not_a_generic_permission_error() -> None:
    locked = build_error(403, _err("capability_not_entitled", reason="not_entitled"), {})
    denied = build_error(403, _err("forbidden"), {})
    assert isinstance(locked, VaultNotEntitled)
    assert not isinstance(denied, VaultNotEntitled)
    assert not isinstance(locked, VaultPermissionDenied)


def test_the_execution_envelopes_win_over_the_http_status() -> None:
    approval = build_error(
        202, {"status": "approval_required", "action_id": "a", "approval_id": "p"}, {}
    )
    denied = build_error(
        403, {"status": "denied", "action_id": "a", "reason_codes": ["r1", 7, "r2"]}, {}
    )
    refused = build_error(
        409, {"status": "refused", "action_id": "a", "code": "live_call_denied", "message": "m"}, {}
    )
    assert isinstance(approval, VaultApprovalRequired)
    assert approval.approval_id == "p"
    assert isinstance(denied, VaultPolicyDenied)
    assert denied.reason_codes == ["r1", "r2"]
    assert isinstance(refused, VaultRefused)
    assert refused.code == "live_call_denied"


def test_the_requested_action_id_is_preferred_to_the_one_in_the_body() -> None:
    error = build_error(
        403, {"status": "denied", "action_id": "from-body"}, {}, action_id="requested"
    )
    assert error.action_id == "requested"


def test_the_retry_after_header_is_parsed_and_a_bad_one_ignored() -> None:
    assert build_error(429, _err("too_many_requests"), {"retry-after": "12"}).retry_after == 12.0  # type: ignore[attr-defined]
    assert build_error(429, _err("too_many_requests"), {"retry-after": "soon"}).retry_after is None  # type: ignore[attr-defined]
    assert build_error(429, None, {}).retry_after is None  # type: ignore[attr-defined]


def test_the_grant_reason_is_only_taken_from_the_known_set() -> None:
    known = build_error(409, _err("vault_grant_unusable", "grant is not usable: not_approved"), {})
    unknown = build_error(409, _err("vault_grant_unusable", "grant is not usable: banana"), {})
    assert isinstance(known, VaultGrantUnusable)
    assert isinstance(unknown, VaultGrantUnusable)
    assert (known.reason, unknown.reason) == ("not_approved", None)


def test_a_server_that_echoes_a_value_is_masked_in_every_text_field() -> None:
    echoed = _err(
        "validation_error",
        f"invalid value {CANARY} for field",
        request_id=f"req-{CANARY}",
    )
    escaped = CANARY.replace("\\", "\\\\").replace('"', '\\"')
    error = build_error(400, echoed, {}, scrub=[CANARY, escaped])
    assert CANARY not in str(error) + repr(error) + str(error.request_id) + repr(error.args)
    assert "**********" in str(error)


def test_every_error_class_survives_pickle_with_its_attributes_and_carries_no_traceback() -> None:
    samples: list[VaultError] = [
        VaultError("c", "m", 400, request_id="r", action_id="a"),
        VaultNotConfigured("cloud_required", "m"),
        VaultNotEntitled(
            "capability_not_entitled",
            "m",
            403,
            capability="agents_vault",
            reason="not_entitled",
            required_plan="p",
        ),
        VaultApprovalRequired("a", "p"),
        VaultPolicyDenied("a", ["r"], "why"),
        VaultGrantUnusable("vault_grant_unusable", "m", 409, reason="exhausted"),
        VaultOutcomeUnknown(
            "a", status_code=504, error_class="x", receipt_id="r", limitations=["l"]
        ),
        VaultExecutionFailed(
            "a", error_class="x", status_code=404, receipt_id="r", result={"k": 1}
        ),
        VaultExecutionInProgress("a", "sent"),
        VaultRateLimited("too_many_requests", "m", 429, retry_after=3.0),
        VaultTransportError("down", action_id="a"),
        ReplayFixtureMissing("t", "b", "blake3:x"),
        ReplayUnsupported("replay_unsupported", "m"),
        VaultRevoked("vault_revoked", "m", 409),
    ]
    for error in samples:
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        assert restored.__dict__ == error.__dict__
        assert restored.__traceback__ is None


def test_exception_pickles_carry_the_scrubbed_text_only() -> None:
    error = build_error(400, _err("validation_error", f"bad {CANARY}"), {}, scrub=[CANARY])
    assert CANARY.encode() not in pickle.dumps(error)


def test_outcome_unknown_says_it_will_not_be_retried_and_how_to_settle_it() -> None:
    error = VaultOutcomeUnknown("a-1")
    assert (error.code, error.status, error.action_id) == ("outcome_unknown", 409, "a-1")
    assert "never retries" in str(error)
    assert "executions.resolve" in str(error)


def test_the_policy_denied_message_falls_back_to_the_reason_codes() -> None:
    assert "no_policy_bound" in str(VaultPolicyDenied("a", ["no_policy_bound"]))
    assert "denied" in str(VaultPolicyDenied("a"))


def test_approval_required_tells_how_to_resume() -> None:
    message = str(VaultApprovalRequired("a-1", "apr-1"))
    assert "apr-1" in message
    assert "same action_id" in message
