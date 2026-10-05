"""Agents Vault: use credentials without ever receiving them.

Optional commercial module of Agenomic Cloud/Enterprise; it needs the Agents
Vault add-on. The agent holds an authorization, the executor holds the
credential: ``client.tools.execute`` asks the executor to perform one logical
action and returns the business result and a receipt, never a secret value.
The administrative plane is ``client.vault``.
"""

from agenomic.vault.errors import (
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
from agenomic.vault.sensitive import MASK, IssuedToken, Sensitive, SensitiveUnavailable

__all__ = [
    "MASK",
    "IssuedToken",
    "ReplayFixtureMissing",
    "ReplayUnsupported",
    "Sensitive",
    "SensitiveUnavailable",
    "VaultApprovalRequired",
    "VaultAuthenticationError",
    "VaultConflict",
    "VaultError",
    "VaultExecutionFailed",
    "VaultExecutionInProgress",
    "VaultFailClosed",
    "VaultGrantUnusable",
    "VaultNotConfigured",
    "VaultNotEntitled",
    "VaultNotFound",
    "VaultOutcomeUnknown",
    "VaultPermissionDenied",
    "VaultPolicyDenied",
    "VaultRateLimited",
    "VaultRefused",
    "VaultRevoked",
    "VaultServerError",
    "VaultTransportError",
    "VaultUnavailable",
    "VaultValidationError",
]
