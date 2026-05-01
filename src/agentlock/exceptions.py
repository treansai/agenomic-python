"""Exception hierarchy for agentlock."""
from __future__ import annotations


class AgentLockError(Exception):
    """Base for all AgentLock SDK errors."""


class ValidationError(AgentLockError):
    """Trace, event, or schema validation failed."""


class CryptoError(AgentLockError):
    """Hashing, signing, or canonical encoding failed."""


class AtepError(AgentLockError):
    """ATEP segment integrity, format, or signature error."""


class ExportError(AgentLockError):
    """Export to JSONL, ATEP, or HTTP failed."""


class CloudError(AgentLockError):
    """AgentLock Cloud HTTP error."""


class AuthenticationError(CloudError):
    """Cloud authentication failed."""


class RedactionError(AgentLockError):
    """Redaction rule could not be applied."""
