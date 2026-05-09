"""Exception hierarchy for agenomic."""

from __future__ import annotations


class AgenomicError(Exception):
    """Base for all Agenomic SDK errors."""


class ValidationError(AgenomicError):
    """Trace, event, or schema validation failed."""


class CryptoError(AgenomicError):
    """Hashing, signing, or canonical encoding failed."""


class AtepError(AgenomicError):
    """ATEP segment integrity, format, or signature error."""


class ExportError(AgenomicError):
    """Export to JSONL, ATEP, or HTTP failed."""


class CloudError(AgenomicError):
    """Agenomic Cloud HTTP error."""


class AuthenticationError(CloudError):
    """Cloud authentication failed."""


class RedactionError(AgenomicError):
    """Redaction rule could not be applied."""
