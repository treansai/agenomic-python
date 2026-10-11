"""Exception hierarchy for agenomic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


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


class ApiError(CloudError):
    def __init__(
        self,
        code: str,
        status: int,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.status = status
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    @property
    def request_id(self) -> Optional[str]:
        value = self.details.get("request_id")
        return value if isinstance(value, str) else None

    @property
    def reason(self) -> Optional[str]:
        value = self.details.get("reason")
        return value if isinstance(value, str) else None

    @property
    def errors(self) -> list[Any]:
        value = self.details.get("errors")
        return list(value) if isinstance(value, list) else []


class RedactionError(AgenomicError):
    """Redaction rule could not be applied."""
