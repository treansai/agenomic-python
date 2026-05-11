"""Attestation models for signed releases and ATEP events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReleaseAttestation(BaseModel):
    """Signed attestation linking a release to a bundle hash and an ATEP root hash.

    Example:
        >>> ReleaseAttestation(
        ...     agent_id="agent://acme/demo",
        ...     release_id="rel-1",
        ...     bundle_hash="00" * 32,
        ...     atep_root_hash="11" * 32,
        ...     signer_key_id="abcd1234",
        ...     signature_hex="ff" * 64,
        ... ).schema_version
        'agenomic-attestation/v0.1'
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = "agenomic-attestation/v0.1"
    agent_id: str
    release_id: str
    bundle_hash: str
    atep_root_hash: str
    issued_at: datetime = Field(default_factory=_utc_now)
    signer_key_id: str
    signature_hex: str
    notes: Optional[str] = None
