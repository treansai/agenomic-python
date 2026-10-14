"""Typed mirrors of the Protect wire types used by the SDK."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ProtectOverlay(BaseModel):
    """Instruction overlay computed by the gateway for one Protect run.

    Example:
        >>> ProtectOverlay(version="1", digest="blake3:x", text="Only call allowed tools.").truncated
        False
    """

    model_config = ConfigDict(extra="allow")

    version: str
    digest: str
    text: str
    policies: list[str] = Field(default_factory=list)
    truncated: bool = False


EntitlementState = Literal["entitled", "not_entitled", "unknown"]
RequirementSource = Literal["mandatory_org", "binding", "run_binding"]
EffectiveState = Literal[
    "not_applicable", "enforced", "shadow", "exempt", "none", "blocked", "unknown"
]
ProtectionDenial = Literal[
    "protect_required:mandatory_org",
    "protect_required:binding",
    "protect_required:run_binding",
    "kill_switch",
    "restriction_active",
    "capability_denied",
    "protect_state_unavailable",
]


class ExemptionRef(BaseModel):
    """The exemption that applies to a run."""

    model_config = ConfigDict(extra="allow")

    id: str
    scope_kind: str
    scope_ref: str = ""
    expires_at: str


class EffectiveProtection(BaseModel):
    """What protects a run right now, as computed by the gateway.

    Only ``state == "enforced"`` means protected. ``shadow``, ``exempt`` and
    ``none`` are reported as such: an organization that is not entitled gets
    ``none`` and nothing is blocked. There is no ``degraded`` mode.

    Example:
        >>> protection = EffectiveProtection.model_validate({
        ...     "entitlement": "not_entitled", "org_mandatory": True, "run_mode": "live",
        ...     "protect_block": False, "state": "none"})
        >>> (protection.state, protection.is_protected)
        ('none', False)
    """

    model_config = ConfigDict(extra="allow")

    entitlement: EntitlementState
    org_mandatory: bool
    requirement: Optional[RequirementSource] = None
    run_mode: Literal["mock", "live", "hybrid"]
    protect_block: bool
    protect_mode: Optional[Literal["enforce", "shadow"]] = None
    exemption: Optional[ExemptionRef] = None
    denial: Optional[ProtectionDenial] = None
    state: EffectiveState

    @property
    def is_protected(self) -> bool:
        return self.state == "enforced"


class ProtectSettings(BaseModel):
    """Organization protect settings; ``mandatory_in_force`` is false when not entitled.

    Example:
        >>> ProtectSettings(protect_mandatory=True, mandatory_in_force=False,
        ...     entitlement="not_entitled", revision=3).mandatory_in_force
        False
    """

    model_config = ConfigDict(extra="allow")

    protect_mandatory: bool
    mandatory_in_force: bool
    entitlement: EntitlementState
    revision: int
    updated_by: Optional[str] = None
    updated_at: Optional[str] = None


class ProtectExemption(BaseModel):
    """A time-boxed exemption from mandatory protection."""

    model_config = ConfigDict(extra="allow")

    id: str
    scope_kind: Literal["org", "run", "tool"]
    scope_ref: str = ""
    reason: str
    status: str
    created_by: Optional[str] = None
    created_at: str
    expires_at: str
    lifted_by: Optional[str] = None
    lifted_at: Optional[str] = None


Overlay = Optional[Union[str, ProtectOverlay]]


def overlay_text(overlay: Overlay) -> Optional[str]:
    """Text of an overlay argument (``None`` when nothing is to be injected).

    Example:
        >>> overlay_text(ProtectOverlay(version="1", digest="d", text="Be safe."))
        'Be safe.'
        >>> overlay_text("") is None
        True
    """
    if overlay is None:
        return None
    text = overlay if isinstance(overlay, str) else overlay.text
    return text or None


__all__ = [
    "EffectiveProtection",
    "ExemptionRef",
    "Overlay",
    "ProtectExemption",
    "ProtectOverlay",
    "ProtectSettings",
    "overlay_text",
]
