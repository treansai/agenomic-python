"""Typed mirrors of the Protect wire types used by the SDK."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

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


ActionFamily = Literal[
    "data_egress",
    "financial",
    "identity_change",
    "code_deploy",
    "destructive_write",
    "external_message",
]
FamilyRiskTier = Literal["elevated", "high", "critical"]
EffectClass = Literal["read", "reversible_write", "irreversible_write", "unknown"]
PartyKind = Literal["user", "api_key", "agent"]


class Party(BaseModel):
    """A principal in a delegation chain: a user or API key at the root, agents below.

    Example:
        >>> Party(kind="agent", id="support-bot").kind
        'agent'
    """

    model_config = ConfigDict(extra="allow")

    kind: PartyKind
    id: str


class DelegationScope(BaseModel):
    """What a delegation lets the delegate do; it can only narrow along a chain.

    ``tools`` are exact ids or a trailing ``*`` pattern (``model:openai/*``
    covers model calls). An empty ``families`` list delegates no classified
    call at all. ``effect_ceiling`` ``unknown`` is refused.

    Example:
        >>> DelegationScope(tools=["crm.*"], families=["financial"],
        ...     effect_ceiling="reversible_write").families
        ['financial']
    """

    tools: list[str]
    families: list[ActionFamily] = Field(default_factory=list)
    effect_ceiling: EffectClass


class DelegationBudget(BaseModel):
    """Action budget of one link, shared by every run that declares it.

    Example:
        >>> DelegationBudget(max_actions=20).max_actions
        20
    """

    max_actions: int


class DelegationLink(BaseModel):
    """The signed document of one delegation link.

    Example:
        >>> link = DelegationLink.model_validate({
        ...     "schema_version": "agenomic.protect.delegation/v1", "id": "d1", "org_id": "o",
        ...     "issued_by": {"kind": "user", "id": "u"}, "delegator": {"kind": "user", "id": "u"},
        ...     "delegate": {"kind": "agent", "id": "a"},
        ...     "scope": {"tools": ["*"], "effect_ceiling": "read"}, "budget": {"max_actions": 1},
        ...     "issued_at": "2026-10-15T00:00:00Z", "expires_at": "2026-10-16T00:00:00Z",
        ...     "issuer": {"key_id": "k", "algorithm": "ed25519"}})
        >>> (link.parent_id, link.delegate.id)
        (None, 'a')
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str
    id: str
    org_id: str
    parent_id: Optional[str] = None
    parent_ref: Optional[str] = None
    issued_by: Party
    delegator: Party
    delegate: Party
    scope: DelegationScope
    budget: DelegationBudget
    issued_at: str
    expires_at: str
    issuer: dict[str, Any]


class SignedDelegation(BaseModel):
    """A delegation link with the organization signature; the SDK never verifies it.

    The gateway verifies every signature of the chain at each admission. A run
    opts in with ``protect.delegation_id`` (the last link) in its run config;
    a run that does not declare it is not attenuated.

    Example:
        >>> SignedDelegation.model_validate({"document": {
        ...     "schema_version": "v", "id": "d1", "org_id": "o",
        ...     "issued_by": {"kind": "user", "id": "u"}, "delegator": {"kind": "user", "id": "u"},
        ...     "delegate": {"kind": "agent", "id": "a"},
        ...     "scope": {"tools": ["*"], "effect_ceiling": "read"}, "budget": {"max_actions": 1},
        ...     "issued_at": "t", "expires_at": "t", "issuer": {}},
        ...     "signature": {"algorithm": "ed25519", "value": "s"}}).document.id
        'd1'
    """

    document: DelegationLink
    signature: dict[str, Any]


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
    "ActionFamily",
    "DelegationBudget",
    "DelegationLink",
    "DelegationScope",
    "EffectClass",
    "EffectiveProtection",
    "FamilyRiskTier",
    "Party",
    "PartyKind",
    "SignedDelegation",
    "ExemptionRef",
    "Overlay",
    "ProtectExemption",
    "ProtectOverlay",
    "ProtectSettings",
    "overlay_text",
]
