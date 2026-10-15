"""Typed mirrors of the correction loop wire types (agenomic-cloud ``/v1/correction``)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

IncidentSource = Literal["protect_decision", "rmp_finding", "failed_run"]
IncidentStatus = Literal["open", "scenario_ready", "candidate_proposed", "resolved", "dismissed"]
CandidateKind = Literal["prompt_change", "policy_change", "tool_contract_change", "config_change"]
CandidateStatus = Literal[
    "proposed",
    "verified",
    "reviewed",
    "approved",
    "staged",
    "promoted",
    "rolled_back",
    "rejected",
]
Weakening = Literal["none", "weakening", "unknown"]
Verdict = Literal["passed", "failed", "inconclusive"]
Severity = Literal["info", "low", "medium", "high", "critical"]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="allow")


class Incident(_Wire):
    """Something Protect or RMP already observed, deduplicated by the gateway.

    Example:
        >>> Incident.model_validate({"id": "i", "org_id": "o", "source": "protect_decision",
        ...     "source_ref": "d", "agent_id": "a", "outcome": "deny", "severity": "high",
        ...     "title": "t", "dedup_key": "k", "occurrence_count": 1, "status": "open",
        ...     "opened_by": "user", "first_seen_at": "t", "last_seen_at": "t",
        ...     "updated_at": "t"}).status
        'open'
    """

    id: str
    org_id: str
    source: IncidentSource
    source_ref: str
    agent_id: str
    tool: Optional[str] = None
    outcome: str
    effective_mode: Optional[str] = None
    severity: Severity
    title: str
    reason_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    dedup_key: str
    occurrence_count: int
    status: IncidentStatus
    proposal_id: Optional[str] = None
    fixture_set_ref: Optional[str] = None
    opened_by: str
    suggestion: Optional[dict[str, Any]] = None
    first_seen_at: str
    last_seen_at: str
    updated_at: str


class OpenedIncident(_Wire):
    """Result of opening an incident: ``created`` is false when it joined a live one.

    Example:
        >>> OpenedIncident.model_validate({"created": False, "incident": {
        ...     "id": "i", "org_id": "o", "source": "failed_run", "source_ref": "r",
        ...     "agent_id": "a", "outcome": "error", "severity": "low", "title": "t",
        ...     "dedup_key": "k", "occurrence_count": 2, "status": "open", "opened_by": "user",
        ...     "first_seen_at": "t", "last_seen_at": "t", "updated_at": "t"}}).created
        False
    """

    incident: Incident
    created: bool


class Candidate(_Wire):
    """A proposed fix: immutable payload bound to ``content_digest``.

    Example:
        >>> Candidate.model_validate({"id": "c", "org_id": "o", "incident_id": "i",
        ...     "kind": "policy_change", "title": "t", "rationale": "", "payload": {},
        ...     "content_digest": "sha256:x", "scenario_digest": "sha256:y", "proposal_id": "p",
        ...     "claims_weakening": False, "weakening": "none", "weakening_acknowledged": False,
        ...     "status": "proposed", "created_at": "t", "updated_at": "t"}).status
        'proposed'
    """

    id: str
    org_id: str
    incident_id: str
    kind: CandidateKind
    title: str
    rationale: str = ""
    payload: dict[str, Any]
    content_digest: str
    scenario_digest: str
    proposal_id: str
    claims_weakening: bool
    weakening: Weakening
    weakening_reasons: list[str] = Field(default_factory=list)
    weakening_acknowledged: bool
    status: CandidateStatus
    replay_report_id: Optional[str] = None
    staged_at: Optional[str] = None
    promotion_ref: Optional[str] = None
    applied: Optional[dict[str, Any]] = None
    rollback_ref: Optional[str] = None
    created_at: str
    updated_at: str


class Transition(_Wire):
    """One append-only state change or verification attempt (``system`` role for the trigger)."""

    id: str
    entity: str
    entity_id: str
    from_status: str
    to_status: str
    actor_role: str
    content_digest: Optional[str] = None
    scenario_digest: Optional[str] = None
    acknowledged_weakening: bool = False
    note: Optional[str] = None
    evidence: Any = None
    created_at: str


class ReplayReport(_Wire):
    """Append-only comparative replay report, bound to a candidate digest.

    ``report`` holds the per-scenario results and the summary; the verdict is
    computed by the gateway.

    Example:
        >>> ReplayReport.model_validate({"id": "r", "candidate_id": "c", "content_digest": "d",
        ...     "scenario_digest": "s", "bindings_digest": "b", "report_digest": "x",
        ...     "verdict": "inconclusive", "report": {}, "created_at": "t"}).verdict
        'inconclusive'
    """

    id: str
    candidate_id: str
    content_digest: str
    scenario_digest: str
    bindings_digest: str
    report_digest: str
    verdict: Verdict
    report: dict[str, Any]
    created_at: str


class VerifyOutcome(_Wire):
    """A verification attempt: the candidate afterwards plus the report it produced."""

    candidate: Candidate
    report: ReplayReport


class CorrectionSettings(_Wire):
    """Per-organization trigger and soak settings (owner only to write).

    Example:
        >>> CorrectionSettings().deny_threshold
        5
    """

    trigger_enabled: bool = False
    deny_threshold: int = 5
    window_secs: int = 3600
    max_incidents_per_hour: int = 3
    min_shadow_secs: int = 300


class SweepOutcome(_Wire):
    """Result of a trigger sweep; it never creates a candidate."""

    enabled: bool
    groups_over_threshold: int
    opened: list[str] = Field(default_factory=list)
    rate_limited: int = 0


class PromptChange(BaseModel):
    """Payload of a ``prompt_change`` candidate (references an existing prompt version)."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    base_version: Optional[int] = None
    candidate_version: int


class PolicyChange(BaseModel):
    """Payload of a ``policy_change`` candidate; ``scope_ref`` is empty exactly for ``org``.

    Only this kind has a replay engine and a promotion path.

    Example:
        >>> PolicyChange(policy_id="p", candidate_version="2", scope_kind="org",
        ...     mode="shadow").model_dump(exclude_none=True)["mode"]
        'shadow'
    """

    model_config = ConfigDict(extra="forbid")

    policy_id: str
    candidate_version: str
    scope_kind: Literal["org", "environment", "agent", "tool_contract", "run"]
    scope_ref: str = ""
    mode: Literal["enforce", "shadow"]
    replaces_binding_id: Optional[str] = None


class ToolContractChange(BaseModel):
    """Payload of a ``tool_contract_change`` candidate."""

    model_config = ConfigDict(extra="forbid")

    contract_name: str
    base_version: Optional[int] = None
    candidate_version: int


class ConfigChange(BaseModel):
    """Payload of a ``config_change`` candidate."""

    model_config = ConfigDict(extra="forbid")

    key: str
    proposed: Any
    base: Any = None


__all__ = [
    "Candidate",
    "CandidateKind",
    "CandidateStatus",
    "ConfigChange",
    "CorrectionSettings",
    "Incident",
    "IncidentSource",
    "IncidentStatus",
    "OpenedIncident",
    "PolicyChange",
    "PromptChange",
    "ReplayReport",
    "Severity",
    "SweepOutcome",
    "ToolContractChange",
    "Transition",
    "Verdict",
    "VerifyOutcome",
    "Weakening",
]
