"""Workflow and multi-agent system manifests (spec v0.2, RFC 0009).

Pydantic mirrors of `agenomic-spec` `workflow.schema.json` and
`system.schema.json`. Parse a YAML/JSON manifest yourself and feed the
resulting dict to ``WorkflowSpec.model_validate`` /
``SystemSpec.model_validate``.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKFLOW_SPEC_VERSION = "agenomic/v0.2"
SYSTEM_SPEC_VERSION = "agenomic/v0.2"

#: Reserved terminal vertex for orchestration edges.
END_VERTEX = "END"

_WORKFLOW_ID_RE = re.compile(r"^workflow://[a-z0-9-]+/[a-z0-9-]+$")
_SYSTEM_ID_RE = re.compile(r"^system://[a-z0-9-]+/[a-z0-9-]+$")
_DURATION_RE = re.compile(r"^[0-9]+(ms|s|m|h|d)$")

StepType = Literal["agent", "tool", "human", "wait", "workflow", "loop"]
OrchestrationStyle = Literal["pipeline", "graph", "supervisor", "swarm", "custom"]
OnError = Literal["fail", "continue", "escalate"]
OnTimeout = Literal["escalate", "fail", "continue"]


def _validate_duration(value: Optional[str], field_name: str) -> Optional[str]:
    if value is not None and not _DURATION_RE.match(value):
        raise ValueError(
            f"invalid {field_name}: {value!r} (expected integer + unit, e.g. 30s, 15m, 2h, 90d)"
        )
    return value


class EngineHint(BaseModel):
    """Non-normative hint about the runtime executing an orchestration."""

    model_config = ConfigDict(extra="allow")

    kind: str
    version: Optional[str] = None


class TriggerSpec(BaseModel):
    """How a workflow run (or agent invocation) starts."""

    model_config = ConfigDict(extra="allow")

    type: Literal["api", "event", "schedule", "signal", "manual"]
    description: Optional[str] = None
    event: Optional[str] = None
    schedule: Optional[str] = None
    signal: Optional[str] = None


class IoField(BaseModel):
    """Declared workflow input or output field."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    required: Optional[bool] = None


class StateRef(BaseModel):
    """Reference to a shared state schema (schema only, never data)."""

    model_config = ConfigDict(extra="allow")

    schema_path: Optional[str] = None
    schema_version: Optional[str] = None


class SignalSpec(BaseModel):
    """External signal a running workflow or system can receive."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    payload_schema_path: Optional[str] = None


class EscalationRule(BaseModel):
    """When ``condition`` holds, route the case to ``route``."""

    model_config = ConfigDict(extra="allow")

    condition: str
    route: str


class RetrySpec(BaseModel):
    """Retry behavior of a step."""

    model_config = ConfigDict(extra="allow")

    max_attempts: int = Field(ge=1)
    backoff: Optional[Literal["none", "fixed", "exponential"]] = None
    initial_interval: Optional[str] = None

    @field_validator("initial_interval")
    @classmethod
    def _check_interval(cls, v: Optional[str]) -> Optional[str]:
        return _validate_duration(v, "initial_interval")


class HumanGate(BaseModel):
    """Who must act on a `human` step, and what happens on SLA breach."""

    model_config = ConfigDict(extra="allow")

    role: str
    action: Literal["approve", "review", "input", "decide"]
    sla: Optional[str] = None
    on_timeout: Optional[OnTimeout] = None
    escalation_route: Optional[str] = None

    @field_validator("sla")
    @classmethod
    def _check_sla(cls, v: Optional[str]) -> Optional[str]:
        return _validate_duration(v, "sla")


class WaitFor(BaseModel):
    """Signals that resume a `wait` step."""

    model_config = ConfigDict(extra="allow")

    signals: list[str] = Field(min_length=1)
    mode: Optional[Literal["any", "all"]] = None
    timeout: Optional[str] = None
    on_timeout: Optional[OnTimeout] = None

    @field_validator("timeout")
    @classmethod
    def _check_timeout(cls, v: Optional[str]) -> Optional[str]:
        return _validate_duration(v, "timeout")


class ToolRef(BaseModel):
    """Deterministic tool or function invoked by a `tool` step."""

    model_config = ConfigDict(extra="allow")

    name: str
    protocol: Optional[Literal["mcp", "http", "grpc", "local"]] = None
    server: Optional[str] = None
    version: Optional[str] = None


class WorkflowStep(BaseModel):
    """One step of a workflow.

    Execution order is the DAG induced by ``depends_on``; a false ``when``
    guard skips the step.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: StepType
    name: Optional[str] = None
    description: Optional[str] = None
    agent: Optional[str] = None
    skill: Optional[str] = None
    tool: Optional[ToolRef] = None
    gate: Optional[HumanGate] = None
    wait_for: Optional[WaitFor] = None
    uses: Optional[str] = None
    body: Optional[list[WorkflowStep]] = None
    until: Optional[str] = None
    max_iterations: Optional[int] = Field(default=None, ge=1)
    depends_on: list[str] = Field(default_factory=list)
    when: Optional[str] = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    retry: Optional[RetrySpec] = None
    timeout: Optional[str] = None
    on_error: Optional[OnError] = None

    @field_validator("timeout")
    @classmethod
    def _check_timeout(cls, v: Optional[str]) -> Optional[str]:
        return _validate_duration(v, "timeout")

    @model_validator(mode="after")
    def _check_type_specific_fields(self) -> WorkflowStep:
        required_by_type: dict[str, tuple[str, ...]] = {
            "agent": ("agent",),
            "tool": ("tool",),
            "human": ("gate",),
            "wait": ("wait_for",),
            "workflow": ("uses",),
            "loop": ("body", "until"),
        }
        for field_name in required_by_type[self.type]:
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"step {self.id!r}: type {self.type!r} requires field {field_name!r}"
                )
        return self


class WorkflowIdentity(BaseModel):
    """Identity block of a workflow manifest."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    domain: str
    criticality: str
    version: Optional[str] = None
    description: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _WORKFLOW_ID_RE.match(v):
            raise ValueError(f"invalid workflow id: {v!r} (must match workflow://org/name)")
        return v


class WorkflowSpec(BaseModel):
    """Workflow manifest (``workflow.yaml``), spec v0.2 / RFC 0009.

    Example:
        >>> spec = WorkflowSpec.model_validate({
        ...     "spec_version": "agenomic/v0.2",
        ...     "workflow": {
        ...         "id": "workflow://acme/pipeline",
        ...         "name": "Pipeline",
        ...         "domain": "claims",
        ...         "criticality": "standard",
        ...     },
        ...     "steps": [
        ...         {"id": "answer", "type": "agent", "agent": "agent://acme/bot"},
        ...     ],
        ... })
        >>> spec.steps[0].id
        'answer'
    """

    model_config = ConfigDict(extra="allow")

    spec_version: str = WORKFLOW_SPEC_VERSION
    workflow: WorkflowIdentity
    engine: Optional[EngineHint] = None
    triggers: list[TriggerSpec] = Field(default_factory=list)
    inputs: list[IoField] = Field(default_factory=list)
    outputs: list[IoField] = Field(default_factory=list)
    state: Optional[StateRef] = None
    steps: list[WorkflowStep] = Field(min_length=1)
    signals: list[SignalSpec] = Field(default_factory=list)
    escalation_rules: list[EscalationRule] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_step_graph(self) -> WorkflowSpec:
        seen: set[str] = set()

        def check_level(steps: list[WorkflowStep]) -> None:
            level_ids = {step.id for step in steps}
            for step in steps:
                if step.id in seen:
                    raise ValueError(f"duplicate step id {step.id!r}")
                seen.add(step.id)
                for dep in step.depends_on:
                    if dep not in level_ids:
                        raise ValueError(
                            f"step {step.id!r} depends on unknown step {dep!r} "
                            "(must exist at the same nesting level)"
                        )
            for step in steps:
                if step.body:
                    check_level(step.body)

        check_level(self.steps)
        return self


class AutonomySpec(BaseModel):
    """Autonomy envelope: actions allowed or forbidden without escalation."""

    model_config = ConfigDict(extra="allow")

    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)


class SystemMember(BaseModel):
    """One member agent of a system: a role bound to an agent identity."""

    model_config = ConfigDict(extra="allow")

    role: str
    id: str
    genome: Optional[str] = None
    description: Optional[str] = None
    autonomy: Optional[AutonomySpec] = None


class OrchestrationEdge(BaseModel):
    """Directed hand-off between roles. ``END`` is the terminal vertex."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    when: Optional[str] = None


class OrchestrationSpec(BaseModel):
    """How the member agents of a system are composed."""

    model_config = ConfigDict(extra="allow")

    style: OrchestrationStyle
    description: Optional[str] = None
    engine: Optional[EngineHint] = None
    entrypoint: Optional[str] = None
    supervisor: Optional[str] = None
    edges: list[OrchestrationEdge] = Field(default_factory=list)


class WorkflowRef(BaseModel):
    """Workflow manifest owned by a system."""

    model_config = ConfigDict(extra="allow")

    id: str
    path: str
    description: Optional[str] = None


class SystemIdentity(BaseModel):
    """Identity block of a system manifest."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    domain: str
    criticality: str
    version: Optional[str] = None
    description: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _SYSTEM_ID_RE.match(v):
            raise ValueError(f"invalid system id: {v!r} (must match system://org/name)")
        return v


class SystemSpec(BaseModel):
    """Multi-agent system manifest (``system.yaml``), spec v0.2 / RFC 0009.

    Example:
        >>> spec = SystemSpec.model_validate({
        ...     "spec_version": "agenomic/v0.2",
        ...     "system": {
        ...         "id": "system://acme/orchestra",
        ...         "name": "Orchestra",
        ...         "domain": "claims",
        ...         "criticality": "standard",
        ...     },
        ...     "agents": [{"role": "solo", "id": "agent://acme/bot"}],
        ...     "orchestration": {"style": "pipeline", "entrypoint": "solo"},
        ... })
        >>> spec.agents[0].role
        'solo'
    """

    model_config = ConfigDict(extra="allow")

    spec_version: str = SYSTEM_SPEC_VERSION
    system: SystemIdentity
    agents: list[SystemMember] = Field(min_length=1)
    orchestration: OrchestrationSpec
    shared_state: Optional[StateRef] = None
    signals: list[SignalSpec] = Field(default_factory=list)
    workflows: list[WorkflowRef] = Field(default_factory=list)
    communication_guardrails: list[str] = Field(default_factory=list)
    escalation_rules: list[EscalationRule] = Field(default_factory=list)
    forbidden_autonomy: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_roles(self) -> SystemSpec:
        roles: set[str] = set()
        for member in self.agents:
            if member.role in roles:
                raise ValueError(f"duplicate member role {member.role!r}")
            roles.add(member.role)
        for key in ("entrypoint", "supervisor"):
            role: Any = getattr(self.orchestration, key)
            if role is not None and role not in roles:
                raise ValueError(f"orchestration.{key} references undeclared role {role!r}")
        for edge in self.orchestration.edges:
            if edge.from_ not in roles:
                raise ValueError(f"edge references undeclared role {edge.from_!r}")
            if edge.to != END_VERTEX and edge.to not in roles:
                raise ValueError(f"edge references undeclared role {edge.to!r}")
        return self

    def shadowed_allowed_actions(self) -> dict[str, list[str]]:
        """Per-role actions allowed by a member but shadowed by ``forbidden_autonomy``.

        System-wide ``forbidden_autonomy`` overrides every member's
        ``allowed_actions``; entries returned here are dead declarations
        worth a warning.
        """
        forbidden = set(self.forbidden_autonomy)
        shadowed: dict[str, list[str]] = {}
        for member in self.agents:
            if member.autonomy is None:
                continue
            hits = [a for a in member.autonomy.allowed_actions if a in forbidden]
            if hits:
                shadowed[member.role] = hits
        return shadowed
