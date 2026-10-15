"""Typed envelope and errors of the tool execution namespace."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agenomic.exceptions import CloudError

TOOL_EXECUTION_SCHEMA_VERSION = "agenomic.tool_execution/v1"

ToolResultSource = Literal[
    "live",
    "recorded",
    "static",
    "scenario",
    "schema_generated",
    "plugin",
    "runtime_local",
    "unrouted",
]
ToolCallStatus = Literal["success", "error", "aborted", "timeout", "pending", "denied"]
ToolExternalState = Literal["none", "confirmed", "indeterminate"]
ToolInvocationPhase = Literal[
    "received",
    "admission_pending",
    "approval_pending",
    "denied",
    "authorized",
    "dispatching",
    "succeeded",
    "failed",
    "outcome_unknown",
]
UnresolvedReason = Literal["in_flight", "ambiguous_outcome", "attempt_in_flight"]


class ToolExecutionError(CloudError):
    """A tool-execution operation was refused.

    ``code`` carries the error code (for example ``mock_unmatched``,
    ``tool_unknown``, ``live_call_denied``, ``plan_approval_required``,
    ``env_reference_missing``); ``status`` the HTTP status, 0 when no
    request was made.

    Example:
        >>> error = ToolExecutionError("tool_unknown", "no binding", 400)
        >>> (error.code, error.status)
        ('tool_unknown', 400)
    """

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.status = status


class ToolProvenance(BaseModel):
    """Where a result came from and how faithful it is.

    Example:
        >>> ToolProvenance(source="static", binding_mode="mock").is_real
        False
    """

    model_config = ConfigDict(extra="allow")

    source: ToolResultSource
    fidelity: str = ""
    binding_mode: Literal["mock", "live"] = "mock"
    adapter: Optional[str] = None
    strategy: Optional[str] = None
    fixture_id: Optional[str] = None
    rule_id: Optional[str] = None
    destination_host: Optional[str] = None
    fault_campaign_ref: Optional[str] = None
    note: Optional[str] = None

    @property
    def is_real(self) -> bool:
        return self.source in ("live", "runtime_local")


class ToolEffect(BaseModel):
    """A declared side effect of a call.

    Example:
        >>> ToolEffect(**{"class": "read", "description": "lookup", "simulated": True}).effect_class
        'read'
    """

    model_config = ConfigDict(populate_by_name=True)

    effect_class: str = Field(alias="class")
    description: str = ""
    simulated: bool = False


class ProtectDecision(BaseModel):
    """Admission decision stamped on an invocation by the Protect gate.

    Example:
        >>> ProtectDecision(decision_id="d1", outcome="deny", effective_mode="enforce",
        ...     policy_snapshot_digest="blake3:x", evaluated_at="2026-09-14T00:00:00Z").outcome
        'deny'
    """

    model_config = ConfigDict(extra="allow")

    decision_id: str
    outcome: str
    effective_mode: str
    reason_codes: list[str] = Field(default_factory=list)
    approval_id: Optional[str] = None
    permit_ref: Optional[str] = None
    policy_snapshot_digest: str
    evaluated_at: str


class SignedPermit(BaseModel):
    """Signed execution permit returned by ``local/authorize``; forwarded verbatim.

    Example:
        >>> SignedPermit(document={"tool": "t"}, signature={"value": "s"}).document["tool"]
        't'
    """

    document: dict[str, Any]
    signature: dict[str, Any]


class ToolCallResult(BaseModel):
    """Native tool result plus the Agenomic technical envelope.

    Example:
        >>> envelope = ToolCallResult(
        ...     result={"ok": True}, record_id="rec_1", status="success",
        ...     provenance=ToolProvenance(source="static"), external_state="none",
        ... )
        >>> (envelope.ok, envelope.source, envelope.is_real)
        (True, 'static', False)
    """

    result: JsonValue = None
    record_id: str
    status: ToolCallStatus
    provenance: ToolProvenance
    external_state: ToolExternalState
    effects: list[ToolEffect] = Field(default_factory=list)
    duration_ms: int = 0
    virtual_time: Optional[str] = None
    expected_error: bool = False
    protect: Optional[ProtectDecision] = None
    approval_id: Optional[str] = None
    decision: Optional[str] = None
    transformation: Optional[dict[str, Any]] = None
    safe_explanation: Optional[str] = None
    #: False when a runtime-local function ran but its report never reached
    #: the gateway; the pending record stays indeterminate server-side.
    reported: bool = True
    phase: Optional[ToolInvocationPhase] = None

    @property
    def source(self) -> ToolResultSource:
        return self.provenance.source

    @property
    def is_real(self) -> bool:
        return self.provenance.is_real

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @classmethod
    def from_response(cls, body: Mapping[str, Any]) -> ToolCallResult:
        """Validate an invoke response ``{result, agenomic: {...}}``.

        Example:
            >>> ToolCallResult.from_response({"result": 1, "agenomic": {
            ...     "record_id": "r", "status": "success",
            ...     "provenance": {"source": "static"}, "external_state": "none"}}).result
            1
        """
        env = body.get("agenomic")
        if not isinstance(env, Mapping):
            raise ToolExecutionError(
                "invalid_response", "invoke response has no agenomic envelope", 0
            )
        try:
            return cls.model_validate({**dict(env), "result": body.get("result")})
        except ValidationError as error:
            raise ToolExecutionError(
                "invalid_response",
                f"invoke envelope is malformed: {error.error_count()} field error(s)",
                0,
            ) from error


class ToolCallError(CloudError):
    """The tool answered with an error outcome (business, protocol, timeout).

    Example:
        >>> envelope = ToolCallResult(
        ...     result={"error": {"code": "not_found"}}, record_id="r", status="error",
        ...     provenance=ToolProvenance(source="scenario"), external_state="none",
        ... )
        >>> ToolCallError("tickets.get", envelope).code
        'not_found'
    """

    def __init__(self, tool: str, envelope: ToolCallResult) -> None:
        detail = envelope.result.get("error") if isinstance(envelope.result, dict) else None
        code = detail.get("code") if isinstance(detail, dict) else None
        self.code = str(code) if code is not None else envelope.status
        super().__init__(f"tool {tool} returned {envelope.status} ({self.code})")
        self.tool = tool
        self.envelope = envelope


class ToolCallDenied(ToolExecutionError):  # noqa: N818
    """The Protect gate refused the call; nothing executed.

    ``code`` is ``policy_denied`` for a gateway refusal and the approval
    status (``rejected``, ``expired``) when a resumed approval was not granted.

    Example:
        >>> envelope = ToolCallResult(record_id="r", status="denied",
        ...     provenance=ToolProvenance(source="unrouted"), external_state="none",
        ...     safe_explanation="credit limit changes need a reviewer")
        >>> error = ToolCallDenied("crm.update_customer", envelope)
        >>> (error.code, error.status, error.tool)
        ('policy_denied', 403, 'crm.update_customer')
    """

    def __init__(self, tool: str, envelope: ToolCallResult, *, code: str = "policy_denied") -> None:
        reasons = envelope.protect.reason_codes if envelope.protect else []
        detail = envelope.safe_explanation or ", ".join(reasons) or envelope.status
        super().__init__(code, f"tool {tool} was not admitted: {detail}", 403)
        self.tool = tool
        self.envelope = envelope
        self.decision = envelope.protect
        self.transformation = envelope.transformation


class ToolApprovalPending(ToolExecutionError):  # noqa: N818
    """The call waits for a human approval; nothing executed yet.

    Example:
        >>> envelope = ToolCallResult(record_id="r", status="pending",
        ...     provenance=ToolProvenance(source="unrouted"), external_state="none")
        >>> error = ToolApprovalPending("payments.refund", "apr_1", "r", envelope)
        >>> (error.code, error.status, error.approval_id)
        ('approval_pending', 202, 'apr_1')
    """

    def __init__(
        self, tool: str, approval_id: str, record_id: str, envelope: ToolCallResult
    ) -> None:
        super().__init__("approval_pending", f"tool {tool} waits for approval {approval_id}", 202)
        self.tool = tool
        self.approval_id = approval_id
        self.record_id = record_id
        self.envelope = envelope
        self.decision = envelope.protect


class InvocationStatus(BaseModel):
    """Durable state of one invocation as read from the Tool Gateway.

    ``requires_reconciliation`` is true while the outcome is unknown and an
    operator has not resolved it.

    Example:
        >>> status = InvocationStatus(
        ...     record_id="r", run_id="u", phase="outcome_unknown", status="error",
        ...     external_state="indeterminate", tool="crm.update", logical_call_id="c1",
        ...     repetition=1, attempt=1, requires_reconciliation=True,
        ...     started_at="2026-10-15T00:00:00Z")
        >>> (status.phase, status.requires_reconciliation)
        ('outcome_unknown', True)
    """

    model_config = ConfigDict(extra="allow")

    record_id: str
    run_id: str
    phase: ToolInvocationPhase
    status: ToolCallStatus
    external_state: ToolExternalState
    tool: str
    logical_call_id: str
    repetition: int
    attempt: int
    error_code: Optional[str] = None
    requires_reconciliation: bool = False
    protect: Optional[ProtectDecision] = None
    approval_id: Optional[str] = None
    reconciliation: Optional[dict[str, Any]] = None
    started_at: str
    phase_at: Optional[str] = None


class ToolCallUnresolved(ToolExecutionError):  # noqa: N818
    """The gateway refused to run a call again because its identity is unresolved.

    HTTP 409 in two wire forms, both raised as this error and never retried
    (re-executing could repeat a side effect):

    - the identity itself is recorded: ``error.code`` is ``conflict``,
      ``error.reason`` is ``in_flight`` (still being processed: read
      :meth:`ToolsResource.get_invocation` instead of retrying) or
      ``ambiguous_outcome`` (may have run, needs operator reconciliation), and
      an ``agenomic`` block carries the :class:`InvocationStatus`;
    - a new attempt of a logical call whose other attempt is unresolved:
      ``error.code`` is ``attempt_in_flight`` (wait, it settles by itself) or
      ``ambiguous_outcome`` (an operator must resolve the other attempt), with
      no ``reason`` and no ``agenomic`` block, so :attr:`invocation` and
      :attr:`record_id` are ``None`` and :attr:`other_attempt` is true.

    Example:
        >>> error = ToolCallUnresolved.from_body("crm.update", {
        ...     "error": {"code": "conflict", "reason": "ambiguous_outcome", "message": "m"},
        ...     "agenomic": {"record_id": "r", "run_id": "u", "phase": "outcome_unknown",
        ...         "status": "error", "external_state": "indeterminate", "tool": "crm.update",
        ...         "logical_call_id": "c1", "repetition": 1, "attempt": 1,
        ...         "requires_reconciliation": True, "started_at": "2026-10-15T00:00:00Z"}})
        >>> (error.reason, error.requires_reconciliation, error.record_id)
        ('ambiguous_outcome', True, 'r')
        >>> other = ToolCallUnresolved.from_body("crm.update", {
        ...     "error": {"code": "attempt_in_flight", "message": "m"}})
        >>> (other.code, other.reason, other.other_attempt, other.record_id)
        ('attempt_in_flight', 'attempt_in_flight', True, None)
    """

    def __init__(
        self,
        tool: str,
        reason: UnresolvedReason,
        message: str,
        invocation: Optional[InvocationStatus] = None,
        *,
        other_attempt: bool = False,
    ) -> None:
        code = reason if other_attempt else "conflict"
        super().__init__(code, f"{reason}: tool {tool}: {message}", 409)
        self.tool = tool
        self.reason: UnresolvedReason = reason
        self.invocation = invocation
        self.other_attempt = other_attempt

    @property
    def record_id(self) -> Optional[str]:
        return self.invocation.record_id if self.invocation else None

    @property
    def requires_reconciliation(self) -> bool:
        return self.reason == "ambiguous_outcome"

    @classmethod
    def from_body(cls, tool: str, body: Mapping[str, Any]) -> Optional[ToolCallUnresolved]:
        """Build the error from a 409 body of either form, or ``None`` for any other conflict."""
        error = body.get("error")
        if not isinstance(error, Mapping):
            return None
        message = str(error.get("message", ""))
        reason = error.get("reason")
        if reason is None:
            code = error.get("code")
            if code == "attempt_in_flight" or code == "ambiguous_outcome":
                return cls(tool, code, message, other_attempt=True)
            return None
        if reason not in ("in_flight", "ambiguous_outcome"):
            return None
        status = body.get("agenomic")
        invocation: Optional[InvocationStatus] = None
        if isinstance(status, Mapping):
            try:
                invocation = InvocationStatus.model_validate(status)
            except ValidationError:
                invocation = None
        return cls(tool, reason, message, invocation)
