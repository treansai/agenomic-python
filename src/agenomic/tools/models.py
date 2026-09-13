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
ToolCallStatus = Literal["success", "error", "aborted", "timeout", "pending"]
ToolExternalState = Literal["none", "confirmed", "indeterminate"]


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

    result: JsonValue
    record_id: str
    status: ToolCallStatus
    provenance: ToolProvenance
    external_state: ToolExternalState
    effects: list[ToolEffect] = Field(default_factory=list)
    duration_ms: int = 0
    virtual_time: Optional[str] = None
    expected_error: bool = False
    #: False when a runtime-local function ran but its report never reached
    #: the gateway; the pending record stays indeterminate server-side.
    reported: bool = True

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
