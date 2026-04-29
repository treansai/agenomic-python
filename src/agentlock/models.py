from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from agentlock.utils import to_jsonable, utc_now


class AgentLockModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TraceEvent(AgentLockModel):
    event_type: str
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        normalized = to_jsonable(value)
        if not isinstance(normalized, dict):
            msg = "metadata must be a mapping"
            raise ValueError(msg)
        return normalized


class AgentRun(TraceEvent):
    event_type: Literal["agent.run"] = "agent.run"
    trace_id: str
    run_id: str
    agent_id: str
    release: str = "dev"
    input_hash: str | None = None
    input: Any | None = None
    labels: list[str] = Field(default_factory=list)

    @field_validator("input", mode="before")
    @classmethod
    def _validate_input(cls, value: Any) -> Any:
        return to_jsonable(value)


class ModelCall(TraceEvent):
    event_type: Literal["model.call"] = "model.call"
    name: str
    provider: str | None = None
    request: Any | None = None
    response: Any | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    success: bool = True

    @field_validator("request", "response", mode="before")
    @classmethod
    def _validate_payloads(cls, value: Any) -> Any:
        return to_jsonable(value)


class ToolCall(TraceEvent):
    event_type: Literal["tool.call"] = "tool.call"
    name: str
    input: Any | None = None
    output: Any | None = None
    success: bool = True
    latency_ms: float | None = Field(default=None, ge=0)

    @field_validator("input", "output", mode="before")
    @classmethod
    def _validate_payloads(cls, value: Any) -> Any:
        return to_jsonable(value)


class MemoryAccess(TraceEvent):
    event_type: Literal["memory.access"] = "memory.access"
    store: str
    operation: Literal["read", "write", "delete", "search"]
    key: str | None = None
    value: Any | None = None
    success: bool = True

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        return to_jsonable(value)


class PolicyCheck(TraceEvent):
    event_type: Literal["policy.check"] = "policy.check"
    name: str
    passed: bool
    details: Any | None = None

    @field_validator("details", mode="before")
    @classmethod
    def _validate_details(cls, value: Any) -> Any:
        return to_jsonable(value)


class HumanFeedback(TraceEvent):
    event_type: Literal["human.feedback"] = "human.feedback"
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None
    reviewer: str | None = None


class RunCompleted(TraceEvent):
    event_type: Literal["run.completed"] = "run.completed"
    success: bool
    final_output_hash: str | None = None
    error: str | None = None


TraceEventUnion = Annotated[
    AgentRun
    | ModelCall
    | ToolCall
    | MemoryAccess
    | PolicyCheck
    | HumanFeedback
    | RunCompleted,
    Field(discriminator="event_type"),
]

TraceEventAdapter = TypeAdapter(TraceEventUnion)


class TraceEnvelope(AgentLockModel):
    trace_id: str
    run_id: str
    agent_id: str
    release: str = "dev"
    timestamp: datetime = Field(default_factory=utc_now)
    input: Any | None = None
    model_calls: list[ModelCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_output: Any | None = None
    labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[TraceEventUnion] = Field(default_factory=list)
    memory_accesses: list[MemoryAccess] = Field(default_factory=list)
    policy_checks: list[PolicyCheck] = Field(default_factory=list)
    human_feedback: list[HumanFeedback] = Field(default_factory=list)
    run_completed: RunCompleted | None = None

    @field_validator("input", "final_output", mode="before")
    @classmethod
    def _validate_payloads(cls, value: Any) -> Any:
        return to_jsonable(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        normalized = to_jsonable(value)
        if not isinstance(normalized, dict):
            msg = "metadata must be a mapping"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def _populate_events(self) -> TraceEnvelope:
        if self.events:
            return self

        events: list[TraceEventUnion] = [
            AgentRun(
                trace_id=self.trace_id,
                run_id=self.run_id,
                agent_id=self.agent_id,
                release=self.release,
                input_hash=_string_or_none(self.metadata.get("input_hash")),
                input=self.input,
                labels=self.labels,
            )
        ]
        events.extend(self.model_calls)
        events.extend(self.tool_calls)
        events.extend(self.memory_accesses)
        events.extend(self.policy_checks)
        events.extend(self.human_feedback)

        if self.run_completed is not None:
            events.append(self.run_completed)
        elif self.final_output is not None or isinstance(self.metadata.get("success"), bool):
            events.append(
                RunCompleted(
                    success=_bool_or_default(self.metadata.get("success"), default=True),
                    final_output_hash=_string_or_none(self.metadata.get("output_hash")),
                    error=_string_or_none(self.metadata.get("error_type")),
                )
            )

        self.events = events
        return self


def validate_trace_event(value: Any) -> TraceEventUnion:
    return TraceEventAdapter.validate_python(value)


def validate_trace_envelope(value: Any) -> TraceEnvelope:
    return TraceEnvelope.model_validate(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
