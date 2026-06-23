"""OpenTelemetry **GenAI** span emission for canonical events.

OTel is the runtime-agnostic capture layer (master document §1, §14): each
canonical event also surfaces as a span carrying the GenAI semantic-convention
attributes (``gen_ai.*``). The dependency is imported lazily so the rest of the
SDK works without OpenTelemetry installed; pass a real tracer to light it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.trace import Tracer

# GenAI semantic-convention attribute keys (subset we populate).
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# Map a canonical event type → GenAI operation name.
_OPERATION: dict[str, str] = {
    "llm.requested": "chat",
    "llm.responded": "chat",
    "tool.call.proposed": "execute_tool",
    "tool.call.executed": "execute_tool",
    "tool.result.observed": "execute_tool",
}


def emit_span(tracer: Tracer | None, event_type: str, attributes: dict[str, Any]) -> None:
    """Emit a GenAI span for a canonical event, if a `tracer` is configured.

    The span name is the event type; GenAI attributes are attached, plus the
    full (already-redacted) attribute set the caller passes. A ``None`` tracer
    is a silent no-op so capture never depends on OTel being present.

    Example:
        >>> emit_span(None, "llm.requested", {})  # no tracer → no-op
    """
    if tracer is None:
        return
    attrs: dict[str, Any] = {"agenomic.event.type": event_type}
    if event_type in _OPERATION:
        attrs[GEN_AI_OPERATION_NAME] = _OPERATION[event_type]
    for key, value in attributes.items():
        if value is not None and isinstance(value, (str, bool, int, float)):
            attrs[key] = value
    with tracer.start_as_current_span(event_type) as span:
        span.set_attributes(attrs)


def llm_attributes(
    *, provider: str, model: str, input_tokens: int | None, output_tokens: int | None
) -> dict[str, Any]:
    """Build the GenAI attributes for an LLM event."""
    attrs: dict[str, Any] = {GEN_AI_SYSTEM: provider, GEN_AI_REQUEST_MODEL: model}
    if input_tokens is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    return attrs


def tool_attributes(*, tool: str) -> dict[str, Any]:
    """Build the GenAI attributes for a tool-call event."""
    return {GEN_AI_TOOL_NAME: tool}


__all__ = ["emit_span", "llm_attributes", "tool_attributes"]
