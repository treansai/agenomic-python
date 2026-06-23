"""Canonical Trace SDK — events surface as OpenTelemetry GenAI spans."""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agenomic.canonical import start_run


def test_genai_spans_carry_gen_ai_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agenomic-test")

    run = start_run("agent://acme/x", tracer=tracer, provider="openai", model="gpt-4o")
    run.log_llm(prompt="hi", response="ok", input_tokens=5, output_tokens=2)
    run.log_tool_call(tool="search", arguments={"q": "x"}, result=[1])
    run.complete_run(output={})

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "llm.requested" in spans
    assert "tool.call.executed" in spans

    llm = spans["llm.requested"]
    assert llm.attributes["gen_ai.system"] == "openai"
    assert llm.attributes["gen_ai.request.model"] == "gpt-4o"
    assert llm.attributes["gen_ai.operation.name"] == "chat"
    assert llm.attributes["gen_ai.usage.input_tokens"] == 5

    tool = spans["tool.call.executed"]
    assert tool.attributes["gen_ai.tool.name"] == "search"


def test_no_tracer_is_a_silent_noop() -> None:
    # Capture must not depend on OTel being wired.
    run = start_run("agent://acme/x")  # no tracer
    run.log_llm(prompt="hi", response="ok")
    trace = run.complete_run(output={})
    assert any(e["type"] == "llm.requested" for e in trace["events"])
