# Quickstart

Install:

```bash
pip install agenomic
```

Decorate any function and write traces to a JSONL file — fully offline:

```python
from agenomic.trace.decorator import trace_agent_run
from agenomic.exporters.jsonl import JsonlExporter

with JsonlExporter("traces.jsonl") as exporter:

    @trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
    def handle(query: str) -> dict:
        return {"answer": query.upper()}

    handle("hello")
```

Each call appends one signed-ready `TraceEnvelope` JSON line to `traces.jsonl`.
Verify with `agenomic-py` (this package) or `agenomic-cli` (Rust).

## Optional integrations

```bash
pip install "agenomic[openai]"      # OpenAI auto-instrumentation
pip install "agenomic[anthropic]"   # Anthropic auto-instrumentation
pip install "agenomic[langgraph]"   # LangGraph state-graph tracing
pip install "agenomic[all]"         # Everything
```

The integrations import their underlying SDKs lazily, so importing
`agenomic.integrations.openai` is safe even without `openai` installed.

## Next

- [Tracing](tracing.md) — what `TraceEnvelope` captures
- [Decorator reference](decorator.md)
- [ATEP](atep.md) — signed event log format
- [Redaction](redaction.md)
- [Integrations](integrations.md)
- [Cloud upload](cloud-upload.md)
- [Non-determinism disclaimer](non-determinism.md)
