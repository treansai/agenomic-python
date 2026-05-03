# Quickstart

Install:

```bash
pip install agentlock
```

Decorate any function and write traces to a JSONL file — fully offline:

```python
from agentlock.trace.decorator import trace_agent_run
from agentlock.exporters.jsonl import JsonlExporter

with JsonlExporter("traces.jsonl") as exporter:

    @trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
    def handle(query: str) -> dict:
        return {"answer": query.upper()}

    handle("hello")
```

Each call appends one signed-ready `TraceEnvelope` JSON line to `traces.jsonl`.
Verify with `agentlock-py` (this package) or `agentlock-cli` (Rust).

## Optional integrations

```bash
pip install "agentlock[openai]"      # OpenAI auto-instrumentation
pip install "agentlock[anthropic]"   # Anthropic auto-instrumentation
pip install "agentlock[langgraph]"   # LangGraph state-graph tracing
pip install "agentlock[all]"         # Everything
```

The integrations import their underlying SDKs lazily, so importing
`agentlock.integrations.openai` is safe even without `openai` installed.

## Next

- [Tracing](tracing.md) — what `TraceEnvelope` captures
- [Decorator reference](decorator.md)
- [ATEP](atep.md) — signed event log format
- [Redaction](redaction.md)
- [Integrations](integrations.md)
- [Cloud upload](cloud-upload.md)
- [Non-determinism disclaimer](non-determinism.md)
