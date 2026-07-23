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
- [Exporters](exporters.md) — JSONL, ATEP, HTTP batched, fan-out
- [Canonical runs](canonical.md) — v0.3 hash-chained event streams
- [ATEP](atep.md) — signed event log format
- [Keys & signing](keys-and-signing.md)
- [Redaction](redaction.md)
- [Online tracking](tracking.md) · [RMP](rmp.md)
- [Integrations](integrations.md)
- [Cloud upload](cloud-upload.md)
- [CLI](cli.md) · [Errors](errors.md)
- [API reference](api-reference.md)
- [Non-determinism disclaimer](non-determinism.md)

Integrating with an AI coding agent? See [`AGENT.md`](../AGENT.md).
