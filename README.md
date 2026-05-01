# agentlock-python

[![CI](https://github.com/agentlock/agentlock-python/actions/workflows/ci.yml/badge.svg)](https://github.com/agentlock/agentlock-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentlock.svg)](https://pypi.org/project/agentlock/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python SDK for [AgentLock](https://agentlock.dev) — agent-native versioning, tracing, and ATEP signed event logs.

Works fully offline. Cloud upload is optional.

## Install

```bash
pip install agentlock
```

Optional integrations:

```bash
pip install "agentlock[openai]"      # OpenAI auto-instrumentation
pip install "agentlock[anthropic]"   # Anthropic auto-instrumentation
pip install "agentlock[langgraph]"   # LangGraph state-graph tracing
pip install "agentlock[all]"         # Everything
```

## Quickstart

```python
from agentlock.trace.decorator import trace_agent_run
from agentlock.exporters.jsonl import JsonlExporter

with JsonlExporter("traces.jsonl") as exporter:

    @trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
    def handle(query: str) -> dict:
        return {"answer": query.upper()}

    handle("hello")
```

That writes a signed-ready `TraceEnvelope` to `traces.jsonl`. No network. No
account required.

## What's in the box

- `@trace_agent_run` decorator (sync + async) with contextvar-based propagation
- `TraceEnvelope` pydantic v2 models compatible with `agentlock-spec`
- ATEP segment writer/reader with BLAKE3 causal hashes and ed25519 signatures
- Boundary redaction (REMOVE / MASK / HASH / TRUNCATE) with dotted paths
- Exporters: JSONL, ATEP local, HTTP batched, multi fan-out
- Optional, lazy-imported integrations: OpenAI, Anthropic, LangGraph, MCP
- Async-first cloud client with idempotency keys + retry

## Documentation

- [Quickstart](docs/quickstart.md)
- [Tracing](docs/tracing.md) · [Decorator](docs/decorator.md)
- [ATEP](docs/atep.md) · [Redaction](docs/redaction.md)
- [Integrations](docs/integrations.md) · [Cloud upload](docs/cloud-upload.md)
- [Non-determinism disclaimer](docs/non-determinism.md)

## Examples

See [`examples/`](examples/) — minimal trace, decorator + JSONL, ATEP local,
OpenAI traced, LangGraph traced, cloud upload, and a full offline signed
release.

## License

Apache-2.0. See [LICENSE](LICENSE).
