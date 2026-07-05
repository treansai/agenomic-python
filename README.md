# agenomic-python

[![CI](https://github.com/agenomic/agenomic-python/actions/workflows/ci.yml/badge.svg)](https://github.com/agenomic/agenomic-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agenomic.svg)](https://pypi.org/project/agenomic/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python SDK for [Agenomic](https://agenomic.dev) — agent-native versioning, tracing, and ATEP signed event logs.

Works fully offline. Cloud upload is optional.

## Install

```bash
pip install agenomic
```

Optional integrations:

```bash
pip install "agenomic[openai]"      # OpenAI auto-instrumentation
pip install "agenomic[anthropic]"   # Anthropic auto-instrumentation
pip install "agenomic[huggingface]" # Hugging Face Hub + Inference
pip install "agenomic[langgraph]"   # LangGraph state-graph tracing
pip install "agenomic[all]"         # Everything
```

## Quickstart

```python
from agenomic.trace.decorator import trace_agent_run
from agenomic.exporters.jsonl import JsonlExporter

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
- `TraceEnvelope` pydantic v2 models compatible with `agenomic-spec`
- `WorkflowSpec` / `SystemSpec` pydantic v2 models for the v0.2 workflow and
  multi-agent system manifests (RFC 0009), with step-graph and role checks
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
- [Hugging Face provider](docs/providers/huggingface.md)
- [Non-determinism disclaimer](docs/non-determinism.md)

## Examples

See [`examples/`](examples/) — minimal trace, decorator + JSONL, ATEP local,
OpenAI traced, LangGraph traced, cloud upload, and a full offline signed
release.

## License

Apache-2.0. See [LICENSE](LICENSE).
