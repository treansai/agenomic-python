# Agenomic Python SDK documentation

Python SDK for [Agenomic](https://agenomic.dev) — agent-native
versioning, tracing, and ATEP signed event logs. Fully offline; cloud
upload is optional.

Using an AI coding agent? Point it at [`AGENT.md`](../AGENT.md) — the
condensed integration guide with import maps, recipes, and pitfalls.

## Start here

- [Quickstart](quickstart.md) — install, first trace, five minutes
- [Non-determinism disclaimer](non-determinism.md) — what traces do and
  don't prove

## Guides

| topic | page |
| ----- | ---- |
| Tracing model (`TraceEnvelope`, recorder) | [tracing.md](tracing.md) |
| `@trace_agent_run` reference | [decorator.md](decorator.md) |
| Exporters (JSONL, ATEP, HTTP, fan-out) | [exporters.md](exporters.md) |
| Canonical v0.3 runs (hash-chained events) | [canonical.md](canonical.md) |
| ATEP signed event log | [atep.md](atep.md) |
| Keys, hashing & signing | [keys-and-signing.md](keys-and-signing.md) |
| Boundary redaction | [redaction.md](redaction.md) |
| Online tracking (production) | [tracking.md](tracking.md) |
| Review · Monitor · Protect | [rmp.md](rmp.md) |
| RMP benchmarks and the agent bridge | [benchmarks.md](benchmarks.md) |
| Workflow & system manifests (RFC 0009) | [orchestration.md](orchestration.md) |
| Integrations (OpenAI, Anthropic, HF, LangGraph, MCP) | [integrations.md](integrations.md) |
| Hugging Face provider | [providers/huggingface.md](providers/huggingface.md) |
| Cloud upload | [cloud-upload.md](cloud-upload.md) |
| `agenomic-py` CLI | [cli.md](cli.md) |
| Errors & logging | [errors.md](errors.md) |

## Reference

- [API reference](api-reference.md) — the complete public surface
- [`examples/`](../examples/) — nine runnable examples, offline ones run
  with zero setup
- [`CHANGELOG.md`](../CHANGELOG.md)
