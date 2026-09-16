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

Full index at [docs/](docs/README.md).

- [Quickstart](docs/quickstart.md) · [API reference](docs/api-reference.md)
- [Tracing](docs/tracing.md) · [Decorator](docs/decorator.md) · [Exporters](docs/exporters.md)
- [Canonical v0.3 runs](docs/canonical.md) · [Online tracking](docs/tracking.md) · [RMP](docs/rmp.md)
- [ATEP](docs/atep.md) · [Keys & signing](docs/keys-and-signing.md) · [Redaction](docs/redaction.md)
- [Integrations](docs/integrations.md) · [Cloud upload](docs/cloud-upload.md)
- [Workflow & system manifests](docs/orchestration.md)
- [Hugging Face provider](docs/providers/huggingface.md)
- [CLI](docs/cli.md) · [Errors](docs/errors.md)
- [Non-determinism disclaimer](docs/non-determinism.md)

Using an AI coding agent (Claude Code, Cursor, Copilot)? Point it at
[`AGENT.md`](AGENT.md) — a condensed SDK guide written for agents:
import maps, canonical recipes, and pitfalls.

## Tool execution for replays (Tool Gateway and Tool Mock Engine)

In cloud mode, `client.tools` routes each tool call of a replay to a real
backend (credentials resolved server-side from `${env:VAR_NAME}` references)
or to the Tool Mock Engine, chosen per tool by an explicit configuration.
Nothing falls back to a real call: an unknown tool or a missing fixture is an
error.

```python
from agenomic import Client
from agenomic.tools import ToolCallError

tools = Client(api_key="agm_...", base_url="https://cloud.example").tools
plan = tools.preflight(config_text=open("tool_execution.yaml").read(), repetitions=3)
run = tools.create_run(name="hybrid", config_text=open("tool_execution.yaml").read(), repetitions=3)
if run["status"] == "planned":
    run = tools.approve_run(run["id"], plan_hash=run["plan_hash"])
tools.start_run(run["id"])

router = tools.router(run["id"], repetition=1)
customer = router.call("crm.get_customer", {"id": "c_1"})   # live or mock, per binding
try:
    router.call("email.send", {"to": "ops@example.test"})
except ToolCallError as error:
    print(error.code, error.envelope.provenance)
print(router.summary())   # {'calls': 2, 'by_source': {...}, 'has_real_calls': ..., 'unreported': 0}
tools.complete_run(run["id"])
```

Functions passed as `local_functions` never run before the engine allows
them: the router calls `local/authorize` first (budget reserved, pending
record), executes only on a `local` decision that carries a record id,
routes the call through the engine when the run binds the tool to a mock,
and settles the record with `report-local`. If the report fails, the call
stays in `router.calls` with `reported=False` and
`external_state="indeterminate"`. `router.calls` holds typed
`ToolCallResult` models (`result`, `status`, `provenance`, `external_state`).

Local mode (`Client()` without `base_url`) runs an in-process engine with the
same statuses and refusals: validation, preflight, run lifecycle with
approval, the `static`, `rules` and `recorded` strategies and local
functions through the same two-phase protocol. What needs the gateway
(`mcp` and `http` adapters, `scenario`, `schema_generated` and `plugin`
strategies, connection tests) is refused by the plan with an explicit
error; neither mode falls back to the other.

For asyncio runtimes use `tools.arouter(run_id)` (local functions may be
coroutines) or `tools.ainvoke(...)`: the per-call path awaits an async HTTP
client so the event loop is never blocked by a gateway round-trip. The
administrative methods (profiles, contracts, fixtures, runs) stay
synchronous.

See `examples/10_tool_execution.py` and the cloud documentation
`docs/tool-execution.md`.

### Protect: policy enforcement

A run whose configuration carries a `protect` block is admitted call by
call by the cloud gateway. The router applies the decision and never
executes what was not admitted: a call waiting for a human approval raises
`ToolApprovalPending` (code `approval_pending`), a refused call raises
`ToolCallDenied` (code `policy_denied`), an unknown decision string is
treated as denied. Both land in `router.calls` with `status` `pending` or
`denied` and carry the gateway `protect` decision.

```python
from agenomic.tools import ToolApprovalPending, ToolCallDenied

router = tools.router(run["id"], before_action=lambda identity: audit(identity))
try:
    router.call("payments.refund", {"amount_minor": 90000, "currency": "EUR"})
except ToolApprovalPending as pending:
    # re-issues the same call once approved
    router.resume(pending, poll_interval=2.0, timeout=900.0)
except ToolCallDenied as denied:
    print(denied.code, denied.envelope.safe_explanation)
```

`resume` polls the approval and raises `ToolCallDenied` with the approval
status as code when it is rejected or expired; an approval already
`consumed` is re-issued once with the original idempotency key, which
recovers the stored result, or raises
`ToolExecutionError("conflict", ..., 409)` when the gateway answers 409.
`before_action` runs with
the call identity before any request and aborts the call when it raises.
Local functions forward the signed permit of a Protect run to
`report-local`. Model calls are covered cooperatively: inject the run
overlay with `instrument_openai(client, overlay=client.protect.overlay(run_id))`
(first system message) or `instrument_anthropic(client, overlay=...)`
(`system` prompt). `client.protect` exposes approvals, decisions, policies,
bindings, restrictions, the kill switch, simulation and the coverage
matrix. The local engine refuses `protect` configurations
(`protect_cloud_required`): policies are evaluated by the gateway only. See
`docs/protect.md`.

## Examples

See [`examples/`](examples/) — minimal trace, decorator + JSONL, ATEP local,
OpenAI traced, LangGraph traced, cloud upload, and a full offline signed
release.

## License

Apache-2.0. See [LICENSE](LICENSE).
