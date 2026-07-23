# Canonical runs (spec v0.3)

`agenomic.canonical` produces **`agenomic/v0.3` run traces**: an
append-only, hash-chained event stream with metadata, a causal execution
graph, and a Merkle integrity block — OTel-GenAI-native. This is the
richer, newer surface next to the v0.1 `TraceEnvelope` path.

```python
from agenomic.canonical import start_run

run = start_run("agent://acme/support", provider="openai", model="gpt-4o")
run.log_llm(prompt={"q": "hi"}, response={"a": "hello"})
run.log_tool_call(tool="search", arguments={"q": "x"}, result=[1])
trace = run.complete_run(output={"answer": "ok"})
trace["spec_version"]   # 'agenomic/v0.3'
```

`start_run` emits `run.started` immediately; `complete_run` emits
`run.completed`, seals the chain, and returns the finalized, schema-valid
trace dict. Appending after completion raises `RuntimeError`.

## Capture methods

| method                  | events emitted                                  |
| ----------------------- | ----------------------------------------------- |
| `log_llm(...)`          | `llm.requested` + `llm.responded`               |
| `log_tool_call(...)`    | `tool.call.proposed` + `tool.call.executed`     |
| `log_memory(...)`       | `memory.read`, or `memory.write.proposed` + `memory.write.committed` |
| `log_policy_check(...)` | `policy.check.performed`                        |
| `request_human_review(...)` | `human.review.requested`                    |
| `log_error(...)`        | `error.raised`                                  |
| `complete_run(...)`     | `run.completed`, returns the trace              |

Every event is **redacted before hashing or export** (pass a
`RedactionEngine` via `start_run(..., redaction=engine)`), carries a
content-addressed `payload_hash`, and is chained via `event_hash` from
`GENESIS_PREV_EVENT_HASH`.

## Run configuration

`start_run(agent_id, **kwargs)` accepts keyword-only settings that seed
the run metadata blocks: `agent_version`, `genome_version`,
`runtime_name`, `runtime_version`, `provider`, `model`, `model_version`,
`temperature`, `top_p`, `seed`, `run_id` (defaults to a fresh ULID),
`input_payload`, `classification`, `redaction`, `tracer`, `signed_by`.

## The hash chain

`agenomic.canonical.hashing` is a byte-exact port of the spec verifier:

```python
from agenomic.canonical import (
    canonical_json, content_hash, event_hash, merkle_root,
    GENESIS_PREV_EVENT_HASH,
)

canonical_json({"b": 1, "a": [True, None]})  # '{"a":[true,null],"b":1}'
content_hash({"k": "v"})                     # 'blake3:<64 hex>'
```

- `canonical_json` — deterministic JSON: sorted keys, compact separators,
  `ensure_ascii=False`. Raises `TypeError` on unsupported types.
- `content_hash(payload)` — `"blake3:" + BLAKE3(canonical_json(payload))`.
- `event_hash(event)` — `"blake3:" + BLAKE3(canonical_json(event_sans_hash) + prev_event_hash)`.
  The dict must carry `prev_event_hash` and must not carry `event_hash`.
- `merkle_root(hashes)` — `"blake3-merkle-v1:" + <hex>` with RFC 0002
  leaf/node domain separation and odd-node duplication.
- `GENESIS_PREV_EVENT_HASH` — `"blake3:" + "0" * 64`, the first event's
  `prev_event_hash`.

## The finalized trace

`complete_run` returns a dict with: `spec_version`, `run_id`, `agent`,
`llm`, `components`, `input`, `output`, `events`, `execution_graph`
(nodes + `caused_by` edges), `risk_scores`, `compliance_checks`,
`alignment_checks`, `environment_snapshot`, and `integrity`
(`run_merkle_root`, `signed_by`, `signature`). The signature field is
`"unsigned:…"` — detached signing is the embedder's job (use
`agenomic.crypto.SigningKey` or `agenomic-cli`).

## OpenTelemetry spans

Pass an OTel tracer to surface each event as a GenAI span
(`gen_ai.system`, `gen_ai.request.model`, token usage, tool name — the
standard GenAI semantic-convention attributes). OTel is imported lazily;
a `None` tracer is a silent no-op:

```python
from opentelemetry import trace as otel_trace

run = start_run("agent://acme/support", tracer=otel_trace.get_tracer("app"))
```

Helpers `emit_span`, `llm_attributes`, and `tool_attributes` live in
`agenomic.canonical.otel` for custom instrumentation.

## LangGraph, canonically

`instrument_langgraph_canonical(graph, run, llm_nodes=("plan",))` wires a
LangGraph state graph into a `CanonicalRun` automatically: nodes named in
`llm_nodes` emit `llm.requested`/`llm.responded`; every other node emits
tool-call events; raising nodes emit `error.raised` and re-raise. See
[Integrations](integrations.md).
