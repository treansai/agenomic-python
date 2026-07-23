# API reference

The complete public surface of `agenomic` 0.1.x. Optional integrations
are lazy — importing their modules never loads `openai`, `anthropic`,
`langgraph`, or `mcp`.

- [Top level](#top-level)
- [`Client` facade](#client-facade)
- [Tracing](#tracing)
- [Exporters](#exporters)
- [Types](#types)
- [Canonical runs (v0.3)](#canonical-runs-v03)
- [Online tracking](#online-tracking)
- [RMP](#rmp)
- [ATEP](#atep)
- [Crypto](#crypto)
- [Redaction](#redaction)
- [Cloud client](#cloud-client)
- [Integrations](#integrations)
- [Hugging Face provider](#hugging-face-provider)
- [Agent genome](#agent-genome)
- [Exceptions](#exceptions)

## Top level

```python
from agenomic import Client, __version__
```

## `Client` facade

`agenomic.Client` — local-first umbrella client. With no `base_url`,
tracking/RMP buffer locally; with a `base_url`, they drive Agenomic
Cloud. No silent fallback between modes.

```python
Client(api_key=None, base_url=None, *, timeout=30.0, transport=None)
```

| attribute        | type               | purpose                          |
| ---------------- | ------------------ | -------------------------------- |
| `client.is_cloud`| `bool`             | `True` when `base_url` is set    |
| `client.tracking`| `TrackingResource` | [online tracking](tracking.md)   |
| `client.agent`   | `AgentResource`    | local genome load/configure      |
| `client.rmp`     | `RmpResource`      | [RMP umbrella sessions](rmp.md)  |
| `client.review`  | `ReviewResource`   | pre-release review               |
| `client.monitor` | `MonitorResource`  | production monitoring            |
| `client.protect` | `ProtectResource`  | alerts, plans, recommendations   |

## Tracing

### `agenomic.trace.decorator`

```python
trace_agent_run(agent_id, *, release=None, exporter=None, redaction=None,
                capture_input=True, capture_output=True)
```

Decorator for sync and async functions; each invocation becomes one
`TraceEnvelope`. See [Decorator](decorator.md).

### `agenomic.trace.recorder.TraceRecorder`

```python
TraceRecorder(agent_id, run_id, trace_id)
```

Accumulates one run: `.model_calls`, `.tool_calls`, `.labels`,
`.metadata`, `.started_at`. Methods: `record_model_call(call)`,
`record_tool_call(call)`, `add_label(k, v)`, `add_metadata(k, v)`.

### `agenomic.trace.context`

- `current_recorder() -> TraceRecorder | None` — the active recorder
  (contextvar); integrations use this to attach calls to the active run.
- `set_current_recorder(recorder) -> Token` / `reset_current_recorder(token)`
  — manual install/restore (the decorator does this for you).

### `agenomic.trace.envelope_builder`

```python
build_envelope(recorder, *, raw_input, raw_output, error, release,
               capture_input, capture_output, redaction) -> TraceEnvelope
```

Materializes an envelope, applying redaction and capture flags last.

## Exporters

`agenomic.exporters` — see [Exporters](exporters.md) for behavior.

| class | constructor |
| ----- | ----------- |
| `Exporter` (ABC) | — subclass; implement `export(envelope)`, optional `close()` |
| `JsonlExporter` | `(path, *, flush_each=True)` |
| `AtepLocalExporter` | `(store, signing_key, node_id=0)` |
| `HttpExporter` | `(client, *, batch_size=100, batch_interval_ms=5000)`; `await flush()`, `await aclose()` |
| `MultiExporter` | `(*exporters)` |

All are context managers.

## Types

`agenomic.types` — pydantic v2 models, `extra="allow"` throughout.

### Trace models

- **`TraceEnvelope`** — `schema_version="agenomic-trace/v0.1"`,
  `trace_id`, `run_id`, `agent_id`, `release?`, `timestamp`,
  `input: TraceInput`, `model_calls: list[ModelCall]`,
  `tool_calls: list[ToolCall]`, `final_output: TraceOutput`, `labels`,
  `metadata`, `error?`, `duration_ms?`. See [Tracing](tracing.md).
- **`ModelCall`** — `provider`, `model`, `fingerprint?`, `temperature?`,
  `prompt_hash?`, `output_hash?`, `latency_ms?`, `cost_estimate?`,
  `status: CallStatus = SUCCESS`.
- **`ToolCall`** — `tool`, `protocol`, `server?`, `input_hash?`,
  `output_hash?`, `latency_ms?`, `status`, `requires_human_approval=False`,
  `approval_present?`.
- **`TraceInput`** — `type="json"`, `payload_inline?`, `payload_ref?`.
- **`TraceOutput`** — `hash?`, `payload_inline?`, `payload_ref?`.
- **`CallStatus`** — `SUCCESS | ERROR | ABORTED | TIMEOUT`.

### Attestation

- **`ReleaseAttestation`** — `schema_version="agenomic-attestation/v0.1"`,
  `agent_id`, `release_id`, `bundle_hash`, `atep_root_hash`, `issued_at`,
  `signer_key_id`, `signature_hex`, `notes?`. Signing convention in
  [Keys & signing](keys-and-signing.md).

### Identifiers

- `AgentId`, `TraceId`, `RunId`, `ReleaseId` — `NewType` string aliases.
- `validate_agent_id(value) -> AgentId` — enforces
  `agent://[a-z0-9-]+/[a-z0-9-]+`; raises `ValueError`.

### Orchestration (spec v0.2, RFC 0009)

`WorkflowSpec`, `WorkflowIdentity`, `WorkflowStep`, `ToolRef`,
`HumanGate`, `WaitFor`, `RetrySpec`, `TriggerSpec`, `IoField`,
`StateRef`, `SignalSpec`, `EscalationRule`, `EngineHint`, `SystemSpec`,
`SystemIdentity`, `SystemMember`, `AutonomySpec`, `OrchestrationSpec`,
`OrchestrationEdge`, `WorkflowRef`, `END_VERTEX`. See
[Orchestration](orchestration.md).

## Canonical runs (v0.3)

`agenomic.canonical` — see [Canonical runs](canonical.md).

- **`start_run(agent_id, **kwargs) -> CanonicalRun`** — begins a run,
  emits `run.started`.
- **`CanonicalRun`** — capture methods `log_llm`, `log_tool_call`,
  `log_memory`, `log_policy_check`, `request_human_review`, `log_error`;
  `complete_run(output=None, status="success", ...)` returns the
  finalized `agenomic/v0.3` trace dict.
- **Hash helpers** — `canonical_json`, `content_hash`, `event_hash`,
  `merkle_root`, `GENESIS_PREV_EVENT_HASH`.
- **OTel helpers** (`agenomic.canonical.otel`) — `emit_span`,
  `llm_attributes`, `tool_attributes`, GenAI attribute constants.

## Online tracking

`agenomic.tracking` — see [Tracking](tracking.md).

- **`TrackingResource.start(*, agent, release_id=None, bundle_id=None,
  genome_hash=None, environment="production", tracking_config=None)
  -> TrackingSession`**
- **`TrackingSession`** — `event(type, **fields)`, `step(name)` (context
  manager), `model_call(...)`, `tool_call(...)`, `intent(value)`,
  `memory_write(...)`, `stop()`, `report()` (cloud), `to_jsonl()`
  (local), `.events`, `.cloud`, `.session_id`. Context manager.
- Constants: `SPEC_VERSION = "agenomic/v0.3"`, `TRACKING_EVENT_TYPES`.

## RMP

`agenomic.rmp` — see [RMP](rmp.md). All shapes are snake_case dicts
stamped `agenomic.rmp/v0.1`.

- **`RmpResource`** — `start(*, agent, release_id=None,
  environment="production", ledger=False, genome_hash=None)`, `get(id)`,
  `list()`, `report(id)`.
- **`ReviewResource`** — `run(*, agent, scenarios=None, risk_matrix=None)`,
  `add_scenario(scenario)`, `list_scenarios(agent=None)`,
  `approve_scenario_enrichment(proposal_id, *, session_id=None,
  reviewer=None)`, `.proposals`.
- **`MonitorResource`** — `start(*, agent, release_id=None,
  environment="production", ledger=False) -> MonitorSession`,
  `event(session_id, event)`, `findings(session_id)`.
- **`MonitorSession`** — `event({"type": ..., ...})`, `stop()`,
  `.events`, `.cloud`. Context manager.
- **`ProtectResource`** — `alerts(session_id)`, `action_plan(alert_id, *,
  session_id=None)`, `recommendations(session_id)`, `notify(alert_id, *,
  session_id=None)`.
- Constants: `RMP_SPEC_VERSION`, `RMP_REPORT_VERSION`,
  `ENRICHMENT_VERSION`, `SCENARIO_VERSION`.

## ATEP

`agenomic.atep` — see [ATEP](atep.md).

- **`AtepStore`** — `open_or_init(root, agent_id)`, `append_event(event)`,
  `read_stream(stream)`, `list_segments(stream=None)`,
  `verify_all(verifying_key) -> VerificationReport`,
  `compute_root_hash() -> bytes`, `manifest()`.
- **`AtepEvent`** — `seal(header, payload, signing_key)` (classmethod),
  `verify(verifying_key) -> bool`,
  `compute_causal_hash(header, payload)` (static), fields `header`,
  `payload`, `causal_hash`, `attestation`.
- **`EventHeader`** — `schema_version=1`, `event_id` (16 bytes),
  `agent_id`, `stream: StreamId`, `stream_seq`, `clock: Hlc`, `parents`,
  `event_type`, `payload_schema_uri`.
- **`EventAttestation`** — `signer_key_id`, `signature` (64 bytes),
  `algo="ed25519"`.
- **`StreamId`** — `IDENTITY | CAPABILITY | KNOWLEDGE | POLICY | RUNTIME
  | INTERACTION | GOVERNANCE`.
- **`SegmentWriter`** / **`SegmentReader`** — `append(event)` +
  `finalize() -> SegmentSummary`; `iter_events()`,
  `verify_merkle_root()`, header attributes (`event_count`, `first_hlc`,
  `last_hlc`, `merkle_root`). Writer is a context manager.
- **`Hlc`** — frozen, ordered dataclass `(physical_ms, logical, node_id)`;
  `now(node_id=0)`, `tick_after(received, now_ms=None)`,
  `to_le_bytes()` / `from_le_bytes(data)` (16-byte LE wire format).
- **`VerificationReport`** — `ok`, `segments_checked`, `events_checked`,
  `failures`.

## Crypto

`agenomic.crypto` — see [Keys & signing](keys-and-signing.md).

- **`SigningKey`** — `generate(key_id=None)`,
  `from_pem_file(path, key_id=None)`, `sign(message) -> bytes`,
  `verifying_key()`, `public_pem()`, `write_pem_file(path)`,
  `write_public_pem_file(path)`, `.key_id`.
- **`VerifyingKey`** — `from_pem(pem)`, `from_pem_file(path)`,
  `verify(signature, message) -> bool`, `.key_id`.
- **Hashing** — `blake3_hex(data)`, `blake3_bytes(data)`,
  `hash_with_domain(domain, *parts)`; domains `LEAF_DOMAIN`,
  `NODE_DOMAIN`, `ATEP_DOMAIN`, `ATTESTATION_DOMAIN`.
- **Canonical CBOR** — `canonical_cbor(value)`,
  `canonical_cbor_decode(data)` (RFC 8949 §4.2).

## Redaction

`agenomic.redaction` — see [Redaction](redaction.md).

- **`RedactionEngine(rules)`** — `apply(data)` returns a redacted deep
  copy; never mutates input; unknown paths silently skipped.
- **`RedactionRule(path, mode, truncate_length=None)`** — frozen model;
  `truncate_length` required (and `>= 0`) when `mode=TRUNCATE`.
- **`RedactionMode`** — `REMOVE | MASK | HASH | TRUNCATE`.

## Cloud client

`agenomic.client` — see [Cloud upload](cloud-upload.md).

- **`AgenomicClient(endpoint, api_key, *, timeout=30.0,
  retry_policy=None, user_agent=None, transport=None)`** — async. Methods
  (all `async`, return `dict`): `whoami()`, `upload_traces(envelopes)`,
  `upload_bundle(agent_id, archive_path)`,
  `upload_atep_segment(agent_id, segment_path)`,
  `create_release(request)`, `get_replay_report(job_id)`, `aclose()`.
  Idempotency keys on every POST; retries on network errors and
  429/502/503/504; `Retry-After` honored; 401 →
  `AuthenticationError`, other errors → `CloudError`.
- **`SyncAgenomicClient(...)`** — same surface, sync via `asyncio.run`;
  `close()`. Not usable inside a running event loop.
- **`RetryPolicy(max_retries=3, base_delay=0.2)`** — frozen dataclass;
  delay is `base_delay * 2**attempt`.
- `agenomic.client.auth.bearer_header(api_key)` →
  `{"Authorization": "Bearer ..."}`.

## Integrations

`agenomic.integrations` — all lazy; see [Integrations](integrations.md).

| function | wraps | records |
| -------- | ----- | ------- |
| `openai.instrument_openai(client)` / `instrument_openai_async(client)` | `chat.completions.create` | `ModelCall(provider="openai", fingerprint=...)` |
| `anthropic.instrument_anthropic(client)` / `instrument_anthropic_async(client)` | `messages.create` | `ModelCall(provider="anthropic")` |
| `huggingface.instrument_huggingface(client)` | `generate_text`, `embeddings` | `ModelCall(provider="huggingface")`, token never logged |
| `huggingface.trace_huggingface_call(fn, *, model, prompt=None, parameters=None, **kwargs)` | any inference callable | `ModelCall` around the call |
| `langgraph.instrument_langgraph(graph)` | each graph node | `ToolCall(protocol="local", server="langgraph")` |
| `langgraph.instrument_langgraph_canonical(graph, run, *, llm_nodes=())` | each graph node | canonical v0.3 events on a `CanonicalRun` |
| `mcp.trace_mcp_call(server, tool, input_data, output_data, *, status=SUCCESS, latency_ms=None, requires_human_approval=False, approval_present=None)` | manual | `ToolCall(protocol="mcp")` |

All record on `current_recorder()` and no-op outside a
`@trace_agent_run` context (except `instrument_langgraph_canonical`,
which targets an explicit `CanonicalRun`). Wrappers record on success
**and** error, then re-raise. Missing optional packages raise
`ImportError` at call time with the install hint.

## Hugging Face provider

`agenomic.providers.huggingface` — httpx-based connector; see the
[provider guide](providers/huggingface.md).

- **`HuggingFaceConfig`** — `from_env()`, `.has_token`,
  `.inference_base`, `redact(text)`, `auth_header()`; fields
  `endpoint_url`, `org`, `default_model`, `timeout_seconds=30.0`; token
  private and repr-excluded.
- **`HuggingFaceClient(config=None, *, transport=None)`** —
  `validate_credentials()`, `resolve_model_metadata(model_id,
  revision="main") -> ModelMetadata`, `generate_text(model, prompt,
  parameters=None)`, `embeddings(model, inputs)`.
- **`ModelMetadata`** — `model_id`, `revision`, `resolved_commit`,
  `task`, `private`.
- **`build_lockfile_model(*, config, metadata, parameters=None)`** —
  deterministic, token-free lockfile entry (SHA-256 over canonical JSON).
- **`normalize_provider(name)`** / **`is_huggingface(name)`** — alias
  handling (`hf`, `hugging_face`, … → `huggingface`).
- Errors: `HuggingFaceError`, `HuggingFaceAuthError` (401/403) — always
  token-redacted.

## Agent genome

`agenomic.agent` — local `genome.yaml` / `genome.json` handling.

- **`load_agent(path) -> Agent`** — path may be a genome file or a
  directory containing `genome.yaml`/`genome.yml`/`genome.json`; a
  missing file yields an empty genome created on first save. Also
  available as `client.agent.load(path)`.
- **`Agent`** — `.runtime`, `configure_model(*, provider, model,
  task=None, revision=None, parameters=None, save=True)` (writes
  `runtime.model`; HF aliases normalized), `save()`.
- **`GenomeError`** — genome could not be located, parsed, or updated.

## Exceptions

See [Errors](errors.md): `AgenomicError` → `ValidationError`,
`CryptoError`, `AtepError`, `ExportError`, `RedactionError`,
`CloudError` → `AuthenticationError`.

## CLI

`agenomic-py` — `atep verify`, `atep inspect`, `traces summarize`,
`keys generate`. See [CLI](cli.md).
