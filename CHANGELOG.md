# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-09-20

### Changed

- The package version is derived from the git tag at build time (hatch-vcs);
  `agenomic.__version__` reads the installed distribution metadata.

### Added

- `client.benchmarks` (cloud only): catalogue, plans, preflight, idempotent
  launch, runs, comparison and Protect policy proposals for RMP benchmarks.
  `rmp.start()` is unchanged and still never executes anything.
- `agenomic.benchmarks.AgentTargetBridge` + `serve_bridge()` and the
  `agenomic-py benchmark serve` command: serve your own agent to benchmark
  turns relayed by Agenomic; benchmark tools replace production tools per
  trial and are executed by the benchmark environment. `AsyncBridgeServer`
  and `aserve_bridge()` are the asyncio-native entry points; the sync ones
  wrap them with `asyncio.run`. Wire types are pydantic models, replies are
  redacted (`DEFAULT_BRIDGE_REDACTION_RULES`) before export and handler
  failures are reported without the exception text.
- Protect policy enforcement on the tool execution path: `ToolCallResult`
  gains `denied` and `pending` statuses plus `protect`, `approval_id`,
  `decision`, `transformation` and `safe_explanation`; `ToolApprovalPending`
  (202) and `ToolCallDenied` (403) subclass `ToolExecutionError`; the routers
  never execute a call the gateway did not admit, treat unknown decisions as
  denied, forward the signed `permit` to `report-local`, accept a
  `before_action` hook and resume approved calls with `router.resume(...)`.
- `client.protect` gains `overlay`, `catalog`, `approvals`, `decisions`,
  `policies`, `bindings`, `restrictions`, `kill_switch`, `simulate`,
  `coverage` and `metrics_summary` (sync and async), all through the typed
  transport with server error codes.
- `instrument_openai(..., overlay=)` and `instrument_anthropic(..., overlay=)`
  inject the Protect instruction overlay deterministically before the
  request hash is computed.
- The local tool engine refuses configurations carrying a `protect` block
  (`protect_cloud_required`).
- `agenomic.integrations.langchain.TrackingCallbackHandler`: a LangChain /
  LangGraph callback handler that mirrors every run (turns, graph nodes,
  model calls, tools, retrievers) into a live `TrackingSession` with
  `span_id`/`parent_span_id`/`turn_id`, timing, token usage and content
  hashes. New tracking event types: `turn.*`, `*.failed`, `retrieval.*`.
  `session.stop()` drains the background emitter and joins its worker on its
  own; `flush()` checkpoints mid-session and `shutdown()` tears down a session
  that is never stopped. `flush()` returns `False` when it timed out **or**
  when an event was dropped while draining, and `dropped_events(session)`
  reports the count, so drainage is no longer mistaken for delivery. The
  worker survives any exception the session raises, so one unserializable
  field cannot silently end tracking. Token usage counts the first choice that
  carries `usage_metadata` per batch instead of summing the batch: providers
  attach the response-level usage to every choice, which multiplied the counts
  by `n`. Batches are still summed, one per request.
- `TrackingSession.on_stop(callback)`: run a callback at the top of `stop()`,
  while the session still accepts events, so a buffering producer can drain
  into it. Each callback runs at most once even if a failed `stop()` is
  retried.

- Complete documentation set under `docs/`: new pages for the API
  reference, exporters, canonical v0.3 runs, online tracking, keys &
  signing, workflow/system manifests (RFC 0009), the `agenomic-py` CLI,
  and the exception hierarchy, plus a `docs/README.md` index and expanded
  ATEP/quickstart pages. All code snippets are verified against the SDK.
- `AGENT.md`: a condensed SDK integration guide written for AI coding
  agents — import map, canonical recipes, environment variables, and
  common pitfalls.

- Hugging Face provider connector (`agenomic.providers.huggingface`): provider
  alias normalization, `HuggingFaceConfig` with env loading and token
  redaction, an httpx-based `HuggingFaceClient`
  (`validate_credentials`, `resolve_model_metadata`, `generate_text`,
  `embeddings`), and a lockfile model-entry builder (SHA-256 over canonical JSON).
- Hugging Face instrumentation integration
  (`agenomic.integrations.huggingface`): `instrument_huggingface` and
  `trace_huggingface_call` record `ModelCall(provider="huggingface", ...)`
  without ever logging the token.
- `client.agent.load(path)` + `agent.configure_model(provider=, model=, task=)`
  to write a provider-agnostic model config into a local `genome.yaml`/`.json`.
- Optional dependency extra `huggingface` (`huggingface-hub>=0.20`). The SDK
  core itself only requires `httpx`.
