# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Managed prompts (`agenomic.prompts`, RFC 0012 of agenomic-spec, whose
  conformance vectors run in the test suite): references (`prm_x:7`,
  `prm_x@alias`, `agenomic://` URIs) with no implicit latest version,
  `agenomic.prompt_content/v1` documents verified by digest, the strict
  `agenomic-fstring/v1` renderer with fragments, chat placeholders and the
  `rendered_hash`, and a secret scan of every template.
- `client.prompts`, `client.bindings` and the read-only `client.channels`,
  each call with an `a*` twin: versions, drafts, aliases, agent resolution,
  per-thread execution bindings (the first writer wins, and `thread_key`
  hashes thread ids before they leave the process) and signed offline
  bundles that `PromptBundle.load` verifies against trusted keys or a
  pinned digest. `Client()` without `base_url` runs the in-process
  `LocalPromptEngine`, which returns the same error codes.
- `Client.from_env()`, `client.workspace_id`, `client.whoami()`, `close()`,
  `aclose()` and context manager use, and
  `agenomic.exceptions.ApiError` with the server code, status and details.
- A prompt cache in memory, and on disk with `PromptCache` or
  `AGENOMIC_PROMPT_CACHE_DIR`, keyed by workspace and holding immutable
  records only. During a registry outage calls retry and then raise
  `RegistryUnavailableError`; only `CloudBindingAuthority` serves the cached
  binding of a thread that already has one.
- Prompt imports: `agenomic.prompts.discovery.scan_paths` finds the prompts
  of existing code without importing or running it, and Agenomic Cloud
  plans the import for review (`import_report`, `apply_import`).
  `plan_declarations` and `apply_declarations` apply
  `agenomic.prompts_file/v1` files, as JSON or as YAML under the
  `agenomic-yaml/1` profile with the new `yaml` extra, and
  `register_runtime` plans the import of live LangChain templates. No
  source file is ever modified.
- `client.bindings.report_usage`, which reports the prompts a binding
  rendered as references and hashes only.
- `agenomic-py prompts scan`, `import`, `render`, `digest` and
  `bundle-verify`.
- `agenomic.integrations.langchain_prompts` with the new `langchain` extra:
  `to_langchain`, `to_langchain_messages` and `from_langchain`.
- `TrackingCallbackHandler` adds `prompt_binding_id`,
  `prompt_manifest_digest`, `agent_version`, `prompt_refs` and
  `prompt_rendered_hash` to model call events when the run metadata carries
  them, and `CanonicalRun(prompt_manifest_digest=...)` records the digest
  as the `prompt_version` component.
- `bind_langgraph` and `prompts_for` (`agenomic.integrations`): every
  LangGraph thread is pinned to one prompt release and keeps it through
  interrupts, resumes and process restarts, while new threads follow the
  channel. `scope_config`, `managed_prompt`, `AgentFactory` and the offline
  `LocalBindingStore` complete the adapter, and examples 12 to 16 run it
  offline.
- Prompt experiments (`agenomic.experiments`, Agenomic Cloud only):
  `client.experiments` with `create`, `update`, `preflight`, `launch`,
  `cancel`, `get`, `results` and `events`, each with an `a*` twin. `launch`
  cites the preflight `spec_digest`, and every frozen spec read is checked
  against its digest (`experiment_spec_digest_mismatch`).
- `ExperimentRunner` and `GraphTarget`, the runner that executes trials on
  your machines with a runner token (`AGENOMIC_RUNNER_TOKEN`): each trial
  runs on a fresh thread, checkpointer and store with the pinned prompts of
  its arm, tools go through `ctx.wrap_tools`, secrets are resolved on the
  runner only (`EnvSecretResolver`), the trial budgets are enforced, and the
  redacted result, with its isolation record, is accepted once per lease.
  Node experiments run one graph node (`GraphNodeEntryPoint`) or a function
  (`CallableEntryPoint`); custom evaluators and model judges run on the
  runner.
- `agenomic-py experiment serve` and `agenomic-py experiment snapshot`;
  `snapshot_case` freezes the state of a production thread into a
  counterfactual `node_state` case. `local_assignment` and
  `ExperimentRunner.run_trial` run a trial offline, and example 17 runs a
  counterfactual offline.
- `client.rmp.start(candidate_release_id=...)` links an RMP session to a
  candidate release.

### Changed

- The `langgraph` extra requires `langgraph>=1.0.10,<2` and
  `langchain-core>=1.6.3,<2` instead of `langgraph>=0.2`, the range of the
  managed prompts adapter. `docs/langgraph-matrix.md` lists the tested
  points, and any other version emits one `AgenomicUntestedVersionWarning`
  per process.

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
  when any event has been dropped, and `dropped_events(session)` reports the
  count, so drainage is no longer mistaken for delivery. A closed emitter
  counts and logs what it refuses rather than queueing it behind a stopped
  worker, and a handler built for an already-stopped session leaves no thread
  behind. The
  worker survives any exception the session raises, so one unserializable
  field cannot silently end tracking. Token usage counts the first choice that
  carries `usage_metadata` per batch instead of summing the batch: providers
  attach the response-level usage to every choice, which multiplied the counts
  by `n`. Batches are still summed, one per request.
- `TrackingSession.stopped` and `TrackingSession.on_stop(callback)`:
  `on_stop` runs a callback at the top of `stop()`,
  while the session still accepts events, so a buffering producer can drain
  into it. Each callback runs at most once even if a failed `stop()` is
  retried.

- Complete documentation set under `docs/`: new pages for the API
  reference, exporters, canonical v0.3 runs, online tracking, keys &
  signing, workflow/system manifests (RFC 0009), the `agenomic-py` CLI,
  and the exception hierarchy, plus a `docs/README.md` index and expanded
  ATEP/quickstart pages. All code snippets are verified against the SDK.
- `AGENT.md`: a condensed SDK integration guide written for AI coding
  agents: import map, canonical recipes, environment variables, and
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
