# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `client.benchmarks` (cloud only): catalogue, plans, preflight, idempotent
  launch, runs, comparison and Protect policy proposals for RMP benchmarks.
  `rmp.start()` is unchanged and still never executes anything.
- `agenomic.benchmarks.AgentTargetBridge` + `serve_bridge()` and the
  `agenomic-py benchmark serve` command: serve your own agent to benchmark
  turns relayed by Agenomic; benchmark tools replace production tools per
  trial and are executed by the benchmark environment.

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
