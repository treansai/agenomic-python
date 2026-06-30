# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
