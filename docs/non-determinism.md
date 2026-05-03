# Non-determinism: an honest disclaimer

`agentlock-python` traces what happened. It does **not** make LLM behavior
reproducible.

## What you get

- A signed envelope per agent run with prompt + output hashes
- A causal chain (ATEP) of every recorded interaction
- Tamper-evident integrity checks (CRC32 + Merkle root + ed25519)

## What you don't get

- Bit-for-bit replay. LLM providers do not guarantee output stability for a
  given (model, fingerprint, prompt) tuple. Even at temperature=0, hidden
  state and routing differences cause drift.
- Wall-clock determinism. Tool latencies and retries produce different
  timing on every run.
- "Did this trace come from the same code as last time?". For that you need
  the bundle hash from `agentlock-cli` (Rust) — the SDK alone can't see
  your repo.

## What this means in practice

Use ATEP for:

- proving an event was emitted by an agent under a specific release
- detecting tampering after the fact
- attributing model_calls and tool_calls to runs and to releases
- audit and compliance evidence (AI Act, SOC 2, internal review)

**Do not** use ATEP to claim "this run is reproducible". For replay-grade
evidence, pair traces with deterministic seed pinning, fixture replay, and
upstream-provider agreements — all out of scope for this SDK.

This is the same stance taken in `agentlock-spec` RFC 0005.
