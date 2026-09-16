# agenomic-python — agent instructions

Public Python SDK for Agenomic. Apache-2.0.

This file governs **developing this repository**. If you are an agent
**using** the SDK in another project, read [AGENT.md](AGENT.md) instead.

## Product invariants

1. Works fully offline. No primitive requires network.
2. ATEP-native. Segments produced here are bit-for-bit compatible with
   agenomic-cli and agenomic-cloud.
3. Integrations are optional and lazy. Top-level imports never load
   openai, anthropic, langgraph or mcp.

## Engineering rules

- mypy strict. No `Any` in public APIs.
- pydantic v2 for all data models.
- No `print()` in library code. Use `logging.getLogger("agenomic.<module>")`.
- All public functions have docstrings with at least one example.
- Async-first for I/O-bound primitives (HTTP, file streams). Provide sync
  wrappers via `asyncio.run` only at the top level.
- No bare `except:`. Catch specific exceptions.

## Naming

- Package: `agenomic`
- CLI: `agenomic-py` (entry point)
- ATEP file extension: `.atep`
- Default config dir: `~/.config/agenomic/`

## Security defaults

- Redaction runs BEFORE any export.
- ed25519 PEM keys loaded with file mode check (warn if not 0600).
- HTTP client uses TLS verify=True. No `verify=False` flag exposed.

## Protect

- `ToolCallDenied` and `ToolApprovalPending` keep the names of the cross-SDK
  design (`docs/protect/design.md` section 9 in agenomic-cloud) and carry a
  `noqa: N818` marker instead of an `Error` suffix so Python, TypeScript and
  the web mirror the same vocabulary.
- The routers never execute a local function unless `local/authorize`
  answers `local` with a `record_id`; `pending` and `denied` are appended to
  `router.calls` with that status and raised as typed errors; any other
  decision string is denied. The gateway path applies the same rule to the
  envelope `status` and to a 403 carrying the invoke envelope.
- `resume` re-issues the identical identity through `call`, so the gateway
  resumes the pending claim on the claim conflict path and the idempotency
  key matches the original request; `approved` and `consumed` both trigger
  the re-issue exactly once, which is the behaviour the TypeScript SDK also
  implements. A consumed approval means the gateway already executed, so the
  replay recovers the stored result when the key replays the cached 200; a
  409 means the evidence stands but cannot be replayed and raises
  `ToolExecutionError("conflict", ..., 409)`, never a denial that would
  suggest nothing ran. `rejected`, `expired` and any other terminal status
  stay `ToolCallDenied` with that status as `code`. The router only resumes
  approvals it held itself: it has no other honest source for the identity.
- The consumed recovery contract is verified against a real gateway, not only
  against an HTTP fixture: `examples/11_protect_consumed_recovery.py` is the
  manual harness (closure criterion 3 of the 2026-09-15 Protect campaign) and
  `examples/protect-consumed-recovery.ts` in agenomic-typescript is its twin,
  asserting the same contract so a divergence fails an assertion on one side.
  Observed on a gateway with Postgres and two user owned API keys: on the
  gateway executed path (`invoke`, where the SDK sends the Idempotency-Key
  derived from run, repetition and logical call id) a resume over a consumed
  approval answers HTTP 200 and replays the stored result with no second
  effect; on the runtime local path (`local/authorize`, which carries no
  idempotency key) the same resume answers HTTP 409 and raises
  `ToolExecutionError("conflict", ..., 409)`. Both branches of the contract
  are therefore real and path dependent, not credential dependent, and both
  SDKs produce them identically. Keep both branches; narrowing `resume` to one
  of them would break the other execution point.
- `resume` deletes its pending entry before re-issuing, so calling `resume`
  twice on the same router never reaches the gateway. The consumed path is
  reachable only from a second holder of the same identity, which the harness
  builds by having two routers issue the identical logical call id and both
  receive the same 202 and approval id. This is deliberate: a router must not
  invent an identity it did not hold.
- `before_action` runs before every request, including on resume, and its
  return value is ignored: a hook can only abort, never widen.
- `client.protect` subclasses the RMP `ProtectResource` and routes every new
  method through `agenomic.tools.resources.typed_request` so refusals carry
  the server error code; local mode raises `cloud_required` because there is
  exactly one policy evaluator and it runs in the gateway. `overlay()` returns
  a `ProtectOverlay` model because the integrations consume it; every other
  method returns plain dicts like the RMP resources.
- Wire shapes are the ones the gateway produces (`handlers_protect.rs`,
  `handlers_policies.rs`): list reads are enveloped under one documented key
  and a body without that array raises `invalid_response` instead of
  answering `[]`, because an empty list and a shape mismatch are different
  facts; single-record reads return the bare record, so `approvals.get`
  reads `status` at the top level and `resume` polls it there.
  `decisions.list` is the one paginated read and returns
  `{"decisions": [...], "next_cursor": str | None}` with a `cursor` argument.
  `policies.register` accepts a document mapping (bare JSON) or a string
  (sent as `{"document_text": ...}`, the YAML form the gateway compiles).
  `bindings.list` carries the `status` filter the gateway exposes.
- `authorize_local` reads a 403 body carrying `decision` or `protect` as the
  denied decision instead of raising a generic `ToolExecutionError`: the
  gateway answers 403 for a denied runtime-local authorize (design section 7),
  and the router must raise `ToolCallDenied` with the decision. A 403 without
  those fields (capability lapse, inactive run) stays a typed transport error.
- Overlay injection happens in the pre call window before the request hash so
  the recorded `prompt_hash` covers what the provider received; it is
  idempotent (first system message or `system` prefix compared verbatim).
- The pre-existing `ruff format --check` failures on markdown files are not
  touched by this feature.
