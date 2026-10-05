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
  stay `ToolCallDenied` with that status as `code`, and the refusal is
  recorded: the pending envelope is copied with `status="denied"`, appended to
  `router.calls` and carried by the error, so an audit consumer tells a
  terminal refusal from an approval still waiting instead of seeing `pending`
  in both cases. The router only resumes
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
  return value is ignored: a hook can only abort, never widen. `AsyncToolRouter`
  awaits an awaitable return so an `async def` hook actually runs before the
  call is admitted; `ToolRouter` cannot await, so it closes the coroutine and
  raises `ToolExecutionError("invalid_hook", ..., 0)` rather than discarding
  it, since a silently skipped hook would let a call the hook meant to abort
  proceed.
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

## Managed prompts

Reasons behind `agenomic.prompts`, `agenomic._transport` and
`agenomic.integrations.langchain_prompts`, kept here because this code carries
no comments and no docstrings. That is a deliberate exception to the docstring
rule of the engineering rules above.

- The contract is RFC 0012 of agenomic-spec. Its conformance vectors are
  vendored byte for byte under `tests/fixtures/spec_vectors/` with
  `SPEC_VECTORS.lock` (`spec_commit`, `manifest_sha256`). Refresh them only with
  `scripts/sync-spec-vectors.sh <spec-commit> [<agenomic-spec checkout>]`,
  never by hand: `tests/test_prompts_conformance.py` checks the lock, the file
  set and every hash before running the `python` vectors, and fails on an
  unknown suite. `.gitattributes` marks the directory `-text` so a Windows
  checkout keeps the bytes the manifest hashes. The YAML suite is added with
  the prompts file importer.
- `render.py` ports the reference algorithm of the vectors. The order of the
  checks decides `errors[0]`, so every name-keyed map is visited in UTF-16 key
  order (`digest.sorted_keys`), secret findings are collected before the chat
  entry checks, and `bool` is always tested before `int`. Digit strings are
  matched with ASCII regular expressions before `int()`, which would accept
  `1_0` and non-ASCII digits.
- `canonical_json_v1` is new. The shared `agenomic.canonical.hashing`
  function keeps its code-point key order and `repr` floats because ATEP and
  trace hashes depend on it; `test_shared_canonical_json_unchanged` pins that.
- Digests are always computed over the AJS-normalized parsed value, never over
  a pydantic dump; the pydantic models are typed views built after
  validation. `AjsError` is a `ValueError`, not an `ApiError`, because the JSON
  subset check has no wire code of its own; each caller maps the reason
  (render error, template error, `bundle_incomplete`).
- Building a `ManagedPromptVersion` runs the full content validation, secret
  patterns included, so a stored version that matches a pattern added later
  fails closed with `prompt_secret_detected`, as a bundle export would.
  The top-level code of a failed validation follows the server and the error
  registry: `prompt_secret_detected` when any secret was found, then
  `prompt_content_too_large` for every size and count limit, then
  `prompt_kind_mismatch`, `prompt_fragment_cycle` and
  `prompt_fragment_depth_exceeded` for those items, else
  `prompt_template_invalid`.
- `PromptUri.to_version_ref(workspace_id)` takes the workspace so a URI can
  never be reduced to a local `prm_x:n` without the workspace check.
- `PromptBundle.load` runs AJS, schema, authenticity, expiry, entry digests,
  the artifact set digest (in file, then the caller's pin), manifest digests,
  exact closure (prompts, fragments, children with their release pins), scope,
  then governance. The signature is ed25519 over BLAKE3 of `canonical_json_v1`
  of the document without `signature`; the embedded PEM is never trusted. A
  signed bundle is verified whenever a trust store is given; governance
  applies only when no `expected_bundle_digest` was given, since the pin is the
  operator's approval. `BundleTrust.from_pem_files` names each key after the
  file stem (save the key of `GET /v1/signing-keys/:key_id` as
  `<key_id>.pem`); `BundleTrust.from_pems` takes explicit ids.
  `from_online_response` refuses a document that carries a signature.
  `expires_at` must be exactly `YYYY-MM-DDTHH:MM:SSZ`, the SPEC timestamp,
  because `datetime.fromisoformat` accepts different forms on Python 3.10 and
  3.11.
- `_transport.py` keeps one pooled `httpx.Client` per `Client`, and one
  `httpx.AsyncClient` per running event loop, in weak maps keyed by the
  `Client`, so the client facade does not need to change for pooling;
  `close_pool` and `aclose_pool` release them. Retries happen only with
  `retry=True`, on transport errors and on 429, 502, 503 and 504, with the CLI
  schedule (0.2 s, 0.8 s, 3.2 s) and `Retry-After` seconds when present; a 500
  is never retried because `artifact_integrity_error` is a 500. A final
  transport error, 429, 502, 503 or 504 raises `RegistryUnavailableError`
  (status 0 for a transport error) with `details.cause`, whether or not the
  call retried. Server codes map to classes through one explicit table
  (`prompts.errors`), never by prefix, because some 409 codes are binding
  errors; unknown codes stay a plain `ApiError`. `request_id` and the other
  error envelope fields are copied into `details`. Tests replace the module
  level `_sleep` and `_asleep`.
- `PromptCache` keys always include the workspace. Disk entries are written
  atomically (temporary file then `os.replace`, files 0600, directories 0700),
  every path segment is validated first, on writes as on reads, since a
  version record does not check the prompt id grammar itself (a bad segment is
  a `ValueError`, so no traversal is possible), and every read is re-verified.
  The SDK-owned `v1` directory is 0700 too; the caller's directory is left as
  it is. A mismatch raises `cache_conflict`; whether that is a miss (online)
  or a failure (offline) is the caller's decision.
- `LocalPromptEngine` is a simulation of the governed path: in Agenomic Cloud a
  channel move is a session-only action with approvals. It returns the server
  codes and statuses; a binding request checks its selector before looking up
  an existing binding, and a conflict carries the server's details (binding
  id, release id, `resolved_from`, manifest digest). It never invents genome
  addresses (`genome_version` stays `null`), and its runtime `bundle_id` and
  `bundle_hash` are deterministic placeholders derived from the agent id.
- `to_langchain` refuses `integer`, `boolean` and `json` variables because
  LangChain would print `True` and Python reprs. Its metadata uses the key
  `agenomic_prompt_content_digest`, which the LangChain importer must read.
- `agenomic.prompts` never imports langchain or langgraph, and
  `agenomic.integrations` does not import `langchain_prompts`; a subprocess
  test checks both.
- `client.prompts`, `client.bindings` and `client.channels`
  (`prompts/resources.py`) write each operation once, as a generator that
  yields `Call` requests and cache steps (plain callables). `run_flow` drives
  it with the pooled `httpx.Client`; `arun_flow` drives it with the per-loop
  `AsyncClient` and runs cache steps in `asyncio.to_thread`, so the disk tier
  never blocks the event loop. Errors are thrown back into the generator, which
  lets a flow (the binding authority) catch them. The sync and `a*` methods
  therefore cannot drift apart.
- Request bodies follow the gateway's request types: unknown members are
  refused and list or map members cannot be `null`, so empty `child_selectors`,
  `variable_descriptions` and the optional `expect` are omitted rather than sent
  as `null`. Responses are read in their wrapped form (`{draft}`, `{alias}`,
  `{channel}`, `{prompt}`, `{version, created}`), a counter in the body must
  equal the `ETag`, and a shape mismatch raises `invalid_response`.
- `get` with `prm_x:n` or a URI reads the cache first, then
  `GET .../versions/n?include=fragments`, verifies the version and its fragment
  closure and caches it; the answered `prompt_id` and version must equal the
  request, and `canonical_uri` must name the client's workspace
  (`workspace_mismatch` otherwise). An alias always goes through
  `POST /v1/prompts/resolve`; only the concrete version is cached, its digest
  must equal the resolved digest, and the result carries `resolved_from`. An
  alias inside a managed run raises `alias_in_managed_run`; the check reads
  LangChain's `var_child_runnable_config` only when
  `langchain_core.runnables.config` is already in `sys.modules`, so
  `agenomic.prompts` still never imports LangChain. Online, a cache conflict is
  a miss on read and is ignored on write (the cache logs it).
- `Client.workspace_id` is the configured value or the `org_id` of one
  memoized `GET /v1/whoami`; once both are known and differ, every call raises
  `workspace_mismatch`. `whoami()` also feeds the later privileged-key check.
- `client.bindings` passes the caller's thread key through unchanged (hashing
  is the caller's job), sends no `Idempotency-Key` (create-or-get on the thread
  key), always asks for `include: ["artifacts"]`, and loads them with
  `from_online_response` pinned to the binding's manifest digest. It also
  refuses an answer whose workspace, agent, thread key, scope or release differs
  from the request, or whose child manifest digests differ from
  `binding.children` (`binding_mismatch`, `manifest_digest_mismatch`).
- `client.bindings.create` never falls back: the outage policy lives in
  `CloudBindingAuthority` (`prompts/authority.py`). On `registry_unavailable`
  it serves a binding only when the same thread key is cached with the same
  scope and selector and its closure re-verifies; it then returns
  `created=False`, logs one WARNING on `agenomic.prompts` and increments
  `counters()["registry_outage_cached_binding_total"]` (a process-wide counter,
  since the SDK has no metrics dependency). Every other case re-raises. A 401,
  403 or 404 evicts the cached binding before raising; a 409 does not. A
  cached binding is rebuilt into a bundle from the binding record plus the
  cached closure and goes through the same online verification again.
- `export_bundle` loads the signed export with `PromptBundle.load`, trusting the
  key of `GET /v1/signing-keys/:key_id` (the same TLS and API key trust as every
  online answer) unless the caller passes `trust`. It allows an unapproved
  release, because exporting the target of an unprotected channel is legitimate;
  offline loading applies the governance step again.
- `client.channels` is read only (no promote, no rollback: moves are session
  only). `history` follows `next_after` until the log ends, because the registry
  pages it, and refuses a `next_after` that does not advance. `aliases.move` sends `If-Match` but every API key gets the registry's
  `session_required`, because alias moves are session only too.
- Local mode (no `base_url`) delegates to `LocalPromptEngine` wherever the engine
  has an equivalent. Listing, version lists, channel lists, move previews,
  counterfactual bindings, `child_selectors`, `whoami` and `export_bundle`
  (use `client.prompts.local.export_bundle` with a signer) raise
  `cloud_required`, and so does `client.prompts.local` on a cloud client.
