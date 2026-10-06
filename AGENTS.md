# agenomic-python: agent instructions

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
  checkout keeps the bytes the manifest hashes. The `prompts-file-yaml` suite
  runs `importer.load_prompts_file`.
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
  3.11. `load` refuses a document with a `signature` and a null or missing
  `expires_at` (`bundle_incomplete`, `missing_field`), signed or pinned: an
  export always carries an expiry (SPEC schema, the registry's own bundle
  parser), and a validly signed bundle without one would stay usable forever.
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
  `workspace_mismatch`. The memoized answer also feeds the adapter's
  privileged-key check, whether it runs at bind time or, after an outage at
  bind time, at the first admission the registry answers.
- `client.bindings` passes the caller's thread key through unchanged (hashing
  is the caller's job), sends no `Idempotency-Key` (create-or-get on the thread
  key), always asks for `include: ["artifacts"]`, and loads them with
  `from_online_response` pinned to the binding's manifest digest. It also
  refuses an answer whose workspace, agent, thread key or scope differs from
  the request, whose artifacts name another release than the binding, or whose
  child manifest digests differ from `binding.children` (`binding_mismatch`,
  `manifest_digest_mismatch`). Comparing `resolved_from` with the requested
  selector is the adapter's target check, not this resource's.
- `client.bindings.create` never falls back: the outage policy lives in
  `CloudBindingAuthority` (`prompts/authority.py`). On `registry_unavailable`
  it serves a binding only when the same thread key is cached with the same
  scope and selector and its closure re-verifies; it then returns
  `created=False`, logs one WARNING on `agenomic.prompts` and increments
  `counters()["registry_outage_cached_binding_total"]` (a process-wide counter,
  since the SDK has no metrics dependency). Every other case re-raises. A 401,
  403 or 404 evicts the cached binding before raising; a 409 does not. This
  holds for `create_or_get` and for the binding read that `resolution` makes on
  a closure miss, because a closure cached by another thread on the same
  manifest would otherwise let a later outage serve a binding the registry had
  refused. `get(agent_id, binding_id)` knows no thread key, so it cannot evict;
  the next `create_or_get` of that thread does. A
  cached binding is rebuilt into a bundle from the binding record plus the
  cached closure and goes through the same online verification again.
- `CloudBindingAuthority(allow_privileged_credential=...)` is how the adapter
  passes its privileged-key rule; `None`, the default for direct use, checks
  nothing. With a bool, `create_or_get`, `get` and the binding read of
  `resolution` run the check inside the same flow, just before their
  request, so no binding request leaves with an unchecked credential, and an
  outage there takes the same cached-binding path as an outage of the
  request itself. `confirm_credential` is the check for a binding read from
  the cache without a request (`revalidate="never"`): an outage leaves it
  pending, while a privileged key or a 401, 403 or 404 raises, and the error
  statuses evict that thread's binding first.
- `export_bundle` loads the signed export with `PromptBundle.load`, trusting the
  key of `GET /v1/signing-keys/:key_id` (the same TLS and API key trust as every
  online answer) unless the caller passes `trust`. It allows an unapproved
  release, because exporting the target of an unprotected channel is legitimate;
  offline loading applies the governance step again.
- `client.channels` is read only (no promote, no rollback: moves are session
  only). `history` follows `next_after` until the log ends, because the registry
  pages it, and refuses a `next_after` that does not advance. `aliases.move`
  sends `If-Match` but every API key gets the registry's
  `session_required`, because alias moves are session only too.
- Local mode (no `base_url`) delegates to `LocalPromptEngine` wherever the engine
  has an equivalent. Listing, version lists, channel lists, move previews,
  counterfactual bindings, `child_selectors`, `whoami` and `export_bundle`
  (use `client.prompts.local.export_bundle` with a signer) raise
  `cloud_required`, and so does `client.prompts.local` on a cloud client.
- `docs/prompts.md` documents only what this package ships, and its "Not in
  this release" list names what is still missing; shorten that list as the
  remaining registry reads land. Experiments have their own page and list,
  `docs/experiments.md`, which `docs/prompts.md` points to.
  Its Python blocks are meant to run in order: the local ones as written, the
  cloud ones against `FakePromptServer` (`tests/prompt_fakes.py`) with
  `transport=` added to the `Client`. No test executes them, so run them again
  after an API change. The `AGENT.md` recipe and the `README.md` example
  follow the same rule. Since the import sections, the cloud blocks run
  against `ImportServer` (`tests/test_prompts_resources.py`), built over the
  engine of the local blocks in workspace
  `0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f` and signing with a key named
  `orgkey_01` whose public PEM is `keys/orgkey_01.pem`. The working
  directory holds a copy of `tests/fixtures/prompt_sources/app` as `app/`
  and the YAML block as `prompts.yaml`. `ImportServer` does not complete the
  file-local fragment pin of a prompts file (`{ prompt_id }` without a
  version) as the registry does, so the run completes it before writing.
- The Python blocks of `docs/experiments.md` also run in order, in one
  module namespace registered in `sys.modules` (LangGraph resolves the
  `State` annotations through it). The graph, offline trial, snapshot and
  node trial blocks run as written. The launch block runs against
  `FakeExperimentApi` (`tests/experiment_fakes.py`) with `transport=` added
  to the `Client`, and with `tool_plan_hashes: {}` added to the fake's
  preflight answer, which the registry always sends. The serving block runs
  against `FakeRunnerServer` built over the engine and the two releases of
  the offline block, with two queued trials, `transport=` added to the
  runner, `AGENOMIC_RUNNER_TOKEN` and `AGENT_RUNTIME_DIGEST` set, and
  `idle_timeout=1` because the fake answers an empty claim at once. The two
  CLI commands of the page run through `agenomic.cli.__main__.main` on
  modules registered in `sys.modules`. The trial result shown on the page is
  trimmed from a real offline `run_trial`. No test executes these blocks
  either, so run them again after an API change; the page documents the
  shipped code where it departs from the design (the snapshot `--graph`
  flag, the result members, the runner-side codes).
- The Markdown files touched for managed prompts pass `markdownlint-cli` with
  its default rules (80 columns, tables and code blocks included) and contain
  no em dash. Their code samples carry no new comments. The umbrella leak
  check skips Markdown, so the public-marker check of the added lines is what
  keeps private names out of these files.
- Static discovery (`prompts/discovery.py`) only parses: `ast.parse` with
  `feature_version=(3, 10)`, so a report does not depend on the interpreter
  that ran the scan (the report says `python_grammar: "3.10"`). Nothing is
  imported, executed or evaluated; `textwrap.dedent`, `inspect.cleandoc` and
  argument-less `strip`, `lstrip` and `rstrip` are applied to literals because
  they are pure. Sources are decoded with PEP 263 and universal newlines, so a
  literal equals its runtime value; a file that does not decode is skipped as
  `not_utf8`, one that does not parse as `syntax_error`. A file whose syntax
  tree is too deep to analyze within the recursion limit (hundreds of literals
  joined with `+`) is also `syntax_error`, the reason `ast.parse` gives for the
  same failure, so one file never aborts a scan. Excluded directory
  names are pruned without being listed; a file that matches an exclude
  pattern is listed as `excluded`. Symlinks that leave the root are ignored.
- Everything the scanner reports fits the SPEC report schema, because the
  registry refuses a whole report when one member does not: a file whose
  relative path holds a backslash is left out, a node name that is empty,
  longer than 256 code points or holds a NUL is treated like a dynamic node
  name (`node_unresolved`), and a longer symbol or enclosing function is
  reported as `null`. The root label follows the schema rule (1 to 128 code
  points, no NUL, no leading `/`, `\`, `~` or drive letter) in the scanner and
  in `check_report` alike (`valid_label`), so the importer never refuses a
  report the scanner wrote. A skipped file also releases the constants its
  templates consumed, so they are reported at their own definition.
- Names are resolved only through explicit imports: a LangChain or LangGraph
  class or function counts when it was imported from a `langchain*` or
  `langgraph` module, and a constant of another scanned file counts when it
  was imported with `from <module> import NAME` (root-relative module paths,
  a `src/` layout and relative imports). Star imports and attribute chains are
  not followed. A module constant is a top-level name bound exactly once and
  never rebound (`global`, augmented assignment, loop targets, a second
  assignment); anything else is a runtime value. A constant whose name ends in
  `prompt`, `template`, `instruction(s)` or `system_message` becomes a
  `python.string_constant` candidate unless a recognized constructor consumed
  it, in which case it is reported once, at that constructor; otherwise one
  text would become two prompts in the import plan.
- Positions are the start of the value node, in 1-based code points (AST
  columns are UTF-8 byte offsets and are converted). Secret findings are
  located through a small decoder of the string tokens, so the line and column
  are exact for plain and raw literals, escapes and implicit concatenation.
  After `dedent` or `cleandoc` the finding falls back to the start of the
  literal, and the literal pieces of an f-string are located at the start of
  the f-string, because the positions of nodes inside an f-string changed in
  Python 3.12. A report never carries a matched value.
- Every other string of a report is scanned with the same patterns too: the
  root label, file paths, symbols, enclosing functions, graph node names,
  the agent variable a slot is named after, and issue messages. A match is
  replaced by `[REDACTED:<pattern>]` (the marker alone when the result would
  pass 256 code points; a label that would pass 128 becomes `repository`).
  Names are redacted before the slot path and prompt id are derived, because
  slugs are lowercased and would carry an `AKIA` key as `akia...`; a slug
  that matches only after lowercasing (`GHP_...`) is replaced by the slug of
  its redaction. Each match in a candidate's names or messages is a
  `secret_findings` entry at the candidate's start, since a name has no
  position of its own, and a supported or unsupported candidate becomes
  `blocked_secret`: the schema allows findings on no supported candidate,
  and a slot proposed from a redacted name would not match the code. An
  unresolved candidate keeps its status. The root label and a file without
  candidates have nowhere to hold a finding and are only redacted. The node
  rule counts the raw names, so two secret node names that redact alike
  still give `node_unresolved`.
- A static message (`SystemMessage`, and the string `prompt` of
  `create_react_agent` or `system_prompt` of `create_agent`) is literal text
  for LangChain, so its braces are escaped on import (`static_message_escaped`).
  Proposals: agent prompts and system messages get usage `system`, human
  messages `user`, AI messages `other`, chat templates `chat`; text templates
  and constants are named by their symbol (`system`, `instruction`,
  `user`/`human`/`question`, `tool`/`description`), else `instructions`. The
  slot path is `<node or name>.<usage>`, made unique with `_2`, `_3` in report
  order with node-linked candidates first, and the prompt id is `prm_` plus the
  slot path with dots turned into underscores. Unresolved candidates propose a
  slot but no prompt id or kind.
- The status split follows the SPEC schema and fixtures: `unsupported` is a
  recognized construct with a refused feature (mustache, jinja2, a format spec,
  a callable or non-scalar partial, a custom role, content blocks), while
  `unresolved` is a construction only the runtime knows (f-strings with fields,
  `.format`, `%`, `+` with a runtime value, call results, `hub.pull`,
  composed templates, callable prompts). Graph node rules are the syntactic
  ones of RFC 0012: a prompt referenced by two different nodes keeps
  `node_path: null` with `node_unresolved`, and a node built inline from a call
  result is not linked.
- Report issues have no `syntax` member, so a template syntax error is reported
  with its reason as the code (for example `format_spec`); `from_langchain`
  returns `PromptIssue(code="syntax_error", syntax=...)` like the content
  validator. Issue messages never quote source text.
- `prompts/importer.py` holds one framework-neutral mapping (`SourcePrompt` to
  `convert_prompt`) that the scanner and `from_langchain` both use, so static
  and runtime imports follow the same table. Secrets are scanned on the raw
  strings, before braces are escaped, so offsets point into what the developer
  wrote, and blocked content is never returned. `from_langchain` reads the
  `agenomic_prompt_content_digest` metadata key that `to_langchain` writes; the
  exact path needs a resolver and a structurally equal object, otherwise the
  info `export_metadata_stale` precedes the structural import. Its result also
  carries `secret_findings` (path, offset, length, pattern) for the runtime
  registration that builds a report from it.
- The YAML profile is implemented on PyYAML events (`yaml.parse`), imported
  inside the function, so `agenomic.prompts` never imports yaml and a missing
  extra raises `yaml_support_not_installed` only for YAML input. A parse error
  is `yaml_syntax_error` and a JSON document with a repeated key is
  `duplicate_key`; neither is in the profile's list. Loading never fills the
  content defaults (`complete_content` does), and a file-local fragment entry
  without a version must name a prompt of the same file and must not cycle.
- The import plan is a server document (plan id, workspace, base versions), so
  the client only builds the upload (`build_import_request`) after checking the
  report: repository-relative paths, content only on supported candidates,
  content digests, and no secret in any content or in any other string or
  key (`secret_detected` with the pointer of the member, or of its parent
  for a key, never the value), so a report written by hand or by an older
  scanner cannot upload one. Every plan received is verified
  (`plan_digest`, `summary`, unresolved items skipped) before it is
  printed or applied. Apply decisions default to the planned action with
  `blocked` turned into `skip`, cite `plan_digest`, carry a generated body
  `idempotency_key` and never send the `Idempotency-Key` header.
- `client.prompts.import_report`, `apply_import`, `plan_declarations`,
  `apply_declarations`, `register_runtime` and `client.bindings.report_usage`
  are flows like the rest of `resources.py`, built on the importer helpers
  (`build_import_request`, `verify_plan`, `default_decisions`,
  `new_idempotency_key`, `load_prompts_file`); `upload_report` and
  `apply_import` of `importer.py` stay for the CLI. All of them need Agenomic
  Cloud (local mode raises `cloud_required`). All but `report_usage` need a
  write or admin key: a read key gets the registry's 403
  `api_key_scope_insufficient` verbatim, and there is no client-side scope
  check, so the message and the request count are the registry's. The
  production execution credential stays read-only: `bind_langgraph` refuses a
  privileged key, and it never registers, imports or reports usage by itself.
- `apply_import` and `apply_declarations` take an optional body
  `idempotency_key` and generate one (`import-apply-<hex>`,
  `declarations-apply-<hex>`) when it is omitted; no `Idempotency-Key` header
  is sent. The body key, the replay of an identical report and the dedupe of
  usage observations make these POSTs idempotent, so they retry like reads
  (`retry=True`). A caller that repeats a whole call after an error passes the
  same key itself; otherwise the second call is a new request and may get
  `prompt_import_already_applied`.
- `ImportPlan` holds the verified plan document, the import record members
  (`import_id`, `status`, `replayed`, `report_digest`, `expires_at`) for a
  report, and the slot summary of a prompts file (`slots` with `revision`,
  `added`, `removed`, `changed`, or `None`). `decisions()` is
  `default_decisions`. A report answer must carry a plan whose `plan_id` is the
  `import_id`, whose `agent_id` is the requested one and whose `source` is
  `{kind: discovery_report, digest}` with the digest of the report that was
  sent (the registry digests the report as received and redacts only the
  plan items and its stored copy); a prompts file plan
  must have `source` equal to `{kind: prompts_file, digest}` with the digest of
  the document that was sent; apply results must echo the import id or the
  cited `plan_digest`. Anything else is `invalid_response`.
- `plan_declarations` and `apply_declarations` take a mapping or a YAML or JSON
  source (text, bytes or a path) and always upload the JSON that
  `load_prompts_file` builds under `agenomic-yaml/1`; content defaults are
  left to the registry. Applying a prompts file with `slots` and `agent_id`
  needs `expected_slots_revision`, sent as `If-Match`, because the plan
  document cannot carry the slot revision; it comes from
  `ImportPlan.slots["revision"]`. `apply_import(..., declare_slots=True)`
  needs it too, and no SDK call reads the slot inventory yet.
- `register_runtime(agent_id, {slot_path: template})` imports
  `langchain_prompts` inside the call, converts each template with
  `from_langchain` (no resolver) and uploads a discovery report labelled
  `runtime_registration` with no files; it returns the plan and applies
  nothing. A live object has no file, so each candidate's source path is its
  slot path at line 1, column 1, which keeps candidate ids stable per slot and
  construct. Candidates are sorted by slot path; the construct follows the
  prompt kind; the usage is the last slot segment when it names a usage, else
  `other`; the prompt id is `prm_` plus the slot path with dots and runs of
  underscores turned into one underscore, made unique with `_2`, `_3`. Issues
  carry no position, a syntax error uses its reason as the code, and secret
  findings keep their pattern and length at line 1, column 1.
- `report_usage` refuses, before any request, an observation member outside
  the usage contract (so no raw text can leave the process), an overlay member
  other than `digest` and `position`, and a `rendered_hash` that is not
  `sha256:` plus 64 hex digits (the tracking `input_hash` is BLAKE3). It sends
  nothing for an empty list and splits a longer list into requests of 500.
- `TrackingCallbackHandler._model_start` turns the `agenomic_*` run metadata
  into the tracking keys `prompt_binding_id`, `prompt_manifest_digest`,
  `agent_version`, `experiment_id`, `experiment_arm_key`,
  `prompt_rendered_hash` and `prompt_refs`. A malformed value is dropped, and
  `prompt_refs` is dropped whole unless the three comma lists have the same
  length and every slot, ref and digest is well formed. No text is added; the
  existing `input_hash` is unchanged and is never the rendered hash.
- `CanonicalRun` accepts `prompt_manifest_digest` for the `prompt_version`
  component; without it the placeholder stays, so existing traces keep their
  hashes.
- `agenomic-py prompts` exits 0 on success, 1 when the registry or a prompt
  check refuses, and 2 for usage, configuration or file errors. `scan` prints
  the report on stdout (or `--out`) and a summary on stderr; `import` uploads
  and prints the plan, and applies it only with `--apply`. `import`, and
  `render` of a registry reference, exit 2 before any request unless both
  `AGENOMIC_ENDPOINT` and `AGENOMIC_API_KEY` are set, as their error says;
  the client would otherwise send the request without a key. `digest` runs
  the content validator first and refuses on any error but
  `fragment_not_found`, because a file without its defaulted members, or any
  other document, would otherwise print a digest the registry never computes,
  while fragment pins cannot be resolved offline and do not change the digest.
- `tests/schemas/v0.4/` (three schemas) and `tests/fixtures/prompt_imports/`
  (a report, two plans and a YAML prompts file) are copies from agenomic-spec
  commit `fcfe12a`, the commit of `SPEC_VECTORS.lock`; refresh them with the
  vectors. `tests/fixtures/prompt_sources/` is a scanned tree, never imported:
  `app/raises_on_import.py` would write a marker file and raise if it ran.
- The `langgraph` and `all` extras accept `langgraph>=1.0.10,<2` with
  `langchain-core>=1.6.3,<2`, but only the points CI runs are supported:
  `ci/constraints/langgraph-1.2.11.txt` (primary, used by the main matrix) and
  `ci/constraints/langgraph-1.0.10.txt` (floor). The floor pins
  `langgraph-prebuilt==1.0.8` because langgraph 1.0.10 with its default
  prebuilt 1.0.13 fails at import. An unpinned install would test whatever
  was released last, so every CI install passes a constraint file except the
  `langgraph-latest` job, which runs weekly or on demand, may fail and only
  reports drift. Supporting another version means editing the constraint
  files and the matrix document together. The `langgraph-floor` job runs
  whichever of `tests/test_langgraph_binding_*.py`,
  `tests/test_experiments_counterfactual.py` and `tests/test_examples.py`
  exist, so later test files join it without a workflow change. The SQLite
  saver (`langgraph-checkpoint-sqlite`) is a dev dependency only, for the
  restart tests.
- `integrations/langgraph_binding.py` imports LangGraph and LangChain at
  module import. `agenomic.integrations` exposes its public names through a
  module `__getattr__` and keeps them out of `__all__`, so neither
  `import agenomic.integrations` nor a star import loads LangGraph; a
  subprocess test checks it.
- `ManagedGraph` is a `PregelProtocol` proxy around an untouched compiled
  graph (or an `AgentFactory`). Its signatures take and return `Any`, like
  LangGraph's own, because the proxy does not know the graph's state types.
  Every run and state update entry point admits first; state reads, graph
  drawing and other members are forwarded. `Runnable` defines the schema
  members (`InputType`, `get_input_schema`, `config_specs` and friends) on
  the class, so they are forwarded explicitly; everything else reaches the
  inner graph through `__getattr__`, which never forwards private names.
  `copy` is refused because a copy of the inner graph would drop the binding.
  Overridden framework members carry `typing_extensions.override`, which is
  also what lets the linter accept their capitalized names.
- Admission is written once, as a generator that yields effects (binding
  create or read, resolution, checkpoint read, cache read or write); `_admit`
  and `_aadmit` drive it with the sync or async authority and saver methods,
  so the sync and async entry points cannot drift apart.
- Order of admission: a config that already carries a consistent pinned set
  is a nested managed call (scope switch to this agent, no binding I/O; the
  agent must be the root or a pinned child). A config without the set while
  LangChain's run context variable carries one raises
  `nested_bind_unsupported`; that variable reaches sync code and asyncio
  tasks on Python 3.11 and later only, so on 3.10 async the case cannot be
  detected. Then reserved keys are refused, the thread key is computed, the
  binding is created or read, its target is checked, the pinned set is built
  and, for a new thread-scope binding, the thread's latest checkpoint stamp is
  compared with the binding (`binding_checkpoint_mismatch`; unstamped threads
  are adopted). The thread id may come from graph-level config bound with
  `with_config`; it is then passed explicitly in the call config, because
  langgraph 1.0.10 does not merge it into the checkpoint config.
- The privileged-key check (04 section 4.2 says "at bind time") runs at bind
  time when the registry answers. When `GET /v1/whoami` is unavailable then,
  binding goes on only for a client with a configured `workspace_id`, since
  every cache key needs it, and the check stays pending: it runs again
  before every binding request or read and before a cached binding is used
  with `revalidate="never"`, each time with the transport retries. Until it
  passes, the proxy serves only bindings already in the cache, with the same
  scope and selector and a re-verified closure, and sends no binding
  request, so a key is never used to create a binding before it was
  checked. Without this, a process restarted during an outage could not
  resume a thread whose binding is in its disk cache (3.13 and AC14). An
  execution-scope resume during an outage still fails, because it reads its
  binding by id and `PromptCache` indexes bindings by thread key only.
- Thread keys are hashed with the workspace (`thread:sha256:`,
  `exec:sha256:`); a non-string `thread_id` is hashed as `str(thread_id)`,
  the form SQLite savers store. A pre-issued binding (runner mode) is
  compared with the raw thread id, because its key is server made
  (`exp:...`) and no request leaves the process.
- Execution scope needs `agenomic_execution_key`; nothing mints one, except
  the runnable returned by `ManagedGraph.with_retry`, which mints one per
  logical input before LangGraph's retry wrapper so every attempt shares it.
  `ManagedGraph.batch` never mints, since a retried batch would get new keys.
  A resume, a `None` input or a state update reads the binding id from the
  checkpoint (the one named by `checkpoint_id` when the caller time travels)
  and the authority confirms it.
- LangGraph copies scalar `configurable` values into checkpoint metadata and
  skips keys that start with `__`, which is why the pin scalars are stamped
  and `__agenomic_prompt_set` is not. LangGraph does put the caller config in
  `checkpoints` and `debug` chunks, so the proxy strips the set from
  `config` and `parent_config` there (and from `invoke` results asked with
  those stream modes), copying only the chunks it changes.
- `PinnedPromptSet` is immutable: copies return the same object, pickling
  raises `prompt_set_not_serializable` and `repr` shows only the binding id
  and digest. A `children` key `K` routes node paths that start with `K|`
  (longest key wins), never `K` itself. A mapped child that the binding does
  not pin fails when a node reads it, not when the set is built, so adding a
  subagent mapping to the code does not break threads pinned to an older
  release that never reach that node.
- `prompt_set_unavailable` has two triggers: the accessor receives a config
  that carries the pin scalars but no pinned set (a config rebuilt from
  checkpoint metadata or a stripped stream chunk, or one that crossed a
  process boundary), and a factory-backed graph is asked for a graph before
  any was built (drawing, schemas, or, without a declared checkpointer, an
  execution-scope resume or state read, whose binding is only known from a
  checkpoint). `binding_missing` stays the case with no pin at all.
- An execution-scope resume, `None` input or state update reads the binding
  id from a checkpoint, and a restarted factory proxy has built no graph to
  read it with, since the factory builds only from a pinned set.
  `AgentFactory(build, checkpointer=saver)` names that saver up front, so
  admission reads the checkpoint through it (4.3 step 4(a) unchanged: the
  authority confirms the id) and only then builds the graph of that
  binding. Every built graph must use that saver object, the first one
  included (`factory_topology_mismatch`). Without it the proxy still fails
  closed with `prompt_set_unavailable`, whose message names the argument,
  and never resolves the channel. An explicit execution key on resume was
  not chosen: it would go through `create_or_get`, which creates a binding
  on the channel's current release when the key is wrong. A thread-scope
  factory never needed this, because its key is the thread id.
- `LocalBindingStore(directory)` stores one JSON file per workspace, agent
  and sha256 of the thread key, with a `record_digest` over its own
  `agenomic.local_execution_binding/v1` document. A write goes to a `.tmp-`
  file in the same directory, is fsynced and published with `os.link` (first
  writer wins), then the directory is fsynced on POSIX only (Windows cannot
  open a directory). A file that does not parse or verify raises
  `binding_store_corrupt`; it is never treated as absent. Finding a binding
  by id scans the agent's directory, which only an execution-scope resume
  does.
- The offline authority writes `resolved_from` from the proxy's selector and
  serves each thread the bundle whose release and manifest digest match its
  binding, among `bundle` and `retained_bundles`. A pre-loaded `PromptBundle`
  is trusted as loaded; `trust` or `expected_bundle_digest` is only needed to
  load a path or a document. Offline-only arguments are refused online.
- `AgentFactory` caches graphs by workspace, agent, manifest digest and
  genome version, builds each key once under a per-key lock and keeps the
  first graph as the reference for topology checks (same nodes, same
  checkpointer object) and for state reads; the reference is never evicted.
  A thread-scope state read before any build admits the thread to build one.
- `managed_prompt` renders a text slot as one system message before the
  history, fills a chat slot's placeholder named `history_key`, composes a
  chat slot without placeholder, and refuses another placeholder name
  (`history_conflict`). `config_for` adds `agenomic_rendered_hash` only when
  exactly one of its slots was rendered through the same accessor object,
  since one hash cannot describe two renders.
- `counters()["langgraph_binding_inflight"]` is a process-wide gauge of runs
  between admission and the end of delegation; the untested-version warning
  is emitted once per process.
- The adapter tests share `tests/langgraph_world.py` (seeded registry,
  graphs, a streaming fake model), so `tests/prompt_fakes.py` stays free of
  LangChain. `tests/langgraph_restart_child.py` is the subprocess of the real
  restart tests; they need the SQLite saver dev dependency and skip without
  it, and the async interrupt tests skip below Python 3.11.
- The LangGraph examples 12 to 16 run offline on `GenericFakeChatModel`,
  carry no comments or docstrings, print what they show and check their own
  claims with `assert`, so `tests/test_examples.py` fails when a behaviour
  regresses, not only when a script crashes. They copy no helper from
  `tests/`, because an example must run on its own. Every node passes
  `prompts.config_for(slot)` to the model and returns only the reply: the
  rendered system prompt never enters checkpointed history, and token
  streaming works on Python 3.10 async. Example 15 starts itself again with
  `sys.executable` and `--resume` as the restarted process; without the
  SQLite saver (the repository `.venv` has none, CI installs it through
  `dev`) it prints a hint and exits 0. Example 16 moves the channel while it
  consumes `stream` between the two nodes, before LangGraph starts the next
  step, so the second node of the in-flight run proves the admission pin.
  Examples 15 and 16 move channels on the local engine, a simulation of the
  governed path, and say so in their output.
- `instrument_langgraph` and `instrument_langgraph_canonical`
  (`integrations/langgraph.py`) are no-ops on real LangGraph graphs. They
  wrap a node only when the node, or its `runnable`, is callable: a
  `StateGraph` holds `StateNodeSpec` entries whose `runnable` is a
  `RunnableCallable`, and a compiled graph holds `PregelNode` objects with
  no `runnable`, and neither is callable. Probed on 2026-10-05 with
  langgraph 1.2.11 and 1.0.10, on a builder graph and on a compiled graph:
  the graph runs unchanged and nothing is recorded (0 tool calls, 0
  canonical events). Their tests and `examples/05_langgraph_traced.py` use
  duck-typed graphs only. They are deliberately left as they are: their
  wrappers are sync and call nodes with the state only, dropping `config`,
  so making them work is a redesign, and `bind_langgraph` does not build on
  them. `docs/integrations.md` says so and points to
  `TrackingCallbackHandler`; `README.md` and `AGENT.md` still present them
  as LangGraph tracing.
- `docs/langgraph-matrix.md` lists only cells whose whole test suite ran,
  with the counts and the reason of each skip. It changes together with the
  constraint files, the CI jobs and `TESTED_LANGGRAPH` /
  `TESTED_LANGCHAIN_CORE`, and a cell enters it only after its run passed.
  The CI cells (Linux, Windows) stay under "Not verified" until a CI run is
  recorded.
- The Python blocks of the "LangGraph managed prompts" section of
  `docs/integrations.md` run in order on a fake chat model, like those of
  `docs/prompts.md`. The cloud block runs with `transport=` set to a
  `FakePromptServer` over the local engine of the earlier blocks; the
  offline block needs `bundles/production-v1.json`,
  `bundles/production-v2.json` (two releases of the `support.system` slot
  exported from that workspace) and `keys/orgkey_01.pem`, with
  `V1_BUNDLE_DIGEST` set to the first bundle's `prompt_bundle_digest`.
- `agenomic.experiments` (S4) imports no framework: `errors`, `models`,
  `resources` and `secrets` use only the core dependencies, so `Client` can
  wire `client.experiments`. The runner names (`ExperimentRunner`,
  `GraphTarget`, `GraphNodeEntryPoint`, `CallableEntryPoint`,
  `RunnerEvaluator`, `TrialContext`, `NamespacedStore`, `local_assignment`,
  `snapshot_case`) load through a module `__getattr__`, kept out of
  `__all__`; a subprocess test checks that importing the package leaves
  LangGraph and LangChain unloaded. `errors.py` keeps the design's exception
  names (`IsolationViolation`, `RecordedFixtureMiss`, `TrialBudgetExceeded`),
  so `pyproject.toml` ignores N818 for that file only.
- `client.experiments` writes each operation once as a flow, like
  `client.prompts`. `update` and `preflight` send `If-Match`; `launch` cites
  the preflight `spec_digest` and carries the body `idempotency_key` (it
  retries like a read), never the header. Every experiment read recomputes
  `spec_digest` (sha256 of `canonical_json_v1` of the spec without
  `identity`) and raises `experiment_spec_digest_mismatch` on a difference;
  `tests/fixtures/experiments/` holds the SPEC 7540bbd fixtures whose digest
  two implementations recorded. Only the 04 section 3.10 methods exist, so
  datasets and runners are managed in the web app or over HTTP. Local mode
  raises `cloud_required`.
- The runner token comes from the constructor or `AGENOMIC_RUNNER_TOKEN`
  (never a CLI flag) and is checked against `agr_` plus 64 hex characters.
  It is sent only as the bearer header and joins the literal secret set, so
  no body can carry it. A runner token cannot call `whoami`, so the
  workspace is pinned by the first assignment's binding (or by
  `workspace_id=`) and any other workspace is `workspace_mismatch`.
- Before anything runs, an assignment must agree with itself: view and
  envelope name the same trial and attempt; the binding is thread scope with
  the key `exp:<experiment>:<trial>:a<attempt>`, the arm's release and arm
  key; arm, binding and prompts name one manifest digest; the arm runtime
  digest is the target's. The prompts load with `from_online_response`
  pinned to that digest. Any mismatch is
  `failure(runner_configuration, ...)` and the factory is never called. The
  runner echoes `runner_view_digest` and never recomputes it (cases may
  hold floats).
- Each trial gets a new `InMemorySaver` (or `checkpointer_factory(thread)`,
  whose thread is deleted afterwards unless `keep_checkpoints`) and a new
  `InMemoryStore` (or the user store wrapped in `NamespacedStore`). The
  factory must compile with exactly those objects (`is` checks). The
  wrapper checks every op in `batch` and `abatch`, so the convenience
  methods cannot bypass it, and `list_namespaces` must name a prefix inside
  the trial namespace. `case.input.store_seed` items are written under
  `ctx.store_namespace`, so graphs prefix their namespaces with it. Secrets,
  the lease token and the runner token never enter `configurable` or
  metadata, because LangGraph copies configurable scalars into checkpoint
  metadata; factories read `ctx.secrets` by closure. Nothing deletes what a
  trial wrote into a wrapped user store (its trial-namespace keys and seeds
  stay; `on_trial_end` may remove them), so `GraphTarget.store` must never
  be a production store.
- `serve` stops only on a refusal of its own credentials or configuration.
  A hello refresh (every 60 s, or after `experiment_runner_hello_required`)
  that ends in `registry_unavailable` or a 5xx is logged and retried after
  1 s, like a failed claim, so a gateway restart never cancels the other
  workers' trials; 401, 403 and other refusals still raise. Only the
  startup hello fails fast.
- With an idle timeout, the claim `wait_seconds` is the remaining idle time
  rounded up, so a claim long-polls until the timeout; truncating it sent
  `wait_seconds: 0` for the last second and every worker polled the gateway
  in a tight loop (about 70 claims in one second against a real gateway).
  Each worker sends at most one claim with `0`, once the timeout has passed.
- Graphs run through `bind_langgraph(binding=, resolution=)` with
  `ainvoke`, so a heartbeat can cancel them. On Python 3.10, `interrupt()`
  fails under `ainvoke` (LangGraph's contextvars), so interrupt cases need
  3.11; the test skips below it.
- Terminal events (fixture miss, ambiguous fixture, budget, lease loss,
  isolation refusals from tools) are also recorded on the trial state and
  win over whatever the graph returns, because `ToolNode` with
  `handle_tool_errors` or user code can swallow the exception. The budget
  callback sets `raise_error` and `run_inline`, counts model calls before
  they start and provider-reported tokens after; a call without usage is
  `usage_source: "not_reported"` with null counts. A model call carries the
  first slot of its `config_for` metadata (the result has one slot per
  call); the runner refuses to report a call whose ref or digest is outside
  the bundle (`prompt_outside_manifest`) before the server sees it.
- Tools reach the proxy only through `ctx.wrap_tools`, a `BaseTool` that
  reads the model's `tool_call_id` in `invoke`/`ainvoke` and otherwise
  hashes namespace, task id, tool and arguments (`call_` plus 16 hex) with
  the float-tolerant `canonical_json`. Mode `none` refuses locally. Arguments
  are sent exact (fixtures match them) except resolved secret values, which
  become `[REDACTED]`. `experiment_tool_call_in_progress` waits 0.5, 1 and
  2 s, then fails as infrastructure. A live call runs the user's tool once
  per logical id: the report body is cached and resent identically, so a
  resent permit or a lost report never repeats the side effect. Nothing may
  raise between the effect and that cache: a value that cannot be reported
  as JSON is reported as an error placeholder and ends the trial with
  `output_not_serializable`, otherwise a retried node would run the effect
  again while the server counts the call as indeterminate.
- Live tools are gated on the runner, never by the server's answer alone. An
  assignment whose `view.tools.mode` is not in `GraphTarget.tool_modes()`
  (live without `live_tools=True`) is
  `failure(runner_configuration, tool_mode_unavailable)` before the
  workspace is pinned, secrets resolve or the factory runs. `TrialContext`
  carries `live_allowed` (default `False`, set from the same
  `tool_modes()`), and the proxied tool runs the inner tool on `authorized`
  only when the trial mode is `live` and `live_allowed` is true. Otherwise
  the trial ends with `live_tools_disabled`, marked terminal so a swallowed
  exception cannot hide it, and nothing is reported for that call: it never
  ran, so the server keeps it as indeterminate. A misconfigured or
  compromised server, or a replayed permit, thus cannot cause a side effect
  on a runner that did not opt in, nor in a mock or recorded trial.
- A live call is reserved under the trial lock before the inner tool runs:
  the first caller of a logical id inserts an empty entry, and a concurrent
  caller of the same id (duplicate ids in one `ToolNode` message, threads)
  finds it without a report body and ends the trial with
  `live_call_concurrent` instead of waiting, since a waiter could block the
  event loop the first caller needs. The server answers that resend with
  the stored permit, so only this reservation keeps the effect single.
- Once the trial state is terminal, every proxied tool call re-raises the
  terminal error before any request, whatever the mode, so a swallowed
  `stop_trial` miss, budget stop or unreportable value cannot be followed
  by another external effect in a trial whose outcome is already decided.
  The fixture-miss policy is read from the view only: the result echoes
  `runner_view_digest`, so a per-response `on_fixture_miss` must not
  change how the trial ran.
- A result is built once, redacted (the key rules of
  `DEFAULT_RUNNER_REDACTION_RULES`, then every resolved secret value), and
  frozen in an outbox keyed by `uuid5(trial, attempt, lease token)`; retries
  resend those exact bytes, so N-14 accepts it once. No `Idempotency-Key`
  header is sent, as the route is idempotent on lease and result digest.
  `experiment_result_secret_detected` and `experiment_result_invalid` turn
  into `failure(runner_configuration, ...)`; a stale lease drops the trial.
  Failure and agent-error messages are redacted like log text (the
  literals, then the `agenomic-secrets/1` patterns, then any `agr_` token)
  and then cut to 2 KiB; envelopes are never scrubbed (`scrub_json` would
  mask `lease_token`). Every replacement of resolved values also covers
  their `repr`, JSON and percent-encoded forms, because exception text
  carries values that way (a `KeyError`, a provider URL in an
  `HTTPStatusError`). `TrialState` keeps its literals, lease token and live
  reports out of `repr`, and `jsonable` dumps models with `warnings=False`
  because pydantic serializer warnings quote the offending value.
- Runner logs use only the `agenomic.experiments` logger, which carries
  `LOG_REDACTOR`. A logger filter runs before every handler but only on
  records of its own logger, so experiments modules never log through
  `__name__` children. It rewrites each record: the formatted message, and
  `exc_info` turned into a formatted `exc_text` with the exception object
  dropped, both redacted with the resolved values of every trial in flight
  (`_process` and `arun_trial` track the trial state until delivery ends),
  then the `agenomic-secrets/1` patterns, then any `agr_` token. Literals
  go first so a pattern cannot leave part of a value behind, and nothing is
  cut, so the exception line at the end of a traceback stays. A record the
  filter cannot rewrite becomes a fixed text, because a raising filter
  would escape into the `except` blocks of `_process` and stop `serve`.
- `snapshot_case` only reads: `get_state` and its parent. It refuses
  pending interrupts, subgraph tasks in flight, a last step with other than
  one writer, another agent's stamp and values that a scratch copy
  (`graph.copy` with a fresh `InMemorySaver` and no store) does not
  reproduce. Values are stored in `dumpd` form and revived with
  `load(..., allowed_objects="messages")`. At trial time a
  `production_snapshot` case must reseed to the same serialized values
  (`fork_unsupported`) and every node case must leave exactly the entry
  point next (`entry_point_not_next`); `interrupt_after` stops after it and
  the checkpoint history proves no other node ran. The CLI needs
  `--graph module:attribute` to reach the user's graph and checkpointer.
- A `callable` entry point runs inside a one-node `StateGraph(dict)` without
  checkpointer, bound with the same pre-issued binding, so `prompts_for`
  works in it; an interrupt there is `interrupt_unsupported_in_callable`.
- `runner_custom` evaluators are declared with a `code_digest` (default:
  sha256 of the function source) that must equal the spec's. Model judges
  need `judge_model=` (a factory from the evaluator's model settings); a
  judge failure leaves the score null and never changes the outcome.
- `LocalPromptEngine.create_binding` refuses `exp:` keys like the registry,
  so `local_assignment` builds the trial binding itself from
  `LocalPromptEngine.resolve`; example 17 and the tests therefore run real
  views offline. Its view digest serves the simulation only.
  `tests/experiment_fakes.py` is a strict in-process runner tier (unknown
  members refused, lease fencing, the duplicate, conflict and stale table, a
  minimal tool proxy) built on those views.
