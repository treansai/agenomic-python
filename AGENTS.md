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
  `workspace_mismatch`. `whoami()` also feeds the later privileged-key check.
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
  importer, the LangGraph adapter, experiments and the CLI commands land.
  Its Python blocks are meant to run in order: the local ones as written, the
  cloud ones against `FakePromptServer` (`tests/prompt_fakes.py`) with
  `transport=` added to the `Client`. No test executes them, so run them again
  after an API change. The `AGENT.md` recipe and the `README.md` example
  follow the same rule.
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
  `not_utf8`, one that does not parse as `syntax_error`. Excluded directory
  names are pruned without being listed; a file that matches an exclude
  pattern is listed as `excluded`. Symlinks that leave the root are ignored.
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
  content digests, and no secret in any content. Every plan received is
  verified (`plan_digest`, `summary`, unresolved items skipped) before it is
  printed or applied. Apply decisions default to the planned action with
  `blocked` turned into `skip`, cite `plan_digest`, carry a generated body
  `idempotency_key` and never send the `Idempotency-Key` header.
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
  and prints the plan, and applies it only with `--apply`.
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
  any was built (drawing, schemas, or an execution-scope state read, whose
  binding is only known from a checkpoint). `binding_missing` stays the case
  with no pin at all.
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
