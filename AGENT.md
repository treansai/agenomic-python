# Using the Agenomic Python SDK: a guide for coding agents

This file is for AI coding agents (Claude Code, Cursor, Copilot, etc.)
integrating the `agenomic` package into a user's project. It is the
condensed, correct-by-construction map of the SDK: exact import paths,
canonical recipes, and the mistakes to avoid. Human-oriented docs live in
[`docs/`](docs/); contributor rules for developing this repo itself are
in [`AGENTS.md`](AGENTS.md).

## What this SDK is

`agenomic` records what an AI agent did, as signed, tamper-evident,
spec-shaped evidence. It does five things:

1. **Trace** agent runs (`@trace_agent_run` → `TraceEnvelope`, or the
   richer v0.3 `CanonicalRun` hash-chained event stream).
2. **Attest** history via ATEP: BLAKE3 causal hashes + ed25519
   signatures in binary `.atep` segments, verifiable offline.
3. **Track** production runs as spec event streams
   (`client.tracking`), feeding Review·Monitor·Protect
   (`client.rmp` / `review` / `monitor` / `protect`).
4. **Upload** (optional) to Agenomic Cloud. Everything else is fully
   offline; there is no hidden network I/O and no account needed.
5. **Manage prompts** (`client.prompts`): versioned, digest-verified
   prompts from the Agenomic Cloud registry or a local registry, pinned
   per thread and renderable offline from signed bundles.

It does **not** call LLMs for you, orchestrate agents, or make runs
reproducible (see `docs/non-determinism.md`).

## Install

```bash
pip install agenomic                 # core: no LLM SDKs pulled in
pip install "agenomic[openai]"       # + OpenAI auto-instrumentation
pip install "agenomic[anthropic]"    # + Anthropic auto-instrumentation
pip install "agenomic[langgraph]"    # + LangGraph tracing
pip install "agenomic[all]"          # everything
```

Python ≥ 3.10. Core deps: pydantic v2, httpx, cbor2, blake3,
cryptography, ulid-py. Integration modules import their SDKs lazily:
`import agenomic.integrations.openai` is safe without `openai` installed;
the `ImportError` fires only when you call `instrument_openai()`.
Managed prompts need no extra; `agenomic[langchain]` adds the conversion to
LangChain prompt templates.

## The 90% recipe: decorator + JSONL

```python
from agenomic.trace.decorator import trace_agent_run
from agenomic.exporters.jsonl import JsonlExporter

exporter = JsonlExporter("traces.jsonl")   # context manager; close() flushes

@trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
def handle(query: str) -> dict:
    return {"answer": query.upper()}
```

- Works on sync **and** async functions.
- Agent ids MUST match `agent://[a-z0-9-]+/[a-z0-9-]+` (lowercase).
- Exceptions re-raise; the envelope still exports with
  `error="ExcType: message"`.
- Exporter failures are logged (`agenomic.*` loggers), never raised into
  the wrapped function.

Record LLM/tool calls inside the wrapped function via the contextvar:

```python
from agenomic.trace.context import current_recorder
from agenomic.types import ModelCall, ToolCall

rec = current_recorder()          # None outside a traced run; always check
if rec is not None:
    rec.record_model_call(ModelCall(provider="openai", model="gpt-4o"))
    rec.record_tool_call(ToolCall(tool="search", protocol="mcp"))
    rec.add_label("env", "prod")
```

Or wrap the provider client once and let calls record themselves:

```python
from agenomic.integrations.openai import instrument_openai       # sync
from agenomic.integrations.anthropic import instrument_anthropic # sync
# async variants: instrument_openai_async / instrument_anthropic_async
oai = instrument_openai(OpenAI())        # wraps chat.completions.create
claude = instrument_anthropic(Anthropic())  # wraps messages.create
```

For MCP tools, record manually after the call (no-op outside a traced
run):

```python
from agenomic.integrations.mcp import trace_mcp_call
trace_mcp_call("server-1", "search", {"q": "x"}, result_dict)
```

## Recipe: signed, tamper-evident log (ATEP), fully offline

```python
from pathlib import Path
from agenomic.atep import AtepStore
from agenomic.crypto import SigningKey
from agenomic.exporters import AtepLocalExporter
from agenomic.trace.decorator import trace_agent_run

sk = SigningKey.generate()                     # or SigningKey.from_pem_file(...)
sk.write_pem_file(Path("key.pem"))             # PKCS#8 PEM, chmod 0600
store = AtepStore.open_or_init(Path("store"), "agent://acme/demo")

@trace_agent_run(agent_id="agent://acme/demo",
                 exporter=AtepLocalExporter(store, sk))
def handle(q: str) -> dict: ...

report = store.verify_all(sk.verifying_key())  # VerificationReport(ok=...)
root = store.compute_root_hash().hex()         # goes into ReleaseAttestation
```

Verify from the shell: `agenomic-py atep verify <seg.atep> --public-key
key.pem.pub`. Generate keys: `agenomic-py keys generate key.pem`.

## Recipe: canonical v0.3 run (hash-chained events)

```python
from agenomic.canonical import start_run

run = start_run("agent://acme/support", provider="openai", model="gpt-4o")
run.log_llm(prompt={"q": "hi"}, response={"a": "hello"})
run.log_tool_call(tool="search", arguments={"q": "x"}, result=[1])
trace = run.complete_run(output={"answer": "ok"})   # dict, spec agenomic/v0.3
```

Appending after `complete_run` raises `RuntimeError`. Hash helpers:
`from agenomic.canonical import content_hash, canonical_json, merkle_root`.
`content_hash(x)` returns `"blake3:<hex>"`; use it wherever the spec
wants an `input_hash`/`output_hash`.

## Recipe: production tracking + RMP

```python
from agenomic import Client

client = Client()   # local buffer; Client(api_key=..., base_url=...) = cloud

session = client.tracking.start(agent="agent://acme/claims",
                                release_id="release_123")
with session.step("classify"):                     # step events + failure capture
    session.model_call(provider="openai", model="gpt-4o")
    session.tool_call(tool="db.lookup", protocol="mcp")
    session.intent("verify_claim")
session.stop()                                     # idempotent
session.to_jsonl()                                 # local mode export
```

RMP loop (all local-first, same `Client`): `client.rmp.start(...)`,
`client.review.run(agent=...)`, `client.monitor.start(...)` →
`session.event({"type": "loop.detected"})` →
`client.monitor.findings(id)`, `client.protect.action_plan(...)`,
`client.review.approve_scenario_enrichment(proposal_id, reviewer=...)`.
See `docs/rmp.md` and `examples/09_rmp.py`.

## Recipe: redaction (always before export)

```python
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule

engine = RedactionEngine([
    RedactionRule(path="kwargs.password", mode=RedactionMode.MASK),
    RedactionRule(path="**.email", mode=RedactionMode.HASH),
    RedactionRule(path="kwargs.notes", mode=RedactionMode.TRUNCATE,
                  truncate_length=80),
])

@trace_agent_run("agent://acme/api", redaction=engine, exporter=exporter)
def login(*, user: str, password: str) -> dict: ...
```

Decorator inputs are shaped `{"args": [...], "kwargs": {...}}`, so rule
paths for function arguments start with `args.` or `kwargs.`. Wildcards:
`*` one segment, `**` any depth. `TRUNCATE` requires `truncate_length`.

## Recipe: cloud upload (async)

```python
from agenomic.client import AgenomicClient

client = AgenomicClient("https://cloud.example.com", api_key="sk-...")
try:
    await client.upload_traces(envelopes)          # list[TraceEnvelope]
    await client.upload_atep_segment("agent://acme/demo", segment_path)
finally:
    await client.aclose()
```

Retries + idempotency keys are automatic. `401` →
`AuthenticationError`; other failures → `CloudError` (both from
`agenomic.exceptions`). Use `SyncAgenomicClient` only outside a running
event loop. For continuous export, `HttpExporter(client)` batches
uploads behind the decorator.

## Recipe: managed prompts

```python
from agenomic import Client
from agenomic.prompts import BundleTrust, PromptBundle, thread_key

client = Client.from_env()
version = client.prompts.get("prm_planner:1")
messages = version.render_messages(
    {"customer": "Acme", "question": "Where is my order?"}
)

agent_id = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
workspace_id = client.workspace_id or client.whoami()["org_id"]
binding, bundle, created = client.bindings.create(
    agent_id,
    thread_key=thread_key(workspace_id, "conversation-42"),
    scope="thread",
    channel="production",
)
system = bundle.version("planner.instructions")

offline = PromptBundle.load(
    "prompt-bundle.json",
    expected_workspace_id=workspace_id,
    expected_agent_id=agent_id,
    trust=BundleTrust.from_pem_files("keys/orgkey_01.pem"),
)
```

- `Client.from_env()` reads `AGENOMIC_ENDPOINT`, `AGENOMIC_API_KEY`,
  `AGENOMIC_WORKSPACE_ID`, `AGENOMIC_PROMPT_CACHE_DIR` and
  `AGENOMIC_TIMEOUT`. Without an endpoint the client runs a local registry
  (`client.prompts.local`) with the same error codes.
- Execution reads take a version (`prm_x:7`), an alias (`prm_x@staging`) or
  a URI (`agenomic://<workspace>/prompts/prm_x/versions/7`).
- `bindings.create` pins the thread to one release; the first writer wins.
  Render the slot prompts of that bundle for the whole thread.
- The exported bundle file comes from
  `client.prompts.export_bundle(agent_id, channel="production", path=...)`;
  `PromptBundle.load` refuses it unless it is signed by a key in `trust` or
  pinned with `expected_bundle_digest`.
- Render errors (`PromptRenderError`) are raised before anything reaches a
  model. `render_text` is for `text` prompts, `render_messages` for `chat`.

## Import map (copy-paste correct)

```python
# RMP benchmarks bridge (cloud only)
from agenomic.benchmarks import (AgentTargetBridge, BridgeCapability,
                                 TurnRequest, TurnReply, serve_bridge)
```

```python
from agenomic import Client, __version__
from agenomic.trace.decorator import trace_agent_run
from agenomic.trace.context import current_recorder
from agenomic.trace.recorder import TraceRecorder
from agenomic.exporters import JsonlExporter, AtepLocalExporter, HttpExporter, MultiExporter
from agenomic.types import (TraceEnvelope, TraceInput, TraceOutput, ModelCall,
                            ToolCall, CallStatus, ReleaseAttestation,
                            WorkflowSpec, SystemSpec, validate_agent_id)
from agenomic.canonical import (start_run, CanonicalRun, content_hash,
                                canonical_json, event_hash, merkle_root)
from agenomic.atep import (AtepStore, AtepEvent, EventHeader, StreamId, Hlc,
                           SegmentReader, SegmentWriter, VerificationReport)
from agenomic.crypto import (SigningKey, VerifyingKey, blake3_hex,
                             hash_with_domain, canonical_cbor)
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule
from agenomic.client import AgenomicClient, SyncAgenomicClient, RetryPolicy
from agenomic.integrations.openai import instrument_openai
from agenomic.integrations.anthropic import instrument_anthropic
from agenomic.integrations.langgraph import instrument_langgraph
from agenomic.integrations.mcp import trace_mcp_call
from agenomic.providers.huggingface import HuggingFaceClient, HuggingFaceConfig
from agenomic.exceptions import (AgenomicError, CloudError, AuthenticationError,
                                 AtepError, CryptoError, RedactionError)
```

Managed prompts (`docs/prompts.md`):

```python
from agenomic.exceptions import ApiError
from agenomic.prompts import (BundleTrust, PromptBundle, PromptCache,
                              LocalPromptEngine, ManagedPromptVersion,
                              ExecutionBinding, PinnedPromptSet,
                              PromptVersionRef, PromptAliasRef, PromptUri,
                              parse_execution_ref, parse_prompt_ref,
                              thread_key, execution_key,
                              validate_content, render_content,
                              content_digest, manifest_digest,
                              canonical_json_v1)
from agenomic.prompts import (PromptRefError, PromptTemplateError,
                              PromptRenderError, PromptIntegrityError,
                              PromptBindingError, PromptConflictError,
                              RegistryUnavailableError)
from agenomic.prompts.resources import Page, PinnedRefs, RenderResult
from agenomic.prompts.authority import CloudBindingAuthority, counters
```

The LangChain bridge needs `agenomic[langchain]` and raises `ImportError` at
import without it:

```python
from agenomic.integrations.langchain_prompts import (to_langchain,
                                                     to_langchain_messages)
```

The top-level package intentionally exports only `Client` and
`__version__`; everything else is imported from its subpackage.

## Pitfalls (each one is a real failure mode)

1. **Invalid agent id.** `agent://Acme/Demo` fails validation: ids are
   lowercase `agent://org/name`.
2. **Recording outside a run.** `current_recorder()` is `None` unless
   you're inside a `@trace_agent_run`-wrapped call; integrations then
   silently no-op. That's by design: don't "fix" it by making a global
   recorder.
3. **Redaction paths that miss.** Decorator input is
   `{"args": ..., "kwargs": ...}`; a rule for `password` must target
   `kwargs.password` (or `**.password`). Unknown paths are silently
   skipped; test your rules.
4. **`SyncAgenomicClient` inside async code.** It calls `asyncio.run`
   per method and will raise inside a running loop. Use
   `AgenomicClient` (async) there.
5. **Forgetting to flush.** `HttpExporter` buffers: call
   `await aclose()` on shutdown. `JsonlExporter`/stores are fine
   (flush-per-line / write-per-event).
6. **Signing key hygiene.** Private PEMs are written `0600`; loading a
   looser file logs a warning. Never commit `key.pem`; commit
   `key.pem.pub` if verifiers need it.
7. **Expecting reproducibility.** Traces prove what happened and that it
   wasn't tampered with; they do not replay LLM behavior
   (`docs/non-determinism.md`).
8. **Hand-rolling hashes.** Use `content_hash` /
   `blake3_hex(canonical_cbor(x))`: ad-hoc `json.dumps` ordering will
   not match the Rust/cloud implementations.
9. **Cloud fallback assumptions.** `Client()` without `base_url` is
   local-only; nothing uploads implicitly. Passing `base_url` switches
   tracking/RMP to cloud with **no local buffering fallback**.
10. **Tag/enum typos in tracking events.** `session.event(type, ...)`
    accepts only the spec vocabulary (`TRACKING_EVENT_TYPES`); unknown
    types raise `ValueError`.
11. **Bare prompt ids in execution.** `client.prompts.get("prm_x")` raises
    `prompt_ref_unversioned`: there is no implicit latest. Use `prm_x:7` or
    an alias.
12. **Re-resolving aliases.** Every `get("prm_x@staging")` resolves again
    and may return another version, and aliases are never cached. Resolve
    once with `client.prompts.pin([...])` and keep the result.
13. **Raw thread ids.** The SDK sends the thread key as given; pass
    `thread_key(workspace_id, thread_id)`. `client.workspace_id` is `None`
    until it is configured or read through `client.whoami()`.
14. **Privileged keys in production.** Run agents with a `read` key.
    Publishing needs `write` or `admin`; alias and channel moves need a
    signed-in session in Agenomic Cloud, so the SDK has no promote and no
    rollback.
15. **Falling back during an outage.** A new thread raises
    `RegistryUnavailableError`; never catch it to render an inline string or
    a local copy instead. Only `CloudBindingAuthority` serves the cached
    binding of a thread that already has one.
16. **A fresh local registry per client.** Each `Client()` without
    `base_url` starts an empty in-memory prompt registry in a new random
    workspace. Keep one client, or use `LocalPromptEngine(state_path=...)`.

## Environment variables

In the library itself only `Client.from_env()` reads the environment; the
CLI, the providers and the examples read the rest:

| variable | consumer |
| -------- | -------- |
| `AGENOMIC_ENDPOINT`, `AGENOMIC_API_KEY` | `Client.from_env()` |
| `AGENOMIC_WORKSPACE_ID`, `AGENOMIC_TIMEOUT` | `Client.from_env()` |
| `AGENOMIC_PROMPT_CACHE_DIR` | `Client.from_env()` (disk prompt cache) |
| `AGENOMIC_BASE_URL`, `AGENOMIC_API_KEY` | `agenomic-py benchmark serve` |
| `HUGGINGFACE_API_TOKEN`, else `HF_TOKEN` | `HuggingFaceConfig.from_env()` |
| `HUGGINGFACE_ENDPOINT_URL` | `HuggingFaceConfig.from_env()` |
| `HUGGINGFACE_ORG` | `HuggingFaceConfig.from_env()` |
| `HUGGINGFACE_DEFAULT_MODEL` | `HuggingFaceConfig.from_env()` |
| `HUGGINGFACE_TIMEOUT_SECONDS` | `HuggingFaceConfig.from_env()` |
| `OPENAI_API_KEY` | `examples/04_openai_traced.py` |
| `AGENOMIC_ENDPOINT`, `AGENOMIC_API_KEY` | `examples/06_cloud_upload.py` |

Cloud credentials go to `Client(...)` / `AgenomicClient(...)` explicitly,
or to `Client.from_env()` through the variables above.

## Where to look next

| need | read |
| ---- | ---- |
| Full API surface | `docs/api-reference.md` |
| Envelope fields & recorder | `docs/tracing.md`, `docs/decorator.md` |
| ATEP format & verification | `docs/atep.md`, `docs/keys-and-signing.md` |
| v0.3 canonical runs | `docs/canonical.md` |
| Production tracking | `docs/tracking.md` |
| RMP loop | `docs/rmp.md` |
| Exporters | `docs/exporters.md` |
| Workflow/system manifests | `docs/orchestration.md` |
| Cloud client | `docs/cloud-upload.md` |
| CLI | `docs/cli.md` |
| Errors & logging | `docs/errors.md` |
| Managed prompts | `docs/prompts.md` |
| Runnable code | `examples/01`–`09` (offline ones run with zero setup) |
