# Managed prompts

`agenomic.prompts` gives Python code versioned prompts whose content is
verified by digest before it is rendered. A prompt version never changes once
published; mutable names (aliases, channels) are resolved once and recorded,
so every run can say exactly which prompt text it used.

The contract is RFC 0012 of `agenomic-spec`. Its conformance vectors are
vendored in this repository and run by the test suite, so references,
digests, rendering and bundle verification give the same answers here as in
the other Agenomic implementations.

```bash
pip install agenomic
pip install "agenomic[langchain]"
pip install "agenomic[yaml]"
```

The core package holds the registry client, the renderer, bundle
verification and the prompt scanner: `agenomic.prompts` needs only the core
dependencies and never imports LangChain or LangGraph. The `langchain` extra
adds the conversion to and from LangChain prompt templates, and the `yaml`
extra reads YAML prompts files.

## Concepts

- **Prompt.** A logical prompt `prm_support_planner` of kind `text`, `chat`
  or `fragment`.
- **Version.** One immutable content document, `prm_support_planner:7`,
  identified by its `sha256:` content digest.
- **Alias.** A mutable name for a version, `prm_support_planner@staging`.
  It is resolved to a version once, and the result records which alias and
  which alias generation it came from.
- **Fragment.** Reusable template text included with `{>name}`, pinned by
  prompt id, version and digest.
- **Release manifest.** The prompt pins of one agent version: each slot path
  (`planner.instructions`) names one prompt version, and child agents are
  pinned by release.
- **Channel.** A named pointer (`production`, `staging`) from an agent to one
  of its releases. In Agenomic Cloud only a signed-in person moves a channel.
- **Execution binding.** The pin of one thread, or one execution, to one
  release. The first writer wins: a thread keeps its release even when the
  channel moves later.
- **Bundle.** The exact prompt closure of one release as a JSON document,
  verified before use, either returned online or exported and signed for
  offline use.

## References

| You write | Meaning |
| --- | --- |
| `prm_support_planner` | the logical prompt, management calls only |
| `prm_support_planner:7` | immutable version 7 |
| `prm_support_planner@staging` | an alias, resolved once |
| `agenomic://<workspace>/prompts/<id>/versions/7` | version 7 in a workspace |

- There is no implicit latest version. Calls that read or render content
  refuse a bare id with `prompt_ref_unversioned`. Management calls
  (`versions.list`, `drafts`, `aliases`) take bare ids.
- Prompt ids match `prm_[a-z0-9]+([_-][a-z0-9]+)*` (at most 64 characters),
  versions are decimal without leading zeros (1 to 2147483647) and aliases
  match `[a-z][a-z0-9_-]{0,31}`.
- References are case sensitive and never trimmed, case-folded or
  percent-decoded. A refusal is `PromptRefError("prompt_ref_invalid")` with
  the reason in `error.reason` (for example `mixed_form` for
  `prm_x:7@staging`).
- A URI that names another workspace than the client's raises
  `prompt_ref_cross_workspace` and is never sent.
  `PromptUri.to_version_ref(workspace_id)` takes the current workspace for
  the same reason.

```python
from agenomic.prompts import (
    PromptAliasRef,
    PromptVersionRef,
    parse_execution_ref,
)

assert parse_execution_ref("prm_support_planner:7") == PromptVersionRef(
    "prm_support_planner", 7
)
alias = parse_execution_ref("prm_support_planner@staging")
assert alias == PromptAliasRef("prm_support_planner", "staging")
```

## Content documents

A version's content is an `agenomic.prompt_content/v1` document. All nine
members are always present (absence is `null`, `{}` or `[]`) and unknown
members are refused. A `fragment` prompt carries `text` content.

```python
SAFETY = {
    "schema": "agenomic.prompt_content/v1",
    "template_format": "agenomic-fstring/v1",
    "renderer_version": "1",
    "kind": "text",
    "body": "Never share internal notes.",
    "variables": {},
    "partials": {},
    "output_contract": None,
    "fragments": {},
}


def planner_content(safety_digest: str) -> dict:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "chat",
        "body": [
            {
                "role": "system",
                "content": "Plan for {customer} in a {tone} tone. {>safety}",
            },
            {"placeholder": "history", "optional": True},
            {"role": "user", "content": "{question}"},
        ],
        "variables": {
            "customer": {"type": "string", "required": True},
            "history": {"type": "messages", "required": False},
            "question": {"type": "string", "required": True},
            "tone": {"type": "string", "required": False},
        },
        "partials": {"tone": "formal"},
        "output_contract": None,
        "fragments": {
            "safety": {
                "prompt_id": "prm_safety",
                "version": 1,
                "content_digest": safety_digest,
            }
        },
    }
```

Templates use `agenomic-fstring/v1`, a strict subset of `str.format`:

- `{name}` is a variable, `{{` and `}}` are literal braces and `{>name}`
  includes the fragment declared under `name` in `fragments`.
- Anything else inside braces is a syntax error: format specs, conversions,
  attribute or index access, positional fields, whitespace and empty braces.
- Variable types are `string`, `integer`, `boolean`, `json` and `messages`.
  A `messages` variable is used only by a chat placeholder, which is how
  conversation history enters a prompt.
- A caller value for a name present in `partials` overrides the partial.
- Integers render as decimal, booleans as `true` and `false` and `json` values
  as canonical JSON. Substituted values are never scanned again. A float in a
  `json` value is refused.

Content is validated when it is published and again when a version is built
on the client. A template that matches a secret pattern is refused with
`prompt_secret_detected`.

## Local mode

`Client()` without `base_url` runs an in-process registry,
`LocalPromptEngine`. It returns the same error codes and statuses as
Agenomic Cloud and never touches the network, so the examples below run as
they are. Every `Client()` gets a new, empty registry in a random workspace
unless you pass `workspace_id`. For state that survives the process, use
`LocalPromptEngine(workspace_id, state_path=path)` directly.

```python
from agenomic import Client

client = Client()
prompts = client.prompts

prompts.create("prm_safety", name="Safety", kind="fragment")
safety = prompts.publish(
    "prm_safety", SAFETY, parent_version=None, change_message="First version"
)
prompts.create("prm_planner", name="Planner", kind="chat")
planner = prompts.publish(
    "prm_planner",
    planner_content(safety.content_digest),
    parent_version=None,
    change_message="First version",
)
print(planner.ref, planner.content_digest)
```

`publish` takes the version you edited as `parent_version`. When another
version was published meanwhile it raises
`PromptConflictError("prompt_version_conflict")` with the latest version in
`error.details["current"]`. Publishing the same content on the same parent
again returns the existing version.

### Rendering

```python
from agenomic.prompts import PromptRenderError

version = prompts.get("prm_planner:1")
variables = {"customer": "Acme", "question": "Where is my order?"}
for message in version.render_messages(variables):
    print(message.role, message.content)

history = [{"role": "user", "content": "Hello"}]
messages = version.render_messages({**variables, "history": history})

try:
    version.render_messages({"customer": "Acme"})
except PromptRenderError as error:
    print(error.code, error.reason, error.details["variable"])
```

- `render_text` renders a `text` prompt and `render_messages` a `chat`
  prompt; the other kind raises `kind_mismatch`. Neither flattens the other.
- Placeholder items are passed through unchanged. An optional placeholder
  without a value renders nothing. Whitespace is never trimmed.
- `compose(variables, history=...)` appends history after the rendered
  messages of a prompt that has no placeholder; on a prompt with a
  placeholder it raises `history_conflict`, so history never enters twice.
- Every render error is raised before anything reaches a model:
  `prompt_render_error` with `error.reason` among `missing_variable`,
  `unknown_variable`, `type_mismatch`, `float_not_allowed`,
  `integer_out_of_range`, `placeholder_not_list`, `history_conflict`,
  `kind_mismatch` and the other reasons of RFC 0012.
- `client.prompts.render(ref, variables)` fetches and renders in one call. Its
  `RenderResult` carries `kind`, `ref`, `content_digest`, `text` or
  `messages`, and `rendered_hash`, the digest of the rendered document as
  RFC 0012 defines it, so every implementation computes the same value.

### Drafts and aliases

```python
draft = prompts.drafts.save(
    "prm_planner",
    planner_content(safety.content_digest),
    base_version=1,
    expected_revision=0,
)
print(draft.revision, draft.validation["ok"])

prompts.aliases.move("prm_planner", "staging", version=1, expected_generation=0)
staged = prompts.get("prm_planner@staging")
print(staged.ref, staged.resolved_from)

pinned = prompts.pin(["prm_planner@staging", "prm_safety:1"])
assert pinned["prm_planner@staging"].ref == staged.ref
```

- One draft exists per prompt. `drafts.save` sends the revision you read
  (`0` creates the draft); a stale revision raises
  `PromptConflictError("prompt_draft_conflict")` with the current revision
  in `error.details["current"]`.
- Each `get("prm_x@alias")` resolves the alias again. To keep one decision
  for a whole task, call `pin(refs)` once and keep the returned `PinnedRefs`.
- In Agenomic Cloud an alias move needs a signed-in session: every API key
  gets `session_required`. Local mode moves the alias.

### Releases, channels and bindings

The local registry also simulates releases and channels.
`client.prompts.local` is the engine; in Agenomic Cloud a channel move is an
approved, session-only action and these calls do not exist.

```python
from agenomic.prompts import thread_key

engine = client.prompts.local
agent_id = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
release_id = engine.create_release(
    agent_id, {"planner.instructions": "prm_planner:1"}
)
engine.move_channel(agent_id, "production", release_id, expected_generation=0)

key = thread_key(client.workspace_id, "conversation-42")
binding, bundle, created = client.bindings.create(
    agent_id, thread_key=key, scope="thread", channel="production"
)
system = bundle.version("planner.instructions")
print(binding.release_name, created, system.ref)
```

The local registry also exports signed bundles (see
[Offline bundles](#offline-bundles)), with a signing key you create:

```python
from agenomic.crypto import SigningKey
from agenomic.prompts import BundleTrust, PromptBundle

signer = SigningKey.generate()
document = engine.export_bundle(agent_id, signer=signer, channel="production")
offline = PromptBundle.load(
    document,
    expected_workspace_id=client.workspace_id,
    expected_agent_id=agent_id,
    trust=BundleTrust.from_pems({signer.key_id: signer.public_pem()}),
)
print(offline.signed, offline.version("planner.instructions").ref)
```

## Agenomic Cloud

```python
from agenomic import Client

client = Client(api_key="agm_...", base_url="https://agenomic.example")
```

`Client.from_env()` reads the same settings from the environment, and its
keyword arguments win:

- `AGENOMIC_ENDPOINT` (the same name as the `agm` CLI) and
  `AGENOMIC_API_KEY`;
- `AGENOMIC_WORKSPACE_ID`, the workspace (organization) uuid, lowercase;
- `AGENOMIC_PROMPT_CACHE_DIR`, which enables the disk cache;
- `AGENOMIC_TIMEOUT`, in seconds.

Without `workspace_id`, the first call that needs it reads `GET /v1/whoami`
once, and `client.workspace_id` is `None` until then. When the configured
workspace and the key's workspace differ, or an answer names another
workspace, every call raises `workspace_mismatch`. For a private certificate
authority set `SSL_CERT_FILE` or `SSL_CERT_DIR`; TLS verification cannot be
turned off. Use the client as a context manager, or call `close()`
(`await aclose()`), to release its pooled connections.

### Keys

- Run production agents with a `read` key. It reads, resolves and creates
  execution bindings, and every publishing call refuses it.
- `create`, `publish` and `drafts.save` need a `write` or `admin` key. A
  `read` key gets the registry's 403 `api_key_scope_insufficient` as is.
- So does every import call, planning included: `import_report`,
  `apply_import`, `plan_declarations`, `apply_declarations`,
  `register_runtime` and `agenomic-py prompts import`.
- Alias moves and channel moves need a signed-in session in Agenomic Cloud;
  every API key gets `session_required`.
- An API key binds, resolves or exports only a release that is `approved`,
  in `production`, or the current target of one of the agent's channels.
  Any other release gets 403 `session_required` with
  `error.reason == "ungoverned_release"`. A `rejected` or `rolled_back`
  release is never bound or exported (409 `release_not_bindable`).

### Reading and publishing

```python
version = client.prompts.get("prm_planner:1")
staged = client.prompts.get("prm_planner@staging")
page = client.prompts.list(tags=["support"], limit=20)
for summary in page.items:
    print(summary.prompt_id, summary.latest_version)
versions = client.prompts.versions.list("prm_planner")
```

- A version or URI is read from the cache first. On a miss the client
  fetches the version with its fragment closure, verifies every digest,
  checks that the answer is the requested version of the client's workspace,
  and caches it.
- An alias always goes to the registry. Only the concrete version is cached,
  never the alias target, so an alias is never served from cache, not even
  during an outage. The result's `resolved_from` records the alias and its
  generation.
- Prompt and version lists return a `Page` with `items` and `next_cursor`.
- `create`, `publish`, `drafts.get`, `drafts.save` and `aliases.get` work as
  in local mode, with the key rules above.

### Agent resolution and execution bindings

```python
from agenomic.prompts import thread_key

agent_id = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
bundle = client.prompts.resolve_agent(agent_id, channel="production")
print(bundle.release_id, bundle.slots())

workspace_id = client.workspace_id or client.whoami()["org_id"]
key = thread_key(workspace_id, "conversation-42")
binding, bundle, created = client.bindings.create(
    agent_id, thread_key=key, scope="thread", channel="production"
)
binding, bundle = client.bindings.get(agent_id, binding.binding_id)
```

- `resolve_agent` returns the closure of the release a channel (or a
  `release_id`) points to now. It creates no binding.
- `bindings.create` is create-or-get on the agent and the thread key. The
  first call pins the thread; later calls return the same binding with
  `created=False`, even after the channel has moved. A call with another
  selector, scope or expected manifest digest raises
  `PromptBindingError("execution_binding_conflict")`.
- Pass exactly one of `channel` and `release_id`. `child_selectors` name the
  release of child agents that the manifest does not pin; the registry
  validates them but, in this release, pins child agents from the release
  manifest only.
- The SDK sends the thread key as given. Compute it with
  `thread_key(workspace_id, thread_id)` (or `execution_key` for a single
  execution), which hashes the identifier, so the registry never sees an
  application thread id. Look a thread up later with the same function.
- The returned bundle is verified against the binding's manifest digest and
  child manifest digests before it is returned.
- `bindings.counterfactual(agent_id, parent_binding_id, thread_key=...,
  release_id=...)` creates a child binding that pins another release for the
  same input.

### Channels

`client.channels` is read only: `list`, `get`, `history` (every event, page
after page) and `move_preview`. There is no promote and no rollback in the
SDK: moving a channel is a signed-in, approved action in Agenomic Cloud.
Registries that do not serve move previews yet answer 404 to `move_preview`.

### Reporting prompt usage

`client.bindings.report_usage` tells Agenomic Cloud which prompts a binding
rendered, which feeds its inventory of observed prompts:

```python
result = client.prompts.render(
    "prm_planner:1", {"customer": "Acme", "question": "Where is my order?"}
)
client.bindings.report_usage(
    agent_id,
    binding.binding_id,
    [
        {
            "slot_path": "planner.instructions",
            "prompt_ref": str(result.ref),
            "content_digest": result.content_digest,
            "rendered_hash": result.rendered_hash,
            "count": 1,
            "first_at": "2026-10-05T09:00:00Z",
            "last_at": "2026-10-05T09:00:00Z",
        }
    ],
)
```

- An observation carries references and hashes only. Its members are
  `slot_path`, `node_path`, `prompt_ref`, `content_digest`, `rendered_hash`,
  `overlay` (`digest` and `position`), `alias`, `alias_generation`,
  `unmanaged`, `role_layout`, `count`, `first_at` and `last_at`. The SDK
  raises `ValueError` before sending anything else, prompt text included.
- `prompt_ref` names a version, never an alias or a bare id. A prompt that
  is not managed is reported with `"unmanaged": True` instead.
- `rendered_hash` is the `rendered_hash` of a render (`sha256:`); the
  BLAKE3 `input_hash` of the tracking handler is refused.
- For a version read through an alias, add `alias` and `alias_generation`
  from `version.resolved_from`.
- Long lists are sent 500 observations per request, and an empty list sends
  nothing. `bind_langgraph` never reports usage on its own.

## Offline bundles

An exported bundle is a signed `agenomic.prompt_bundle/v1` document with the
exact prompt closure of one release, its governance state and an expiry.

```python
bundle = client.prompts.export_bundle(
    agent_id,
    channel="production",
    expires_in_days=30,
    path="prompt-bundle.json",
)
print(bundle.prompt_bundle_digest)
```

`export_bundle` verifies the export with the organization key from
`GET /v1/signing-keys/{key_id}` (or the `trust` you pass) before writing it.
It accepts a release that is not approved yet, since the target of an
unprotected channel may be exported; loading the file applies the governance
check again.
The registry refuses to export a release whose pinned prompt matches a secret
pattern (`prompt_bundle_export_blocked`). Save the public key of
`GET /v1/signing-keys/{key_id}` as `<key_id>.pem` for the machines that load
the bundle.

```python
from agenomic.prompts import BundleTrust, PromptBundle

bundle = PromptBundle.load(
    "prompt-bundle.json",
    expected_workspace_id="0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f",
    expected_agent_id="2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c",
    trust=BundleTrust.from_pem_files("keys/orgkey_01.pem"),
)
system = bundle.version("planner.instructions")
```

`PromptBundle.load` refuses the bundle at the first failed step:

1. the document is in the Agenomic JSON subset (no floats, no lone
   surrogates, no NUL) and has the bundle schema;
2. it is signed by a key in `trust` (the embedded public key is never
   trusted), or its `prompt_bundle_digest` equals `expected_bundle_digest`;
   a bundle that is neither is refused with `bundle_untrusted_key`;
3. a signed bundle carries an `expires_at` (`bundle_incomplete` otherwise)
   and it has not passed (`bundle_expired`);
4. every prompt matches its digest, and so does the whole closure;
5. every manifest matches its digest (and `expected_manifest_digest`);
6. the closure is exact: every slot, fragment and child is present, and
   nothing else (`bundle_incomplete`);
7. the bundle belongs to `expected_workspace_id` and `expected_agent_id`
   (`bundle_scope_mismatch`), so another agent's bundle signed by the same
   organization key is refused;
8. a signed bundle whose release is not approved raises `bundle_ungoverned`
   unless `allow_ungoverned_bundle=True`. A digest-pinned bundle skips this
   step: the pin is the operator's approval.

Pin with `expected_bundle_digest` rather than `expected_manifest_digest`
alone: only the bundle digest covers child agents and their prompts.
`BundleTrust.from_pem_files` uses each file name without its extension as the
key id; `BundleTrust.from_pems({key_id: pem})` takes explicit ids. Online
answers are loaded with `PromptBundle.from_online_response`, which refuses a
signed document.

Know the limits of offline use:

- A disconnected process cannot learn that a release, a key or a channel was
  revoked.
- A bundle stays usable until its `expires_at` (always set on an export:
  30 days by default, at most 365) or until the operator removes it.
- Organization signing keys have no revoked state. After a key compromise,
  rotate the key and remove its key id from every `BundleTrust`.
- Holding an artifact never means being authorized to run it now.

## Cache

`Client(prompt_cache=PromptCache(directory))` adds a disk tier to the default
memory cache.

- Every key includes the workspace; a display name or a bare prompt id is
  never a key.
- Only immutable records are cached: prompt versions, verified closures and
  execution bindings. Alias targets, channel pointers and credentials are
  never cached.
- Files are written atomically with mode 0600 in 0700 directories, and every
  read is verified again. A mismatch raises `cache_conflict`, which the
  online client treats as a miss.

## Registry outages

The registry is unavailable after a transport error or a 429, 502, 503 or
504 answer. Reads, `publish`, alias resolution and binding calls retry first
(after 0.2 s, 0.8 s and 3.2 s, or the `Retry-After` delay); `create`, draft
saves and a 500 are never retried. The SDK then raises
`RegistryUnavailableError` (`registry_unavailable`, with the cause in
`error.details["cause"]`), except in these cases:

- `client.prompts.get("prm_x:7")` serves a cached, verified version.
- `CloudBindingAuthority` (in `agenomic.prompts.authority`) serves the
  cached binding of a thread that already has one, with the same scope and
  selector, after verifying its cached closure again. It returns
  `created=False`, logs one WARNING on the `agenomic.prompts` logger and
  increments `counters()["registry_outage_cached_binding_total"]`.

`client.bindings.create` itself never falls back. A new thread, an alias or
an uncached version raises during an outage: nothing falls back to a bundle,
a cached latest or an inline string, so one conversation never splits across
two releases. A cache never overrides an authorization answer: on a 401, 403
or 404, `CloudBindingAuthority` evicts the cached binding of that thread
before raising.

A process restarted during an outage starts with an empty memory cache, so
it finds a thread's binding only in the disk cache
(`AGENOMIC_PROMPT_CACHE_DIR`), and only when the client knows its
`workspace_id` without asking the registry. The LangGraph adapter then
binds with its credential check pending, serves only cached bindings, and
runs the check before any binding request once the registry answers
([Integrations](integrations.md#credential-check-and-registry-outages)).

```python
from agenomic.prompts.authority import CloudBindingAuthority, counters

authority = CloudBindingAuthority(client)
binding, bundle, created = authority.create_or_get(
    agent_id, key, "thread", {"channel": "production"}
)
print(counters()["registry_outage_cached_binding_total"])
```

## Importing existing prompts

Prompts that already live in your code come under management in three
steps, and nothing is written before the last one:

1. `agenomic-py prompts scan` reads your source files statically and writes
   a discovery report.
2. The report, never the source code, is uploaded. Agenomic Cloud answers
   with an import plan that proposes one action per prompt found.
3. You review the plan and apply it. The apply call cites the plan's
   `plan_digest`, so exactly the reviewed plan is written.

Neither the scanner nor the import ever changes a source file. Replacing a
string in your code by a managed prompt stays your change: the SDK has no
rewrite command. The report and plan formats
(`agenomic.prompt_discovery_report/v1`, `agenomic.prompt_import_plan/v1`)
are defined by agenomic-spec.

### Scanning code

```bash
agenomic-py prompts scan . --out report.json
```

The command writes the report and prints the number of scanned files and
of candidates per status on standard error. `scan_paths` does the same from
Python:

```python
from agenomic.prompts.discovery import scan_paths

report = scan_paths(["app"], root=".", label="support-agent")
for candidate in report["candidates"]:
    proposal = candidate["proposal"]
    print(candidate["status"], proposal["slot_path"], proposal["prompt_id"])
```

- The scanner parses with the Python 3.10 grammar and nothing more: no
  module is imported, executed or evaluated, so a module that fails on
  import is scanned like any other.
- It reads the `.py` files under the given paths, skips
  `DEFAULT_EXCLUDES` (`.git`, `.venv`, `node_modules`, `build`, `dist`, ...)
  and your `exclude` globs, and parses at most `max_files` files (4000) of
  at most `max_file_bytes` each (512 KiB). Every file is listed in the
  report with its sha256, as `scanned` or `skipped` with the reason, except
  a file whose path holds a backslash, which the report format cannot carry.
- It finds LangChain `PromptTemplate`, `ChatPromptTemplate`, messages and
  message templates, the prompt of `create_react_agent` and
  `create_agent`, and module string constants whose name ends in `prompt`,
  `template`, `instructions` or `system_message`. A prompt used by exactly
  one LangGraph node (`add_node`) gets that node as its `node_path`, and a
  node that runs a compiled subgraph or an agent is listed for mapping to
  a child agent.
- The report holds repository-relative paths, file hashes and the extracted
  templates only. `label` (default: the root directory name) names the
  root, and `commit` records a git commit; the SDK never reads git itself.

Each candidate has a status:

- `supported`: the template was ported exactly to `agenomic-fstring/v1`.
  Only these candidates carry `content` and its `content_digest`.
- `unsupported`: a known construct with a refused feature, such as a
  mustache or jinja2 template, a format spec or a callable partial.
- `unresolved`: the prompt is built at runtime: an f-string, `.format`,
  `hub.pull`, the result of a call, a subgraph node.
- `blocked_secret`: the template contains a credential. The report keeps
  the pattern id and its position (`secret_findings`), never the text.

Each candidate also proposes a prompt id, a slot path such as
`planner.instructions` and a usage, and explains every decision with issue
codes (`python_fstring`, `dynamic_template`, `callable_partial`,
`variable_types_defaulted`, ...) that never quote the source. Imported
variables are typed `string`: review the types in the plan.

### Import plans

Upload the report with a `write` key:

```bash
agenomic-py prompts import report.json --agent-id "$AGENT_ID"
```

The command reads the client settings from the environment
(`AGENOMIC_ENDPOINT` and `AGENOMIC_API_KEY` are required), prints the
import with its plan, and the plan id and `plan_digest` on standard error.
`--apply` applies every proposed action right away. From Python:

```python
plan = client.prompts.import_report(report, agent_id=agent_id)
print(plan.import_id, plan.plan_digest, plan.plan["summary"])
for item in plan.items:
    print(item["item_id"], item["action"], item["prompt_id"])
```

- The server computes one item per candidate with a proposed action:
  `create_prompt`, `create_version` (on top of `base_version`, the latest
  version when the plan was made), `reuse_version` (a version with the same
  content digest exists), `map_slot_only`, `skip` or `blocked`.
- An unresolved candidate is always `skip` with slot status `unresolved`. A
  plan never claims complete coverage: `plan.plan["summary"]["unresolved"]`
  counts what the scanner could not port.
- `import_report` verifies the plan digest and the summary before it
  returns. Uploading the same report for the same agent again returns the
  same plan (`plan.replayed`) until it expires (`plan.expires_at`).

Applying cites the digest of the plan you reviewed:

```python
result = client.prompts.apply_import(
    plan.import_id, plan_digest=plan.plan_digest, items=plan.decisions()
)
for outcome in result["results"]:
    print(outcome["item_id"], outcome["outcome"], outcome.get("version"))
```

- `plan.decisions()` accepts every proposed action and skips blocked items.
  Edit a decision to rename its `prompt_id`, change its `slot_path` or skip
  it; every item of the plan is listed once.
- A plan that changed, or a `create_version` whose base is no longer the
  latest version, raises `PromptImportError("prompt_import_plan_stale")` and
  writes nothing. A plan is applied once (`prompt_import_already_applied`),
  and an expired plan raises `prompt_import_expired`: upload the report
  again for a new plan.
- The SDK generates `idempotency_key` when you omit it. Pass your own to
  retry an apply safely: the same key returns the first result.
- `mode="draft"` saves each item as the prompt's draft instead of
  publishing a version. Outcomes are `created`, `versioned`, `unchanged`,
  `drafted`, `mapped` or `skipped`.
- `declare_slots=True` also records the slots in the agent's slot
  declarations, and needs `expected_slots_revision`, the agent's current
  slot revision. No SDK call reads that revision for a report import yet;
  `plan_declarations` returns it for a prompts file.

### Prompts files

An `agenomic.prompts_file/v1` declares a family of prompts and, optionally,
the slots of one agent. Keep it in your repository as `prompts.yaml`:

```yaml
schema: agenomic.prompts_file/v1
agent_id: 2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c
prompts:
  - prompt_id: prm_support_safety
    kind: fragment
    name: Support safety rules
    content:
      kind: text
      body: Never share internal notes.
      variables: {}
  - prompt_id: prm_support_planner
    kind: chat
    name: Support planner
    content:
      kind: chat
      body:
        - role: system
          content: |-
            You plan support work for locale {locale}.
            {>safety}
        - role: user
          content: "{question}"
      variables:
        locale: { type: string, required: false }
        question: { type: string, required: true }
      partials: { locale: en }
      fragments:
        safety: { prompt_id: prm_support_safety }
slots:
  - slot_path: planner.instructions
    node_path: planner
    usage: instructions
    prompt_id: prm_support_planner
```

```python
from pathlib import Path

planned = client.prompts.plan_declarations(Path("prompts.yaml"))
print(planned.plan_digest, planned.slots)
applied = client.prompts.apply_declarations(
    Path("prompts.yaml"),
    plan_digest=planned.plan_digest,
    expected_slots_revision=planned.slots["revision"],
)
```

- Pass a file as a `Path`. A `str` or `bytes` value is read as the document
  text itself, and a mapping is sent as it is.
- YAML needs `agenomic[yaml]` (`yaml_support_not_installed` otherwise). The
  SDK converts it to JSON under the `agenomic-yaml/1` profile before upload,
  so every Agenomic client reads a file the same way: one document, no
  anchors, aliases, merge keys or tags, no duplicate keys, no floats, and
  only `true` and `false` are booleans (`yes` and `on` stay strings).
- In each `content`, `schema`, `template_format`, `renderer_version`,
  `partials`, `output_contract` and `fragments` may be omitted. A fragment
  entry `{ prompt_id }` names a prompt of the same file, and
  `{ prompt_id, version }` an existing version.
- `plan_declarations` is a dry run: the plan is not stored, and
  `planned.slots` summarizes the slot changes with the agent's current
  `revision`. `apply_declarations` cites the plan digest; the server
  computes the plan again and raises `prompt_import_plan_stale` when
  anything moved, including the `expected_latest_version` of a prompt.
- A file that declares slots needs `expected_slots_revision`. A revision
  that moved raises `PromptConflictError("agent_prompt_slots_conflict")`.
- Applying creates prompts and publishes the versions whose content
  changed. It never moves an alias or a channel and never creates a
  release.

### Runtime registration

Templates that exist only as live LangChain objects are registered from the
running application, with `agenomic[langchain]`:

```python
from langchain_core.prompts import ChatPromptTemplate

triage = ChatPromptTemplate.from_messages(
    [("system", "Sort the ticket for the {team} team."), ("human", "{ticket}")]
)
runtime_plan = client.prompts.register_runtime(
    agent_id, {"triage.system": triage}
)
print(runtime_plan.plan_digest, runtime_plan.plan["summary"])
```

- Each template is converted with `from_langchain` ([LangChain](#langchain))
  into a discovery report labeled `runtime_registration`, which is uploaded
  like a scanned one. Slot paths are lowercase, such as `triage.system`.
- It returns the plan only. Apply it with `apply_import`, as above. Nothing
  calls it implicitly, and `bind_langgraph` never registers prompts.

### Command line

`agenomic-py prompts` has five commands:

```bash
agenomic-py prompts scan . --out report.json --exclude "tests"
agenomic-py prompts import report.json --agent-id "$AGENT_ID" --apply
agenomic-py prompts render planner.yaml --vars vars.json
agenomic-py prompts digest planner.yaml
agenomic-py prompts bundle-verify prompt-bundle.json \
  --workspace "$WORKSPACE_ID" --agent "$AGENT_ID" \
  --trust-key keys/orgkey_01.pem
```

- `scan` takes `--root` (default: the scanned directory), `--label`,
  `--commit`, `--exclude` (repeatable), `--max-files` and
  `--max-file-bytes`.
- `import` takes `--apply`, `--mode publish|draft`, `--declare-slots` with
  `--slots-revision`, and `--idempotency-key`.
- `render` renders a prompt file with the variables of a JSON file, or
  fetches a version such as `prm_planner:1` with the environment settings.
  It prints the text or messages with `content_digest` and `rendered_hash`.
  `digest` prints the content digest of a prompt file. A prompt file is an
  `agenomic.prompt_file/v1` with the full content of one prompt, or a bare
  content document, as JSON or YAML.
- `bundle-verify` runs the checks of `PromptBundle.load` and prints the
  bundle and manifest digests, the refs and the slots. Repeat `--trust-key`
  for several keys, or pin the bundle with `--expect-bundle-digest`.
- The exit code is 0 on success, 1 when Agenomic or a check refused
  (`error: <code> (<reason>): <message>`), and 2 for a usage, configuration
  or file error.

## Errors

Every error is an `agenomic.exceptions.ApiError` with `code`, `status` (0
when no request was sent), `message` and `details`; `request_id`, `reason`
and `errors` read `details`. Server codes map to these classes, all exported
by `agenomic.prompts`:

- `PromptRefError`: invalid, unversioned or cross-workspace references,
  `workspace_mismatch`.
- `PromptTemplateError`: `prompt_template_invalid` (with `error.errors`),
  `prompt_secret_detected`, `prompt_content_too_large`,
  `prompt_kind_mismatch`, fragment cycles and depth.
- `PromptRenderError`: `prompt_render_error`, with `error.reason`.
- `PromptIntegrityError`: digest mismatches, bundle refusals,
  `cache_conflict` and the registry's `artifact_integrity_error`.
- `PromptBindingError`: `execution_binding_conflict`, `binding_mismatch`,
  `slot_not_in_manifest`, `child_agent_not_pinned`, `release_not_bindable`,
  `session_required`.
- `PromptImportError`: `prompt_import_invalid` (an invalid report, prompts
  file or YAML document, with the failures in `error.errors`),
  `prompt_import_plan_stale`,
  `prompt_import_expired`, `prompt_import_already_applied`,
  `prompt_import_item_blocked` and `yaml_support_not_installed`.
- `PromptConflictError`: stale versions, revisions and generations,
  `agent_prompt_slots_conflict` included, with the current value in
  `error.details["current"]` when the registry provides it.
- `RegistryUnavailableError`: `registry_unavailable`.

Unknown codes stay a plain `ApiError`. Calls that need Agenomic Cloud raise
`ApiError("cloud_required")` in local mode: `prompts.list`,
`versions.list`, `channels.list`, `channels.move_preview`,
`bindings.counterfactual`, `child_selectors`, `whoami`,
`prompts.export_bundle` (use `client.prompts.local.export_bundle` with a
signer), the import calls (`import_report`, `apply_import`,
`plan_declarations`, `apply_declarations`, `register_runtime`) and
`bindings.report_usage`. `client.prompts.local` raises it on a cloud
client.

## Async

Every call has an `a*` twin (`aget`, `arender`, `acreate`, `apublish`,
`aresolve_agent`, `aexport_bundle`, `aimport_report`, `aapply_import`,
`bindings.acreate`, `bindings.areport_usage`, `channels.ahistory`, ...).
The async side uses one connection pool per event loop and runs disk cache
work in a thread, so it never blocks the loop.

```python
import asyncio

from agenomic import Client


async def main() -> None:
    client = Client(api_key="agm_...", base_url="https://agenomic.example")
    async with client:
        version = await client.prompts.aget("prm_planner:1")
        print(version.content_digest)


asyncio.run(main())
```

## LangChain

```python
from agenomic.integrations.langchain_prompts import (
    to_langchain,
    to_langchain_messages,
)

template = to_langchain(version)
messages = to_langchain_messages(version.render_messages(variables))
```

- `to_langchain` (also `version.to_langchain()`) builds a `PromptTemplate`
  or `ChatPromptTemplate` from the expanded template, with placeholders as
  `MessagesPlaceholder` and the partials applied. Its metadata carries
  `agenomic_prompt_ref` and `agenomic_prompt_content_digest`.
- It refuses `integer`, `boolean` and `json` variables (`unsupported_content`)
  because LangChain would print Python values where Agenomic renders `true`
  and canonical JSON. `render_messages` stays the authoritative renderer.
- `to_langchain_messages` maps `system`, `user` and `assistant` to LangChain
  messages and passes LangChain messages through.

`from_langchain` goes the other way, from a LangChain template to a content
document:

```python
from agenomic.integrations.langchain_prompts import from_langchain

imported = from_langchain(template, resolver=client.prompts.get)
print(imported.status, imported.exact, imported.content_digest)
```

- It accepts f-string templates within the `agenomic-fstring/v1` grammar,
  `MessagesPlaceholder` and scalar partials. Mustache and jinja2 templates,
  format specs, callable partials, content blocks and other message roles
  give status `unsupported`, and a credential gives `blocked_secret` with
  `secret_findings` but no text. `issues` explains each decision, and only
  a `supported` result carries `content`.
- A template made by `to_langchain` carries its ref and digest in its
  metadata. With a `resolver` (a function from a ref to a version, such as
  `client.prompts.get`), an unchanged template returns that published
  version exactly (`imported.exact`). A changed one is imported from its
  structure, with the issue `export_metadata_stale`.
- Any other object raises `TypeError`. `register_runtime` uses
  `from_langchain` to plan an import of live templates.

## LangGraph

`bind_langgraph` pins every thread of a LangGraph graph to one release and
gives nodes their pinned prompts. See
[LangGraph managed prompts](integrations.md#langgraph-managed-prompts) and
the [LangGraph version matrix](langgraph-matrix.md).

## Not in this release

These parts of managed prompts are not in this SDK release yet:

- experiments;
- reading an agent's slot declarations and its candidate agent versions;
- reading or listing stored import plans, and registering a prompt family
  in one call;
- the inventory of observed prompts, which `report_usage` feeds;
- rewriting source code to use managed prompts: neither the scanner nor the
  import modifies a file.
