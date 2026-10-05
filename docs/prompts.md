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
```

The core package holds the registry client, the renderer and bundle
verification: `agenomic.prompts` needs only the core dependencies and never
imports LangChain or LangGraph. The `langchain` extra adds the conversion to
LangChain prompt templates.

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
- `PromptConflictError`: stale versions, revisions and generations, with
  the current value in `error.details["current"]` when the registry
  provides it.
- `RegistryUnavailableError`: `registry_unavailable`.

Unknown codes stay a plain `ApiError`. Calls that need Agenomic Cloud raise
`ApiError("cloud_required")` in local mode: `prompts.list`,
`versions.list`, `channels.list`, `channels.move_preview`,
`bindings.counterfactual`, `child_selectors`, `whoami` and
`prompts.export_bundle` (use `client.prompts.local.export_bundle` with a
signer). `client.prompts.local` raises it on a cloud client.

## Async

Every call has an `a*` twin (`aget`, `arender`, `acreate`, `apublish`,
`aresolve_agent`, `aexport_bundle`, `bindings.acreate`, `channels.ahistory`,
...). The async side uses one connection pool per event loop and runs disk
cache work in a thread, so it never blocks the loop.

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

## LangGraph

`bind_langgraph` pins every thread of a LangGraph graph to one release and
gives nodes their pinned prompts. See
[LangGraph managed prompts](integrations.md#langgraph-managed-prompts) and
the [LangGraph version matrix](langgraph-matrix.md).

## Not in this release

These parts of managed prompts are not in this SDK release yet: importing
prompts from existing code (scan, import plans, declaration files, runtime
registration), `from_langchain`, usage reporting, experiments and the
`agenomic-py prompts` commands.
