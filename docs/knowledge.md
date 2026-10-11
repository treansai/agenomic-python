# Knowledge bases

A knowledge base (KB) is a governed workspace resource of Agenomic Cloud:
documents with a mutable working set, immutable content-addressed versions,
a published pointer, agent bindings and retrieval events. The SDK reads and
manages knowledge bases over HTTP with `client.knowledge`, and gives agents
a LangChain tool and retriever that work unchanged in LangGraph graphs.

Knowledge bases need Agenomic Cloud: a `Client()` without `base_url` raises
`ApiError("cloud_required")` on every `client.knowledge` call. Every method
has an `a*` twin for asyncio (`search` and `asearch`, `get` and `aget`).

## Quick start

```python
from agenomic import Client

client = Client(api_key="agm_...", base_url="https://agenomic.example")

kb = client.knowledge.get("kb_customer_support")
print(kb.record.name, kb.published_version)

found = kb.search("How long is the refund window?", version="v3", top_k=5)
for result in found.results:
    print(result.rank, result.path, result.heading_path, result.score)
    print(result.citation.uri)

document = kb.get_document("kdoc_01jb3m5q7s9v1x3z5b7d9f0001")
section = kb.get_section(
    document_id="kdoc_01jb3m5q7s9v1x3z5b7d9f0001",
    section="Refund Policy",
    version="published",
)
print(section.content)
```

`client.knowledge.get(kb_id)` reads the knowledge base and returns a
`KnowledgeBase` handle with its record, statistics and health.
`client.knowledge.base(kb_id)` returns the same handle without a request.
The handle offers `search`, `query`, `get_document`, `get_section`,
`versions`, `publish` and `refresh`.

`get_section` takes a section id (`sec_` plus 16 hex digits) or a heading,
a heading path (`Authorization/Tool Permissions`) or an anchor. A heading is
resolved through the structured query route (`get_section` operation),
which matches the document by id first, then by path, file name or title.
Pass `document_id=` (the SDK then keeps only sections of that document) or
`document="faq/refunds.md"`.

## Versions

Every call that takes a version accepts the same selectors:

| Value | Meaning |
| ----- | ------- |
| `3`, `"3"`, `"v3"` | the immutable version 3 |
| `"published"` | the version the KB currently publishes |
| `"draft"` | the working set (editor sessions only) |

The SDK sends numbers on the wire. Omitting the version lets the gateway
use the published version. Pin a number when you need the same evidence
tomorrow: a published pointer moves when someone publishes. Routes whose
path names a version (`versions.get`, `diff`, `verify`, `publish`) take a
number only.

## Search, query and answer

```python
found = client.knowledge.search(
    "kb_customer_support",
    "refund window",
    version=3,
    mode="hybrid",
    top_k=5,
    filters={"collections": ["faq"], "tags": ["refunds"]},
    include_context=True,
)
print(found.context)
print(found.retrieval.event_id, found.retrieval.version_manifest_digest)

matches = client.knowledge.query(
    "kb_customer_support", 'get "Refund Policy" from "Refunds and Returns"'
)
listing = client.knowledge.query(
    "kb_customer_support",
    operation={"op": "list_children", "section": "Refunds and Returns"},
    version=3,
)

answer = client.knowledge.answer("kb_customer_support", "refund window?")
if answer.abstained:
    print("no evidence:", answer.reason)
```

Modes are `keyword`, `semantic`, `hybrid` (the default), `section` and
`exact`. Filters take `collections`, `document_ids`, `path_prefix`, `tags`,
`classification_max` and `metadata`; any other key raises `ValueError`
before a request leaves. Scores are rank derived, not calibrated
probabilities.

Retrieved text is untrusted data. Each result carries a `risk` assessment,
and with `include_context=True` the gateway renders the evidence inside
`<knowledge_evidence>` delimiters with a preamble saying the content is
data, not instructions. Give that rendering to a model rather than raw
`text`. Grounded answers (`answer`) need the commercial
`knowledge.advanced` capability.

## Documents

```python
kb_id = "kb_customer_support"
written = client.knowledge.documents.create(
    kb_id,
    "faq/gift-cards.md",
    "# Gift Cards\n\nGift cards never expire.\n",
    collection="faq",
    tags=["gift-cards"],
)
job = client.knowledge.jobs.wait_for(written.job.job_id, timeout=120)

uploaded = client.knowledge.documents.upload(
    kb_id, "policies/refunds.pdf", collection="faq"
)
raw = client.knowledge.documents.upload(
    kb_id, b"# Shipping\n", path="faq/shipping.md"
)

document_id = uploaded.document.document_id
revision = uploaded.document.current_revision
client.knowledge.documents.put_content(
    kb_id, raw.document.document_id, "# Shipping\n\nTwo days.\n",
    if_match=raw.document.current_revision,
)
client.knowledge.documents.upload_revision(
    kb_id, document_id, "policies/refunds-v2.pdf", if_match=revision
)
client.knowledge.documents.update(
    kb_id, document_id, if_match=uploaded.document.metadata_revision,
    path="policies/refund-policy.pdf", title=None,
)
tree = client.knowledge.documents.sections(kb_id, document_id, version=3)
links = client.knowledge.documents.backlinks(kb_id, document_id)
```

`upload` sends the raw bytes (no multipart) with the document path in the
`x-agenomic-document-path` header. A `str` or `os.PathLike` source is a file
to read, and its file name is the default document path; `bytes` need
`path=`. The content type defaults to `application/octet-stream`, so the
gateway infers the format from the path extension; pass `content_type=` to
declare it. Every `x-agenomic-*` header value (path, collection, each tag,
classification, change message) is percent-encoded UTF-8, so non-ASCII
values such as `change_message="Révision des remboursements"` are fine;
tags may not hold a comma. Uploading identical bytes to the same path
answers `created=False`. Edits to a document need `if_match`: the document
`current_revision` for content, its `metadata_revision` for metadata.
`title`, `collection` and the knowledge base `description` and `owner` are
tri-state: leave them out to keep them, pass `None` to clear them.

Documents also have `get`, `list` (with `tree=True` for folders),
`duplicate`, `delete`, `restore`, `revisions`, `text` and `section`.
Collections have `list`, `create`, `update` and `delete`.

## Versions and publication

```python
created = client.knowledge.versions.create(
    "kb_customer_support",
    change_message="Extend the refund window",
    idempotency_key="ci-build-2026-10-08",
)
client.knowledge.jobs.wait_for(created.job.job_id, timeout=600)

diff = client.knowledge.versions.diff("kb_customer_support", 4, against=3)
print(diff.summary.sections_modified, diff.affected_agents)
check = client.knowledge.versions.verify("kb_customer_support", 4)
assert check.valid

kb = client.knowledge.get("kb_customer_support")
kb.publish(4, if_match=kb.record.publication_generation, reason="Refunds")
```

A version snapshots the working set; nothing ever edits it. Publishing and
rolling back move the published pointer with a compare-and-set on
`publication_generation` (`if_match`). Publish, rollback, approve and
reject are session actions in Agenomic Cloud: an API key gets
`session_required` unless the KB setting `allow_api_key_publish` lets CI
keys publish. `jobs.wait_for` polls until the job succeeds, fails or is
cancelled and returns it; it raises `knowledge_job_timeout` after
`timeout` seconds.

## Agents and execution identity

```python
agent_id = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
knowledge = client.knowledge.agents.get(agent_id)
client.knowledge.agents.attach(
    agent_id, "kb_compliance", version="published", collections=["policies"]
)
client.knowledge.agents.put(
    agent_id,
    if_match=knowledge.config.revision,
    enabled=True,
    bindings=[{"knowledge_base": "kb_customer_support", "version": 3}],
)

found = client.knowledge.agent_search(
    agent_id,
    "Can a customer be verified with an email address only?",
    include_context=True,
    execution={"binding_id": "bnd_01j9x4w6k2m8n0p3q5r7s9t1v3"},
)
print(found.execution.manifest_digest, found.execution.resolved_via)
```

Agent-scoped retrieval reads the knowledge the agent is bound to, through
an agent knowledge manifest that the gateway freezes once per execution.
`execution` names that execution: a prompt `binding_id` (the gateway checks
that it belongs to the agent and uses the knowledge manifest of its release
genome), an `execution_id`, and optional `run_id`, `trace_id`, `session_id`
and `parent_span_id` for the run ledger and live tracing. Without
`binding_id` or `execution_id`, nothing is pinned and the current manifest
answers. A published-version change never alters the knowledge of a running
execution. Naming an agent can only narrow access: the effective access is
the intersection of the key's access and the agent's bindings.

## LangChain and LangGraph

`knowledge_tool` (in `agenomic.knowledge` and `agenomic.integrations`)
returns a LangChain tool that needs `agenomic[langchain]`. Its arguments
are `query` and an optional `top_k`; it returns the delimited evidence the
gateway rendered followed by a compact citation list.

```python
from langgraph.prebuilt import create_react_agent

from agenomic.integrations import knowledge_tool

search_kb = knowledge_tool(
    knowledge_base="kb_customer_support", version="v3", client=client
)
agent = create_react_agent(model, [search_kb])
agent.invoke({"messages": [("user", "How long is the refund window?")]})
```

The tool works in `create_react_agent` and `ToolNode` as is, sync or async.
It keeps nothing in graph state, checkpoints or stores: the only trace in
the graph is the tool message the model reads. With `include_context=False`
the tool returns the citation list only, never document text. A refusal
from the gateway (`knowledge_access_denied`, `knowledge_index_not_ready`)
becomes a tool error message the model sees; a local client raises
`cloud_required`.

With `agent_id`, the tool calls agent-scoped retrieval. Inside a graph bound
with `bind_langgraph` for the same agent, it passes the pinned prompt
binding as `execution={"binding_id": ...}`, so the knowledge follows the
release that the thread is pinned to:

```python
from agenomic.integrations import bind_langgraph, knowledge_tool

search_kb = knowledge_tool(
    knowledge_base="kb_customer_support", agent_id=agent_id, client=client
)
graph = create_react_agent(model, [search_kb], checkpointer=saver)
managed = bind_langgraph(graph, client=client, agent_id=agent_id,
                         channel="production")
managed.invoke({"messages": [("user", "refund?")]},
               {"configurable": {"thread_id": "conversation-42"}})
```

An explicit `execution={"binding_id": ...}` wins; `version` is refused with
`agent_id` because the execution pin decides the versions.
`knowledge_base=None` with `agent_id` searches every knowledge base bound to
the agent.

`KnowledgeRetriever` is a LangChain retriever over the same calls. Its
documents carry the citation in their metadata (`citation_uri`, `kb_id`,
`version`, `document_id`, `document_revision`, `section_id`, `chunk_id`,
`content_digest`, `risk_level`, `risk_flags`, `retrieval_event_id`).

```python
from agenomic.knowledge import KnowledgeRetriever

retriever = KnowledgeRetriever(
    client=client, knowledge_base="kb_customer_support", version=3, top_k=4
)
documents = retriever.invoke("refund window")
```

## Tracking and canonical traces

The gateway records every retrieval as a retrieval event; read one with
`client.knowledge.retrievals.get(event_id)` and rebuild its evidence with
`client.knowledge.retrievals.snapshot(event_id)`. Agent-scoped retrievals
that carry a `run_id` or a tracking `session_id` are also written to the
run ledger and to live tracing by the gateway.

`knowledge.retrieve` is a tracking event type. In a canonical run, record
references and digests only, never the retrieved text:

```python
from agenomic.canonical import start_run

run = start_run("agent://acme/support")
found = client.knowledge.agent_search(agent_id, "refund window")
run.log_knowledge(**found.knowledge_refs(), query="refund window")
trace = run.complete_run(output={"answer": "14 days"})
```

`knowledge_refs()` keeps the retrieval references (event id, KB, version,
manifest and index config digests, mode) and the citations; the query is
stored as its sha256 digest. An agent knowledge manifest digest becomes the
run's `knowledge_version` component, as does
`start_run(..., knowledge_manifest_digest=...)`.

## Errors

Server refusals are `agenomic.exceptions.ApiError` with the gateway code,
status and details, for example `knowledge_base_not_found`,
`knowledge_access_denied` (`details.reason`), `knowledge_index_not_ready`,
`knowledge_version_not_publishable`, `knowledge_publication_conflict`,
`knowledge_document_conflict` and `session_required`. The SDK raises two
codes of its own with status 0: `knowledge_section_not_found` when a heading
matched no section of the requested document, and `knowledge_job_timeout`.
Invalid arguments raise `ValueError` before any request.

## Not in this release

Sources, connectors and sync, provider connections, reindexing, export and
import, analytics, comparative replays and evaluations, publication event
listings and version retraction are managed in the web app or over HTTP.
