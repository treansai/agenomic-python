# Integrations

`agenomic-python` ships first-party integrations for OpenAI, Anthropic,
LangGraph, LangChain, and MCP, all **optional**. The `instrument_*` ones are
lazy: you can `import agenomic.integrations.openai` without `openai`
installed, and the import error only raises when you call
`instrument_openai()`. The LangChain handler and the LangGraph managed
prompts adapter are the exceptions: they use `langchain_core` and `langgraph`
types, so importing `agenomic.integrations.langchain`, or importing
`bind_langgraph` and its helpers, without the extra raises immediately.
Importing the `agenomic.integrations` package itself is always safe.

## OpenAI

```bash
pip install "agenomic[openai]"
```

```python
from openai import OpenAI
from agenomic.integrations.openai import instrument_openai

client = instrument_openai(OpenAI())
```

Wraps `chat.completions.create` (sync + async) so each call records a
`ModelCall` on the active `TraceRecorder`.

## Anthropic

```bash
pip install "agenomic[anthropic]"
```

```python
from anthropic import Anthropic
from agenomic.integrations.anthropic import instrument_anthropic

client = instrument_anthropic(Anthropic())
```

## Hugging Face

Built on `httpx`: no extra is required to instrument. The bundled
`HuggingFaceClient` resolves Hub metadata and runs inference; wrapping it
records a `ModelCall(provider="huggingface", model=...)` on success and error,
without ever logging the token.

```python
from agenomic.providers.huggingface import HuggingFaceClient, HuggingFaceConfig
from agenomic.integrations.huggingface import instrument_huggingface

client = instrument_huggingface(HuggingFaceClient(HuggingFaceConfig.from_env()))
client.generate_text("gpt2", "hello")
```

For inference functions you call yourself, use `trace_huggingface_call`. See
[the Hugging Face provider guide](providers/huggingface.md) for setup, env
vars, and security details.

## LangGraph

```bash
pip install "agenomic[langgraph]"
```

Two LangGraph integrations exist:

- [Managed prompts](#langgraph-managed-prompts) (`bind_langgraph`) pins
  every thread of a graph to one prompt release.
- `instrument_langgraph` and `instrument_langgraph_canonical` wrap the
  callables of a `graph.nodes` mapping into the trace channel.

**`instrument_langgraph` records nothing on a real LangGraph graph.** It
wraps a node only when the node, or its `runnable`, is a plain callable. A
`StateGraph` holds `StateNodeSpec` entries whose `runnable` is a
`RunnableCallable`, and a compiled graph holds `PregelNode` objects; neither
is callable, so no node is wrapped and no `ToolCall` is recorded. The graph
still runs unchanged. `instrument_langgraph_canonical` behaves the same and
emits no canonical event. Both work only on duck-typed graphs whose `nodes`
mapping holds functions, such as `examples/05_langgraph_traced.py`. To
observe a real graph, use `TrackingCallbackHandler` (next section).

## LangChain (live tracking)

To feed the **live tracking** channel from a LangChain or LangGraph app, pass
`TrackingCallbackHandler` in the runnable config: LangChain propagates it to
every child run, so subgraphs, nodes, chat models, tools and retrievers are
all observed without touching the graph.

```bash
pip install "agenomic[langgraph]"
```

```python
from agenomic import Client
from agenomic.integrations.langchain import TrackingCallbackHandler

session = Client().tracking.start(agent="agent://acme/support")

await graph.ainvoke(state, config={"callbacks": [TrackingCallbackHandler(session)]})

session.stop()   # drains the emitter, then closes the session
```

Build one handler per request and share the session. The mapping:

| LangChain run             | tracking events                           |
| ------------------------- | ----------------------------------------- |
| root chain                | `turn.started` / `.completed` / `.failed` |
| node (`graph:step:N` tag) | `agent.step.*`                            |
| chat model or LLM         | `model.call.*` with `usage`               |
| tool                      | `tool.call.*`                             |
| retriever                 | `retrieval.*`                             |

Only the root chain and `graph:step:`-tagged nodes produce chain spans. Every
other chain run is silent: LangChain's own plumbing (`RunnableSequence`,
`ChannelWrite`, `seq:step:N`), but equally any sub-chain of your own that
LangGraph did not tag. Their children are re-parented onto the nearest emitted
span, so the hierarchy matches the graph rather than the runnable tree.

Events are queued to one background worker per session, so a slow or failing
gateway never blocks the run and never raises into your code. That is also why
delivery is not guaranteed by default:

```python
from agenomic.integrations.langchain import dropped_events, flush

if not flush(session):                     # checkpoint mid-session
    print("dropped:", dropped_events(session))
```

`flush()` returns `False` when it timed out **or** when an event was dropped
while draining. Read `dropped_events()` before `stop()`: teardown forgets the
emitter. `session.stop()` drains and joins the worker on its own, so
`shutdown()` is only needed if you never stop the session.

Raw prompts, arguments and completions never leave the process; the handler
sends `input_hash` / `output_hash` only, and an error sends the exception class
name without its message. `capture_turn_title=True` is the one opt-in
exception: it takes the **last** message of the root run's `messages` state
whatever its role, collapses its whitespace and sends the first 120
characters.

## LangGraph managed prompts

`bind_langgraph` wraps a compiled graph in a `ManagedGraph` proxy that pins
every thread to one prompt release of an agent. Nodes read their prompts from
the run config, so a promotion changes new threads only: a thread that has
started, is paused on an interrupt, or resumes after a restart keeps the
prompts it started with. [Managed prompts](prompts.md) explains prompts,
releases, channels and bundles. The supported LangGraph versions are in the
[LangGraph version matrix](langgraph-matrix.md).

```python
import itertools

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from agenomic.integrations import bind_langgraph, prompts_for

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
model = GenericFakeChatModel(
    messages=itertools.cycle([AIMessage("Your order left today.")])
)


class State(TypedDict, total=False):
    question: str
    answer: str


def answer(state: State, config: RunnableConfig) -> State:
    prompts = prompts_for(config)
    system = prompts.render_text("support.system", {"locale": "en"})
    reply = model.invoke(
        [SystemMessage(system), HumanMessage(state["question"])],
        prompts.config_for("support.system"),
    )
    return {"answer": str(reply.content)}


builder = StateGraph(State)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
graph = builder.compile(checkpointer=InMemorySaver())
```

Use your own chat model instead of the fake one. `Client()` without
`base_url` runs the registry in process, which is how the examples run
offline:

```python
from agenomic import Client

client = Client()
client.prompts.create("prm_support", name="Support", kind="text")
client.prompts.publish(
    "prm_support",
    {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": "Answer delivery questions in the {locale} locale.",
        "variables": {"locale": {"type": "string", "required": True}},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    },
    parent_version=None,
    change_message="First version",
)
engine = client.prompts.local
release_id = engine.create_release(
    AGENT_ID, {"support.system": "prm_support:1"}
)
engine.move_channel(AGENT_ID, "production", release_id, expected_generation=0)

managed = bind_langgraph(
    graph, client=client, agent_id=AGENT_ID, channel="production"
)
config: RunnableConfig = {"configurable": {"thread_id": "ticket-1001"}}
result = managed.invoke({"question": "Where is my order?"}, config)
pin = managed.get_state(config).metadata
print(pin["agenomic_release_id"], pin["agenomic_prompt_manifest_digest"])
```

With Agenomic Cloud, pass a cloud client that holds a `read` API key:

```python
client = Client(api_key="agm_...", base_url="https://agenomic.example")
managed = bind_langgraph(
    graph, client=client, agent_id=AGENT_ID, channel="production"
)
```

- `agent_id` is the agent's uuid, lowercase. Name exactly one of `channel`
  and `release_id`. There is no default, so an unnamed target never means
  the latest release.
- Every run entry point admits the call before LangGraph sees it: `invoke`,
  `stream`, `stream_events`, `update_state`, `bulk_update_state`, their
  `a*` twins, and the `Runnable` helpers that go through them (`batch`,
  `abatch`, `with_retry`, ...). Admission creates or reads the thread's
  execution binding, checks that it was made for this target, verifies the
  prompt closure and puts the pin in the call config. State reads and graph
  drawing (`get_state`, `get_state_history`, `get_graph`, ...) pass through.
- The first call of a thread pins it. Later turns, resumes after an
  interrupt, time travel and `update_state` get the same binding even after
  the channel has moved, while new threads get the channel's current
  release. The pin is resolved once per call: no node, token or subgraph
  fetches anything.
- Online, each invocation makes one idempotent binding request, so revoked
  access takes effect on long-lived threads; `revalidate="never"` reuses the
  cached binding instead. During a registry outage a thread whose binding is
  cached goes on with it and a new thread fails with `registry_unavailable`
  ([Registry outages](prompts.md#registry-outages)).
- The thread id is hashed with the workspace before it leaves the process.
- A key that has every scope, `write` or `admin` raises
  `privileged_credential` unless you pass `allow_privileged_credential=True`.
  Run agents with a `read` key. When this is checked is described under
  [Credential check and registry outages](#credential-check-and-registry-outages).
- Checkpoint and callback metadata carry the pin as plain values
  (`agenomic_binding_id`, `agenomic_prompt_manifest_digest`,
  `agenomic_agent_id`, `agenomic_release_id`, `agenomic_genome_version`),
  never prompt text. Neither a caller config nor the graph's own config may
  set the keys the adapter injects (`agenomic_reserved_key`).
- A `langgraph` or `langchain-core` version outside the tested points emits
  one `AgenomicUntestedVersionWarning` per process. Execution goes on.

### Node accessor

`prompts_for(config)` reads the pinned prompts from the config that LangGraph
passes to a node with a `config: RunnableConfig` parameter. It does no I/O.

- `version(slot)` returns the pinned `ManagedPromptVersion`, `render_text`
  renders a `text` slot, `render_messages` renders a `chat` slot as
  LangChain messages, and `compose(slot, variables, history=...)` appends
  history to a chat slot that has no placeholder. A slot that the release
  does not pin raises `slot_not_in_manifest`.
- Pass `prompts.config_for(slot, ...)` to the model call. It adds
  `agenomic_prompt_slots`, `agenomic_prompt_refs` and
  `agenomic_prompt_content_digests`, plus `agenomic_rendered_hash` when
  exactly one of the slots was rendered through this accessor, to the run
  metadata. `TrackingCallbackHandler` and the `messages` stream mode read
  them there. On Python 3.10 async it also keeps the model run attached to
  the node.
- Return only the model reply from the node, as above, so the rendered
  system prompt never enters the checkpointed message history.
- A graph that runs without the proxy raises `binding_missing`. Nothing
  falls back to an inline string or to the latest version.

### Thread and execution scope

- `pin_scope="thread"`, the default, pins a thread and needs
  `configurable["thread_id"]` (`thread_id_required`).
- `pin_scope="execution"` pins one logical execution. A new run needs
  `configurable["agenomic_execution_key"]` (`execution_key_required`): send
  the same key again when you retry the same request, and use an
  unguessable uuid4 when the key comes from request data. A resume, a
  `None` input or a state update reads the binding from the thread's
  checkpoint (`execution_binding_unrecoverable` when there is none).
  `managed.with_retry(...)` sets one key per input, so every attempt shares
  one binding.

### Credential check and registry outages

Online, the proxy reads the credential's scopes from `GET /v1/whoami`,
which each client asks once and then remembers.

- At bind time, when the registry answers, a privileged key raises
  `privileged_credential` at once.
- When the registry is unavailable at bind time, `bind_langgraph` returns
  the proxy only if the client has a `workspace_id`, since cache entries are
  keyed by workspace; otherwise it raises `registry_unavailable`. The
  credential check is then pending.
- While the check is pending, every admission asks `GET /v1/whoami` first,
  with the usual retries: before a binding request, before a binding read,
  and before a cached binding is used with `revalidate="never"`. When the
  registry answers, a privileged key raises `privileged_credential` before
  any binding request is sent. A 401, 403 or 404 is raised, and when the
  call names a thread key (a thread binding, a new execution, or
  `revalidate="never"`) it also evicts that thread's cached binding.
- While the registry is unavailable, no binding request gets through. A
  proxy whose check is pending sends none, because `GET /v1/whoami` fails
  first; a proxy whose check has passed sends the request with the usual
  retries, and it fails. Either way, a thread whose binding is cached with
  the same scope and target resumes on it after its cached closure is
  verified again; one WARNING is logged on `agenomic.prompts` and
  `registry_outage_cached_binding_total` increments. With
  `revalidate="never"`, a cached binding is used without a binding request,
  as at any other time. Any other thread raises `registry_unavailable`
  before a node runs. No channel is resolved and nothing falls back to the
  latest release.
- A restarted process finds a thread's binding only in a disk cache: set
  `AGENOMIC_PROMPT_CACHE_DIR` (or pass `prompt_cache=PromptCache(directory)`
  to the `Client`) together with `workspace_id`.
- An execution-scope resume reads its binding by id, which the cache does
  not index, so it raises `registry_unavailable` during an outage.

### Subagents

One binding pins the root agent and the child agents its release pins. Tell
the adapter which agent a node belongs to:

- A compiled child graph added as a node: `children={"research": CHILD_ID}`.
  A key `K` matches the nodes inside `K` (node paths that start with `K|`),
  never the node `K` itself, and the longest key wins.
- A child graph invoked in a wrapper node or a tool: pass
  `scope_config(config, CHILD_ID)` as its config.
- A child with its own thread and checkpointer:
  `scope_config(config, CHILD_ID, thread_id=child_thread_id)`. Its prompts
  still come from the parent's binding.
- Any node or `Send` worker can read a child's prompts with
  `prompts_for(config, agent_id=CHILD_ID)`.

`examples/13_langgraph_subagent_prompts.py` shows the first two patterns.

- Always pass `config` to a nested graph. Without it, the nested graph
  inherits the root agent's scope on sync paths and on Python 3.11 and
  later, which is the wrong agent, and gets nothing on Python 3.10 async,
  where `prompts_for` fails closed with `binding_missing`.
- Bind only the root graph; `bind_langgraph` refuses a graph that is already
  bound. A bound graph used as a node of another bound run switches to its
  own agent without a new binding, but `get_state` cannot read nested
  namespaces through it, so prefer the raw compiled child with `children=`.
  A bound child invoked inside a managed run with a fresh config raises
  `nested_bind_unsupported` (detected on sync paths and on Python 3.11 and
  later).
- A mapped child that the binding does not pin raises
  `child_agent_not_pinned` when a node reads it, so threads pinned to an
  older release keep running as long as they never reach the new subagent.

### Prompts fixed at construction

When a constructor takes a prompt Runnable, `managed_prompt(slot)` reads the
pinned prompt at each model call:

```python
from langgraph.prebuilt import create_react_agent

from agenomic.integrations import AgentFactory, managed_prompt

agent = create_react_agent(
    model,
    [],
    prompt=managed_prompt("support.system", variables={"locale": "en"}),
    checkpointer=InMemorySaver(),
)
react = bind_langgraph(
    agent, client=client, agent_id=AGENT_ID, channel="production"
)
react.invoke(
    {"messages": [("user", "Where is my order?")]},
    {"configurable": {"thread_id": "ticket-1002"}},
)
```

When the prompt is a string captured while the graph is built, use an
`AgentFactory`. It builds one graph per pinned release:

```python
saver = InMemorySaver()


def build(prompts):
    text = prompts.version("support.system").render_text({"locale": "en"})
    return create_react_agent(model, [], prompt=text, checkpointer=saver)


factory = bind_langgraph(
    AgentFactory(build), client=client, agent_id=AGENT_ID, channel="production"
)
factory.invoke(
    {"messages": [("user", "Where is my order?")]},
    {"configurable": {"thread_id": "ticket-1003"}},
)
```

- `managed_prompt(slot, variables=..., history_key="messages")` renders a
  `text` slot as one system message before the history. A `chat` slot gets
  the history in its placeholder named `history_key`, or after its messages
  when it has no placeholder; a placeholder with another name raises
  `history_conflict`. `variables` is a mapping or a function of the state.
- `AgentFactory(build, max_entries=32)` builds each pinned manifest (and
  genome version) at most once and keeps the most recent `max_entries`
  graphs. Every graph it builds must have the same nodes and the same
  checkpointer object as the first, so threads can move between them
  (`factory_topology_mismatch` otherwise). `build` receives the pinned set,
  which has the same `version(slot)` reader.
- With `pin_scope="execution"`, pass the saver your graphs are compiled
  with: `AgentFactory(build, checkpointer=saver)`. A resume, a `None` input
  or a state update reads the binding from the thread's checkpoint, and a
  restarted process has built no graph yet to read it with. Every graph
  the factory builds must then use that object.
- `create_react_agent` is deprecated in LangGraph 1.x and warns with
  `LangGraphDeprecatedSinceV10`, but works. `langchain.agents.create_agent`
  and its middleware are not supported.

### Streaming, batching and cancellation

- Every stream mode, list of modes, `subgraphs=True` and `version="v2"`
  passes through chunk by chunk; nothing is buffered. LangGraph puts the
  call config in `checkpoints` and `debug` chunks, so the proxy removes the
  in-memory pinned set from those chunks, and from `invoke` results asked
  with those modes, which keeps them JSON serializable.
- `astream_events(version="v2")` works. `version="v3"` passes through as
  experimental, and LangGraph 1.0.10 raises `NotImplementedError` for it.
  Sync `stream_events` passes through too; LangGraph itself refuses sync v2.
- Close a stream you leave early with `contextlib.closing`, or
  `contextlib.aclosing` for async, so the proxy's cleanup runs at once
  instead of when the generator is collected. Whether LangGraph keeps
  running the node in flight after an early close is LangGraph behavior,
  which the proxy does not change; do not rely on it.
- Task cancellation and `asyncio.wait_for` timeouts propagate. Callbacks,
  tags, run names, `recursion_limit`, `max_concurrency` and your metadata
  are kept.
- `batch` and `abatch` admit each input on its own, so one batch can mix
  threads pinned to different releases.
- The pinned set is immutable and never serialized: pickling it raises
  `prompt_set_not_serializable`, and its `repr` shows only the binding id
  and the manifest digest.

### Offline bundles

A process without network access binds to signed bundles exported from the
registry (see [Offline bundles](prompts.md#offline-bundles)):

```python
from agenomic.integrations import LocalBindingStore
from agenomic.prompts import BundleTrust

offline = bind_langgraph(
    graph,
    agent_id=AGENT_ID,
    channel="production",
    offline=True,
    workspace_id="0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f",
    bundle="bundles/production-v2.json",
    retained_bundles=[("bundles/production-v1.json", V1_BUNDLE_DIGEST)],
    trust=BundleTrust.from_pem_files("keys/orgkey_01.pem"),
    binding_store=LocalBindingStore("var/prompt-pins"),
)
```

- Offline mode takes no client. It needs `bundle` (a path, a document or a
  loaded `PromptBundle`), `workspace_id`, and `trust` or
  `expected_bundle_digest` to load a path or a document. Loading runs every
  check of `PromptBundle.load`.
- `channel` must be the channel the bundle was exported from, or
  `release_id` its release (`binding_target_mismatch` otherwise).
- A graph with a checkpointer needs `binding_store=LocalBindingStore(dir)`,
  so pins survive a restart. Each thread's pin is one file, written
  atomically. A file that does not parse or verify raises
  `binding_store_corrupt` and is never treated as absent.
- To ship a promotion, deploy the new bundle as `bundle` and keep the
  earlier ones in `retained_bundles`, as loaded `PromptBundle` objects or
  `(path, expected_bundle_digest)` pairs. New threads pin to the new bundle;
  existing threads resume on theirs.
- A signed bundle whose release is not approved raises `bundle_ungoverned`
  unless you pass `allow_ungoverned_bundle=True`.

### Errors

The adapter raises `PromptBindingError` unless noted. A wrong argument to
`bind_langgraph` raises `ValueError`.

- `agenomic_reserved_key`: a caller config or the graph's own config sets a
  key that the adapter injects.
- `thread_id_required`, `execution_key_required` and
  `execution_binding_unrecoverable`: see the pin scopes above.
- `binding_target_mismatch`: the thread's binding was made for another
  channel or release, or the offline target does not match the bundle.
- `binding_checkpoint_mismatch`: in thread scope, a binding was just
  created for a thread whose latest checkpoint carries another manifest
  digest, for example after its pin was lost.
- `binding_missing` and `binding_mismatch`: the accessor finds no pinned
  set, or one that disagrees with the pin values of the config.
- `prompt_set_unavailable`: the config carries the pin values but not the
  pinned set (a config rebuilt from checkpoint metadata or a stream chunk,
  or one that crossed a process boundary), or an `AgentFactory` proxy is
  asked for its graph before it built one, or, without `checkpointer`, to
  resume or read an execution-scope thread.
- `child_agent_not_pinned` and `slot_not_in_manifest`: the agent or the slot
  is not in the pinned release.
- `nested_bind_unsupported`, `factory_topology_mismatch`,
  `prompt_set_not_serializable`, `privileged_credential` and
  `binding_store_corrupt`: see above.
- `history_conflict` is a `PromptRenderError` reason, and
  `registry_unavailable` a `RegistryUnavailableError`.

### Limits

- LangGraph Server and LangGraph Platform deployments call the compiled
  graph directly and bypass the proxy, so their nodes fail with
  `binding_missing`. They are not supported.
- On Python 3.10, LangGraph's `interrupt()` in async nodes and implicit
  config propagation into nested graphs do not work. The adapter reads only
  the config passed to the node; use Python 3.11 or later for async
  interrupts.
- An `AgentFactory` bound with `pin_scope="execution"` and no
  `checkpointer` cannot resume, run a `None` input, update or read the
  state of a thread in a process where it has not built a graph yet. The
  call raises `prompt_set_unavailable` and never resolves the channel
  instead. Pass `AgentFactory(build, checkpointer=saver)`.
- During a registry outage, only threads whose binding is in the cache go
  on, and an execution-scope resume does not (see
  [Credential check and registry outages](#credential-check-and-registry-outages)).
- The SDK cannot promote or roll back. Moving a channel is an approved
  action in Agenomic Cloud; the local engine's `move_channel`, used by the
  examples, simulates it.

### Examples

These examples run offline with a fake chat model:

- `examples/12_managed_prompts_stategraph.py`: three slots on a
  `StateGraph`, and `to_langchain`.
- `examples/13_langgraph_subagent_prompts.py`: a compiled subgraph pinned
  with `children=` and a wrapper node that uses `scope_config`.
- `examples/14_langgraph_async_streaming.py`: `ainvoke`, token streaming
  with `astream(stream_mode="messages")` and `astream_events(version="v2")`.
- `examples/15_langgraph_interrupt_restart.py`: an interrupt on SQLite, a
  promotion shipped as a new signed bundle, and a restarted process that
  resumes the old thread on its original prompts. It needs
  `langgraph-checkpoint-sqlite` and prints a hint without it.
- `examples/16_langgraph_promotion_pinning.py`: a promotion during a
  streamed run and between two turns. The run in flight and the existing
  thread keep version 1; a new thread gets version 2.

## MCP

MCP doesn't ship a single SDK we can wrap. Call `trace_mcp_call` after
invoking your MCP client:

```python
from agenomic.integrations.mcp import trace_mcp_call

result = mcp_client.call("search", {"q": "x"})
trace_mcp_call("server-1", "search", {"q": "x"}, result)
```

## Hermes Agent

```bash
pip install "agenomic[hermes]"   # plus an editable clone of Hermes v2026.9.24
```

A Hermes plugin (`plugins.enabled: [agenomic]`), a fail closed shell hook
(`agenomic-hermes-guard`) and a supervisor (`agenomic-hermes-supervisor`)
connect a Hermes runtime to the Agenomic control plane. The plugin is
cooperative and not a security boundary. See [hermes.md](hermes.md).

## Writing your own

The pattern is small:

1. Read the active recorder via `current_recorder()`. If `None`, no-op.
2. Hash inputs with `blake3_hex(canonical_cbor(input))`.
3. Call the underlying SDK.
4. Hash outputs the same way and call
   `recorder.record_model_call(...)` or `recorder.record_tool_call(...)`.
