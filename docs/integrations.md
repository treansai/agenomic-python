# Integrations

`agenomic-python` ships first-party integrations for OpenAI, Anthropic,
LangGraph, LangChain, and MCP — all **optional**. The `instrument_*` ones are
lazy: you can `import agenomic.integrations.openai` without `openai`
installed, and the import error only raises when you call
`instrument_openai()`. The LangChain handler is the exception — it subclasses
a `langchain_core` type, so importing `agenomic.integrations.langchain`
without the extra raises immediately. Importing the `agenomic.integrations`
package itself is always safe.

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

Built on `httpx` — no extra is required to instrument. The bundled
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

```python
from langgraph.graph import StateGraph
from agenomic.integrations.langgraph import instrument_langgraph

graph = instrument_langgraph(StateGraph(...))
```

Each node execution records a `ToolCall`.

## LangChain (live tracking)

`instrument_langgraph` feeds the **trace** channel. To feed the **live
tracking** channel instead, pass `TrackingCallbackHandler` in the runnable
config: LangChain propagates it to every child run, so subgraphs, nodes, chat
models, tools and retrievers are all observed without touching the graph.

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

| LangChain run                      | tracking events                        |
| ---------------------------------- | -------------------------------------- |
| root chain                         | `turn.started` / `.completed` / `.failed` |
| node (`graph:step:N` tag)          | `agent.step.*`                         |
| chat model or LLM                  | `model.call.*` with `usage`            |
| tool                               | `tool.call.*`                          |
| retriever                          | `retrieval.*`                          |

Only the root chain and `graph:step:`-tagged nodes produce chain spans. Every
other chain run is silent — LangChain's own plumbing (`RunnableSequence`,
`ChannelWrite`, `seq:step:N`) but equally any sub-chain of your own that
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

## MCP

MCP doesn't ship a single SDK we can wrap. Call `trace_mcp_call` after
invoking your MCP client:

```python
from agenomic.integrations.mcp import trace_mcp_call

result = mcp_client.call("search", {"q": "x"})
trace_mcp_call("server-1", "search", {"q": "x"}, result)
```

## Writing your own

The pattern is small:

1. Read the active recorder via `current_recorder()`. If `None`, no-op.
2. Hash inputs with `blake3_hex(canonical_cbor(input))`.
3. Call the underlying SDK.
4. Hash outputs the same way and call
   `recorder.record_model_call(...)` or `recorder.record_tool_call(...)`.
