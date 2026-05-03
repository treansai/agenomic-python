# Integrations

`agentlock-python` ships first-party integrations for OpenAI, Anthropic,
LangGraph, and MCP — all **optional and lazy-imported**. You can
`import agentlock.integrations.openai` without `openai` installed; the
import error only raises when you call `instrument_openai()`.

## OpenAI

```bash
pip install "agentlock[openai]"
```

```python
from openai import OpenAI
from agentlock.integrations.openai import instrument_openai

client = instrument_openai(OpenAI())
```

Wraps `chat.completions.create` (sync + async) so each call records a
`ModelCall` on the active `TraceRecorder`.

## Anthropic

```bash
pip install "agentlock[anthropic]"
```

```python
from anthropic import Anthropic
from agentlock.integrations.anthropic import instrument_anthropic

client = instrument_anthropic(Anthropic())
```

## LangGraph

```bash
pip install "agentlock[langgraph]"
```

```python
from langgraph.graph import StateGraph
from agentlock.integrations.langgraph import instrument_langgraph

graph = instrument_langgraph(StateGraph(...))
```

Each node execution records a `ToolCall`.

## MCP

MCP doesn't ship a single SDK we can wrap. Call `trace_mcp_call` after
invoking your MCP client:

```python
from agentlock.integrations.mcp import trace_mcp_call

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
