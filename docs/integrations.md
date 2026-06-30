# Integrations

`agenomic-python` ships first-party integrations for OpenAI, Anthropic,
LangGraph, and MCP — all **optional and lazy-imported**. You can
`import agenomic.integrations.openai` without `openai` installed; the
import error only raises when you call `instrument_openai()`.

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
