# AgentLock Python SDK

`agentlock-python` is a lightweight, open-source SDK for instrumenting Python AI agents and
emitting AgentLock-compatible traces. It is designed to be safe to inspect, easy to extend,
and usable without any hosted dependency in local-only mode.

## Features

- Manual trace creation with typed models and a trace recorder
- Decorator-based instrumentation for sync and async agent entrypoints
- JSONL export for offline inspection and replay
- HTTP ingestion client with optional local-only mode
- PII redaction hooks with dotted-path targeting
- Local validation of trace event payloads
- OpenAI, Anthropic, and LangGraph integration placeholders that do not require those packages

## Installation

```bash
python3 -m pip install agentlock-python
```

For local development:

```bash
python3 -m pip install -e ".[dev]"
```

## Basic Usage

```python
from agentlock import AgentLockClient, TraceRecorder

client = AgentLockClient()

trace = TraceRecorder(
    agent_id="claims-agent",
    release="dev",
    input={"claim_id": "clm_123", "amount": 249.0},
)
trace.add_model_call(
    name="gpt-4.1-mini",
    provider="openai",
    request={"prompt": "Review claim"},
    response={"decision": "approve"},
    input_tokens=24,
    output_tokens=8,
)
trace.add_tool_call(
    name="claim_history_lookup",
    input={"claim_id": "clm_123"},
    output={"previous_claims": 0},
)

envelope = trace.complete(final_output={"decision": "approve"})
client.emit_trace(envelope)
```

## Decorator Instrumentation

```python
from agentlock import AgentLockClient, Redactor, trace_agent_run

client = AgentLockClient()
redactor = Redactor(
    redact=["customer.email", "customer.phone"],
    mode="mask",
)

@trace_agent_run(
    agent_id="claims-agent",
    release="dev",
    client=client,
    redactor=redactor,
)
def handle_claim(payload, trace=None):
    if trace is not None:
        trace.add_tool_call(
            name="policy_lookup",
            input={"policy_id": payload["policy_id"]},
            output={"active": True},
        )
    return {"decision": "approve", "claim_id": payload["claim_id"]}
```

The decorator:

- Generates `trace_id` and `run_id`
- Captures input and final output hashes
- Records success or failure
- Emits the resulting trace through `AgentLockClient`
- Applies configured redaction before raw payloads are persisted

## JSONL Export

```python
from agentlock import AgentLockClient

client = AgentLockClient()
client.export_jsonl("traces.jsonl", client.local_traces)
```

Each line is a standalone JSON document representing one `TraceEnvelope`.

## HTTP Ingestion

`AgentLockClient` works in local-only mode when `endpoint` is omitted.

```python
from agentlock import AgentLockClient

client = AgentLockClient(
    api_key="agentlock_test_key",
    endpoint="https://ingest.example.com/v1/traces",
)
```

When an endpoint is configured, traces are sent with `POST` using `httpx`. The API key is
attached as a bearer token.

## Redaction

Redaction is path-based and supports `remove`, `mask`, and `hash` modes.

```python
from agentlock import Redactor

redactor = Redactor(
    redact=[
        "customer.email",
        "customer.phone",
        "payment.card.number",
    ],
    mode="hash",
)

safe_payload = redactor.redact(
    {
        "customer": {
            "email": "casey@example.com",
            "phone": "+1-555-0100",
        }
    }
)
```

You can also provide `RedactionRule` objects for per-field modes.

## LangGraph Integration

Deep LangGraph instrumentation is planned. The current `instrument_langgraph(...)` helper is a
minimal wrapper around graph-like objects that expose `invoke(...)` or `ainvoke(...)`, and it
does not require the `langgraph` package to be installed just to import the SDK.

## Schema Compatibility

The SDK emits `TraceEnvelope` payloads containing:

- `trace_id`
- `run_id`
- `agent_id`
- `release`
- `timestamp`
- `input`
- `model_calls`
- `tool_calls`
- `final_output`
- `labels`
- `metadata`

Additional typed event models are included for richer local validation and downstream
compatibility:

- `AgentRun`
- `TraceEvent`
- `ModelCall`
- `ToolCall`
- `MemoryAccess`
- `PolicyCheck`
- `HumanFeedback`
- `RunCompleted`
- `TraceEnvelope`
