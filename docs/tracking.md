# Online tracking

Online tracking instruments a **production** agent run as a stream of
spec-shaped runtime events (`agenomic/v0.3` `tracking-event`). It is the
"Monitor" input surface: `agenomic-cli` and Agenomic Cloud run detection
(loops, drift, harness violations) over these events.

Local-first, like everything else in the SDK: with no `base_url` the
session buffers events in memory; with a `base_url` it streams each event
to the cloud. There is no silent fallback between the two modes.

## Starting a session

```python
from agenomic import Client

client = Client()  # local mode; Client(api_key=..., base_url=...) for cloud

session = client.tracking.start(
    agent="agent://acme/claims-agent",
    release_id="release_123",       # optional
    bundle_id=None,                 # optional
    genome_hash=None,               # optional
    environment="production",       # default
    tracking_config=None,           # optional dict
)
```

In cloud mode, `start` POSTs `/v1/tracking/sessions` and raises
`CloudError` if the response has no `session_id`. In local mode the
session id is a fresh ULID.

## Emitting events

The generic emitter is `session.event(event_type, **fields)`. The event
type must be one of the spec's `TRACKING_EVENT_TYPES`:

```
agent.started        agent.completed       agent.failed
turn.started         turn.completed        turn.failed
agent.step.started   agent.step.completed  agent.step.failed
model.call.started   model.call.completed  model.call.failed
tool.call.started    tool.call.completed   tool.call.failed
retrieval.started    retrieval.completed   retrieval.failed
memory.read          memory.write          policy.evaluated
intent.detected      loop.detected         drift.detected
harness.violation    alert.created
```

A `turn` is one user-facing exchange and groups the spans beneath it;
`retrieval.*` covers RAG lookups. Both come from framework instrumentation
rather than the typed helpers below.

Unknown types raise `ValueError`; emitting on a stopped session raises
`RuntimeError`. Each event is stamped with `spec_version`, a ULID
`event_id`, `session_id`, `timestamp`, `sequence_number`, `type`, and
`agent_id`.

Typed helpers cover the common cases:

```python
with session.step("classify_claim"):          # agent.step.started / .completed
    session.model_call(provider="openai", model="gpt-4o",
                       input_hash="blake3:" + "0" * 64)
    session.tool_call(tool="claims_db.lookup", protocol="mcp",
                      input_hash="blake3:" + "1" * 64,
                      output_hash="blake3:" + "2" * 64)
    session.intent("verify_claim_validity")   # intent.detected
    session.memory_write(schema_version="1.0.0")
session.stop()                                # idempotent
```

`session.step(name)` is a context manager: it emits `agent.step.started`
on entry, `agent.step.completed` on success, and `agent.failed` (then
re-raises) when the body raises. The session itself is also a context
manager — leaving the `with` block calls `stop()`.

## From a framework

Emitting by hand is only worth it for code you own. For a LangChain or
LangGraph app, `TrackingCallbackHandler` mirrors every run into the session
on its own — turns, nodes, model calls, tools and retrievers, with
`span_id`/`parent_span_id` and token usage. See
[integrations.md](integrations.md#langchain-live-tracking).

Producers that buffer, like that handler, register teardown with
`session.on_stop(callback)`; `stop()` runs those callbacks while the session
still accepts events, so nothing queued is lost.

## Reading back

| accessor            | mode  | result                                        |
| ------------------- | ----- | --------------------------------------------- |
| `session.events`    | local | copy of the buffered event dicts              |
| `session.to_jsonl()`| local | buffered events serialized as JSONL           |
| `session.report()`  | cloud | `GET /v1/tracking/sessions/{id}/report`       |
| `session.cloud`     | both  | `True` when streaming to cloud                |

`report()` raises `RuntimeError` in local mode — feed the JSONL to
`agenomic-cli` instead.

## Hashes, not payloads

Tracking events carry **hashes** (`input_hash`, `output_hash`,
`genome_hash`) rather than raw payloads. Compute them with the canonical
helpers so they match the rest of the toolchain:

```python
from agenomic.canonical import content_hash

session.model_call(provider="openai", model="gpt-4o",
                   input_hash=content_hash({"prompt": "..."}))
```

See `examples/08_online_tracking.py` for the full runnable walkthrough.
