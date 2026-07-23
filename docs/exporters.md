# Exporters

An exporter is a sink for `TraceEnvelope` instances. The
`@trace_agent_run` decorator drives sync and async exporters alike, and
exporter failures are logged, never raised into your traced function.

All exporters are context managers — `close()` runs on exit:

```python
from agenomic.exporters import (
    Exporter, JsonlExporter, AtepLocalExporter, HttpExporter, MultiExporter,
)
```

## `JsonlExporter`

Append-only JSONL file, one envelope per line. Parent directories are
created automatically.

```python
JsonlExporter(path, *, flush_each=True)
```

```python
with JsonlExporter("traces.jsonl") as exporter:
    ...
```

Exporting after `close()` raises `RuntimeError`.

## `AtepLocalExporter`

Converts each (already-redacted) envelope into one signed ATEP
`interaction.run_completed` event and appends it to a local `AtepStore`.
v0.1 emits a linear causal chain — each event's parent is the previous
interaction event's causal hash. Resuming an existing store picks up the
sequence, clock, and last hash from disk.

```python
from agenomic.atep import AtepStore
from agenomic.crypto import SigningKey

store = AtepStore.open_or_init(Path("store"), "agent://acme/demo")
exporter = AtepLocalExporter(store, SigningKey.generate(), node_id=0)
```

See [ATEP](atep.md) for the store and segment format.

## `HttpExporter`

Asynchronous, batched upload via `AgenomicClient.upload_traces`. Buffers
envelopes until `batch_size` is reached (default 100) or
`batch_interval_ms` elapses (default 5000), whichever comes first.

```python
from agenomic.client import AgenomicClient

client = AgenomicClient("https://cloud.example.com", api_key="sk-...")
exporter = HttpExporter(client, batch_size=100, batch_interval_ms=5000)
...
await exporter.aclose()   # flush + drain; use close() only outside a loop
```

Upload errors are logged and swallowed — telemetry never takes your agent
down. `export()` on a closed exporter raises `RuntimeError`.

## `MultiExporter`

Fan out to several sinks. An error in one exporter never prevents the
others from receiving the envelope.

```python
exporter = MultiExporter(
    JsonlExporter("traces.jsonl"),
    AtepLocalExporter(store, signing_key),
)
```

`close()` closes every child, logging per-child failures.

## Writing your own

Subclass `Exporter` and implement `export(envelope)` (sync or async);
override `close()` if you hold resources:

```python
from agenomic.exporters.base import Exporter
from agenomic.types import TraceEnvelope

class StdoutExporter(Exporter):
    def export(self, envelope: TraceEnvelope) -> None:
        print(envelope.model_dump_json(exclude_none=True))
```
