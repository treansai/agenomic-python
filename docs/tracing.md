# Tracing

A `TraceEnvelope` represents one agent run. Fields:

| field             | type                | meaning                                |
| ----------------- | ------------------- | -------------------------------------- |
| `schema_version`  | `str`               | always `agenomic-trace/v0.1`          |
| `trace_id`        | `str`               | parent trace correlation ID            |
| `run_id`          | `str`               | unique ID of this run                  |
| `agent_id`        | `agent://org/name`  | logical agent identity                 |
| `release`         | `str?`              | release tag (e.g. `v1.2.3`)            |
| `timestamp`       | UTC datetime        | when the run started                   |
| `input`           | `TraceInput`        | redacted input arguments               |
| `model_calls`     | `list[ModelCall]`   | LLM invocations                        |
| `tool_calls`      | `list[ToolCall]`    | tool invocations                       |
| `final_output`    | `TraceOutput`       | redacted final output                  |
| `labels`          | `dict[str, str]`    | freeform labels                        |
| `metadata`        | `dict[str, Any]`    | freeform metadata                      |
| `error`           | `str?`              | exception summary if the run raised    |
| `duration_ms`     | `int?`              | wall-clock duration                    |

## Recorder

While a run is in progress, `TraceRecorder` accumulates `model_calls`,
`tool_calls`, labels, and metadata. The current recorder is exposed via a
`contextvar` and accessible from anywhere inside the wrapped function:

```python
from agenomic.trace.context import current_recorder

rec = current_recorder()
if rec is not None:
    rec.add_label("env", "prod")
```

The integrations record on the current recorder automatically.
