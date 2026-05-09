# `@trace_agent_run` reference

```python
from agenomic.trace.decorator import trace_agent_run
```

```python
trace_agent_run(
    agent_id: str,
    *,
    release: str | None = None,
    exporter: Exporter | None = None,
    redaction: RedactionEngine | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
)
```

| parameter         | default | meaning                                                     |
| ----------------- | ------- | ----------------------------------------------------------- |
| `agent_id`        | —       | logical identity, e.g. `agent://acme/claims`                |
| `release`         | `None`  | release tag added to every envelope                         |
| `exporter`        | `None`  | sink for the produced envelope                              |
| `redaction`       | `None`  | applied to input + output before export                     |
| `capture_input`   | `True`  | when `False`, input is replaced by `{"redacted": True}`     |
| `capture_output`  | `True`  | when `False`, output is replaced by `{"redacted": True}`    |

Works on both sync and async functions. When the wrapped function raises:

- the exception is re-raised
- the envelope is built with `error="<ExcType>: <message>"`
- the exporter still receives the envelope

The decorator always cleans up the contextvar in `finally`, so subsequent calls
get a fresh recorder.
