# Redaction

Redaction runs at the SDK boundary, **before any export**. Pass a
`RedactionEngine` to the decorator:

```python
from agenomic.redaction import RedactionEngine, RedactionMode, RedactionRule
from agenomic.trace.decorator import trace_agent_run

engine = RedactionEngine([
    RedactionRule(path="kwargs.password", mode=RedactionMode.MASK),
    RedactionRule(path="**.email", mode=RedactionMode.HASH),
    RedactionRule(path="payload.notes", mode=RedactionMode.TRUNCATE, truncate_length=80),
])

@trace_agent_run("agent://acme/api", redaction=engine)
def login(*, user: str, password: str) -> dict: ...
```

## Path syntax

| pattern        | meaning                                                  |
| -------------- | -------------------------------------------------------- |
| `a.b.c`        | exact path                                               |
| `a.*.c`        | single wildcard segment (any key/index)                  |
| `a.**.c`       | recursive wildcard (any depth, including zero)           |

Indexed list paths are also supported: `items.0`.

## Modes

| mode        | result                                                       |
| ----------- | ------------------------------------------------------------ |
| `REMOVE`    | delete the field                                             |
| `MASK`      | replace with `"***"`                                         |
| `HASH`      | replace with `f"hash:{blake3(json(value))[:16]}"`            |
| `TRUNCATE`  | keep first `truncate_length` chars (string values only)      |

Unknown paths are silently skipped. The engine never mutates its input.
