# Workflow & system manifests (spec v0.2, RFC 0009)

`agenomic.types` ships pydantic models for the v0.2 orchestration
manifests: `WorkflowSpec` (`workflow.yaml`) and `SystemSpec`
(`system.yaml`). Parse the YAML/JSON yourself and validate the dict:

```python
import yaml
from agenomic.types import WorkflowSpec, SystemSpec

spec = WorkflowSpec.model_validate(yaml.safe_load(open("workflow.yaml")))
```

All models accept and preserve unknown fields (`extra="allow"`), so
forward-compatible manifests round-trip.

## `WorkflowSpec`

| field              | type                  | notes                                  |
| ------------------ | --------------------- | -------------------------------------- |
| `spec_version`     | `str`                 | defaults to `agenomic/v0.2`            |
| `workflow`         | `WorkflowIdentity`    | `id` must match `workflow://org/name`  |
| `engine`           | `EngineHint?`         | non-normative runtime hint             |
| `triggers`         | `list[TriggerSpec]`   | `api`/`event`/`schedule`/`signal`/`manual` |
| `inputs`/`outputs` | `list[IoField]`       | declared I/O fields                    |
| `state`            | `StateRef?`           | shared-state schema reference          |
| `steps`            | `list[WorkflowStep]`  | at least one; validated as a DAG       |
| `signals`          | `list[SignalSpec]`    | resumable external signals             |
| `escalation_rules` | `list[EscalationRule]`| `condition` → `route`                  |
| `labels`           | `dict[str, str]`      | freeform                               |

### Steps

`WorkflowStep.type` is one of `agent`, `tool`, `human`, `wait`,
`workflow`, `loop` — and each type requires its matching field
(validated at parse time):

| type       | required field | shape                                             |
| ---------- | -------------- | ------------------------------------------------- |
| `agent`    | `agent`        | agent id string                                   |
| `tool`     | `tool`         | `ToolRef(name, protocol?, server?, version?)`     |
| `human`    | `gate`         | `HumanGate(role, action, sla?, on_timeout?, …)`   |
| `wait`     | `wait_for`     | `WaitFor(signals, mode?, timeout?, on_timeout?)`  |
| `workflow` | `uses`         | referenced workflow id                            |
| `loop`     | `body` + `until` | nested steps + exit condition                   |

Execution order is the DAG induced by `depends_on`; a false `when` guard
skips the step. Steps also carry `inputs`, `outputs`, `retry`
(`RetrySpec(max_attempts>=1, backoff?, initial_interval?)`), `timeout`,
and `on_error` (`fail`/`continue`/`escalate`).

Durations (`timeout`, `sla`, `initial_interval`) are validated against
`^[0-9]+(ms|s|m|h|d)$` — e.g. `30s`, `15m`, `2h`, `90d`.

The model validator rejects duplicate step ids and `depends_on`
references to ids that don't exist at the same nesting level.

## `SystemSpec`

Multi-agent system manifest:

| field                      | type                     | notes                                |
| -------------------------- | ------------------------ | ------------------------------------ |
| `system`                   | `SystemIdentity`         | `id` must match `system://org/name`  |
| `agents`                   | `list[SystemMember]`     | at least one; roles must be unique   |
| `orchestration`            | `OrchestrationSpec`      | see below                            |
| `shared_state`             | `StateRef?`              | schema only, never data              |
| `signals`                  | `list[SignalSpec]`       |                                      |
| `workflows`                | `list[WorkflowRef]`      | workflows owned by the system        |
| `communication_guardrails` | `list[str]`              |                                      |
| `escalation_rules`         | `list[EscalationRule]`   |                                      |
| `forbidden_autonomy`       | `list[str]`              | system-wide forbidden actions        |

`OrchestrationSpec.style` is `pipeline`, `graph`, `supervisor`, `swarm`,
or `custom`, with `entrypoint`/`supervisor` role references and directed
`edges` (`from` → `to`, optional `when` guard). `END` (exported as
`END_VERTEX`) is the reserved terminal vertex. Validation requires
`entrypoint`, `supervisor`, and every edge endpoint (except `END`) to
reference declared member roles.

### Autonomy shadowing

Members declare an `AutonomySpec` (`allowed_actions` /
`forbidden_actions`). Actions a member allows that the system forbids in
`forbidden_autonomy` are dead declarations worth a warning:

```python
spec.shadowed_allowed_actions()   # {"role": ["action", ...]}
```

## Identifier validation

```python
from agenomic.types import validate_agent_id

validate_agent_id("agent://acme/claims")   # ok
validate_agent_id("not-an-id")             # ValueError
```

Agent ids match `agent://[a-z0-9-]+/[a-z0-9-]+`; workflow and system ids
follow the same shape with their own schemes.
