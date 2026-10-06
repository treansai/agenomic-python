# Hermes Agent

`agenomic.integrations.hermes` connects a [Hermes Agent](https://github.com/NousResearch/hermes-agent)
runtime to Agenomic. Hermes stays the runtime (planner, conversation loop,
memory, tools); Agenomic stays the control plane (identity, mode, decisions,
evidence). The adapter (`agenomic-hermes-adapter` 1.0.0) has three parts:

| Part | Runs | Role |
| --- | --- | --- |
| plugin `agenomic` (`agenomic.integrations.hermes.plugin:register`) | inside the Hermes process | reports the instance, sessions, model and tool calls; asks Agenomic before covered tool actions; reports results with the signed permit |
| `agenomic-hermes-guard` | spawned by Hermes for every tool call | second, independent `pre_tool_call` check: the plugin is loaded and the instance is not paused, quarantined or revoked |
| `agenomic-hermes-supervisor` | trusted process outside the agent's control | starts Hermes, stops it on quarantine/revoke, attests isolation, syncs approved skills |

**The plugin is cooperative and is not a security boundary.** It runs in the
agent's process, with the agent's permissions. Controls that must hold even
when the agent misbehaves come from the deployment: the Model Gateway as the
only route to a model, network egress restricted to Agenomic, read only
configuration, skills and plugins, the Tool Gateway for business tools, and the
supervisor. Observing an action is never presented as controlling it.

## Pinned upstream

| Field | Value |
| --- | --- |
| Repository | https://github.com/NousResearch/hermes-agent |
| Tag | `v2026.9.24` |
| Commit | `f97608f178d1ffeca59860195ab7da295f7c8e5f` |
| Version | `hermes_cli.__version__ == "0.21.5"`, `__release_date__ == "2026.9.24"` |
| License | MIT, reproduced in `src/agenomic/integrations/hermes/NOTICE` (no Hermes code is vendored) |

`COMPATIBLE_HERMES = {"0.21.5"}`. Any other version is reported in `/hello`
(`hermes_version_compatible: fail`) and, in enforce, every protected action is
blocked locally as well as by the server.

## Install

Hermes is **not** a dependency of `agenomic` and cannot be installed from a
git URL: its `setup.py` refuses to build wheels, so `pip install git+...`
fails by design. The supported install is an editable install of a clone at
the pinned tag, in the same environment as the adapter:

```bash
git clone --branch v2026.9.24 https://github.com/NousResearch/hermes-agent
python3.11 -m venv hermes-venv            # Hermes needs Python >=3.11,<3.14
hermes-venv/bin/pip install -e ./hermes-agent
hermes-venv/bin/pip install "agenomic[hermes]"   # or -e ./agenomic-python[hermes]
```

The `hermes` extra adds PyYAML (YAML adapter config files). The package
registers:

- entry point `[project.entry-points."hermes_agent.plugins"] agenomic`;
- console scripts `agenomic-hermes-guard` and `agenomic-hermes-supervisor`.

Hermes loads an entry point plugin only when its name is in
`plugins.enabled`.

## Configure

1. In Agenomic, create the instance (`POST /v1/hermes/instances`, Owner). The
   answer carries a runtime token `agmhr_...` and a supervisor token
   `agmhs_...`, shown once.
2. Give the Hermes process `AGENOMIC_HERMES_RUNTIME_TOKEN=agmhr_...` and no
   provider key. Give the supervisor `AGENOMIC_HERMES_SUPERVISOR_TOKEN=agmhs_...`.
3. Generate the Hermes configuration:

```python
import yaml
from agenomic.integrations.hermes.config import render_hermes_config

print(yaml.safe_dump(render_hermes_config("https://agenomic.example", model="demo-model")))
```

which produces:

```yaml
model:
  provider: custom
  base_url: https://agenomic.example/v1/hermes/runtime/model/v1
  api_key: ${AGENOMIC_HERMES_RUNTIME_TOKEN}
  api_mode: chat_completions
  default: demo-model
plugins:
  enabled: [agenomic]
  hook_callback_timeout: 30
  entries:
    agenomic:
      settings:
        endpoint: https://agenomic.example
skills:
  write_approval: true
hooks:
  pre_tool_call:
    - {command: agenomic-hermes-guard, fail_closed: true, timeout: 10}
hooks_auto_accept: true
```

- `model.*` routes the main loop (and the `auxiliary.*` default route) to the
  Agenomic Model Gateway. Hermes expands `${VAR}` and `${env:VAR}` in string
  values; the rendered file uses `${VAR}`.
- `skills.write_approval: true` stages every `skill_manage` write instead of
  applying it.
- `plugins.hook_callback_timeout` also bounds shell hooks, so it is kept at
  least as large as the guard timeout.
- `hooks_auto_accept: true` registers the guard without the TTY consent
  prompt; keep `config.yaml` read only so nothing else can be added there.

### Adapter config `agenomic.hermes.adapter_config/v1`

The adapter reads `plugins.entries.agenomic.settings` (through
`ctx.get_config`) and, when `AGENOMIC_HERMES_CONFIG` names a YAML or JSON
file, that file (its keys win). Unknown keys are rejected.

| Key | Default | Meaning |
| --- | --- | --- |
| `endpoint` | required | Agenomic base URL |
| `runtime_token` | `${env:AGENOMIC_HERMES_RUNTIME_TOKEN}` | must be an `${env:VAR}` reference; a missing variable is a `ConfigError` naming the variable, never its value |
| `mode_hint` | none | informational; the server decides the mode |
| `timeouts` | `connect_s: 3`, `decision_s: 5`, `report_s: 10` | HTTP budgets |
| `buffer` | `max_events: 10000`, `max_bytes: 16 MiB`, `flush_interval_s: 1`, `batch_size: 500`, `spool_path: none`, `spool_max_bytes: 64 MiB` | event exporter bounds and optional JSONL spool |
| `capture` | `content: metadata`, `preview_chars: 200` | `metadata` sends hashes only; `redacted_preview` adds redacted, truncated previews |
| `fail_mode` | `closed` | the only value: in enforce, no valid decision means the action is blocked |

Hermes expands `${...}` in plugin settings before the adapter reads them, so a
`runtime_token` set in the plugin settings would arrive as a literal and is
refused. Keep the default variable, or put the reference in the
`AGENOMIC_HERMES_CONFIG` file (`render_adapter_config(...)` writes one).

## Modes

The server computes the effective state on every admission and returns it in
`/hello`, `/heartbeat`, `/sessions` and every `authorize` answer.

| Effective state | Adapter behaviour |
| --- | --- |
| `observe` | events only, `authorize` is never called |
| `shadow` | `authorize` is called and recorded as counterfactual, nothing is blocked; an authorization outage emits `authorization.unavailable` and the call proceeds |
| `enforce` | `allow` executes; `deny` and `require_approval` block; any error, timeout or invalid answer blocks |
| `enforce_blocked`, `paused`, `quarantined`, `revoked` | treated as enforce; the server denies |
| unknown (no answer yet) | treated as enforce: no valid decision, no execution |

## How a tool call is controlled

Hermes has two call orders, both verified in the pinned source:

- agent loop (`agent/tool_executor.py`): the `tool_execution` middleware runs
  **outside** `pre_tool_call`;
- direct dispatch (`model_tools.handle_function_call`): `pre_tool_call` runs
  first, then the middleware.

Whichever of the two runs first for a `tool_call_id` calls
`POST /sessions/:sid/actions/authorize` (`decision_s` budget); the other reuses
that decision. The middleware never raises before `next_call` (Hermes would
skip the frame and execute: fail open); it returns
`{"error": "Agenomic: no valid authorization for this action"}` instead.

- `allow`: the adapter keeps `record_id`, the permit and the arguments hash,
  executes once, then posts `/actions/report` with the permit, `is_error`,
  `duration_ms` and `result_hash` (`result_preview` only with
  `redacted_preview`). A failed report emits `action.report_failed` with
  `external_state: unknown` and is retried by the heartbeat; the action is
  never executed again.
- `deny`: `Agenomic denied <tool>: <explanation> (decision <id>)`.
- `require_approval`: `Agenomic approval <id> required; the action was not
  executed. Retry the same call after approval.` The adapter remembers the
  identity (`logical_call_id`, `attempt`) under `(session, tool, arguments
  hash)`. When the model issues the same call again, the adapter reads
  `GET /approvals/:id`: still pending blocks again, rejected or expired blocks
  and forgets the identity, approved retries `authorize` with the original
  identity so the gateway resumes and consumes the approval.
- The arguments hash is `blake3` over the gateway's canonical JSON
  (`agenomic.canon/v1`, test vectors generated from agenomic-cloud in
  `tests/fixtures/hermes_canonical_vectors.json`). Arguments that change
  between authorization and execution are blocked in enforce;
  `post_tool_call` compares the executed arguments and emits
  `authorization.argument_mismatch` when another plugin modified them.
- `delegate_task` first reserves `POST /sessions/:sid/delegations` (`count` =
  number of tasks). `subagent_start` links the child session (it fires before
  the child's `on_session_start`), which is admitted with
  `parent_hermes_session_id`, `subagent_id` and `delegation_id`.
- Writes by `write_file`/`patch` into `$HERMES_HOME/skills`, `plugins`,
  `config.yaml`, `.env`, `/etc/hermes` or the profile's `protected_paths` are
  decided by the server and, in enforce, also denied locally.
- Other plugins' `pre_tool_call` callbacks and `tool_request`/`tool_execution`
  middleware can change arguments after the decision. They are reported in
  `/hello` as `foreign_mutators`; the server refuses enforce while any exists.

The adapter never answers a Hermes approval: approval hooks are observed only.

## Events

Events follow `agenomic.hermes.event/v1` (`event_id` ULID, per process `seq`,
`trace_id` = root Hermes session, `span_id` per API request and tool call,
`parent_span_id` for subagents). Content is reduced to `blake3:` hashes before
it reaches the queue or the spool. With `capture.content: redacted_preview`,
previews go through `RedactionEngine` (credential keys masked at any depth),
credential pattern masking (`agmhr_`, `sk-`, bearer tokens, ...) and
truncation.

The exporter never blocks the agent: a bounded buffer, one daemon thread,
batches of at most 500 to `POST /events`, 3 retries with backoff,
deduplication by `event_id`, drop and count on overflow, and an optional
size capped JSONL spool (mode `0600`) for undelivered batches. Its stats
(`buffered`, `dropped`, `buffer_full`, `last_flush_error`) go into every
heartbeat. Telemetry is not the security record: decisions are stored server
side when they are made, so dropped events never change a decision.

## Commands

`POST /heartbeat` returns commands for the plugin; each is acknowledged
`received` first.

| Command | Plugin effect | `applied` when |
| --- | --- | --- |
| `pause`, `revoke` (instance) | local status set, guard blocks; the server already denies | immediately (local state changed) |
| `resume` | local status cleared | immediately |
| `cancel` subagent | `tools.delegate_tool_registry.interrupt_subagent(id)` (cooperative) | `subagent_stop` is observed |
| `cancel` session | further tool calls in that session are blocked; a subagent session is interrupted | `on_session_end` with `interrupted`, `subagent_stop` or `on_session_finalize` is observed |
| `quarantine` | local status set; the process stop is the supervisor's | by the supervisor |

A root session exposes no interrupt handle to plugins, so a cancel of a root
session blocks its tools and waits for Hermes to end the session. `refused` is
sent for unknown commands, missing targets and subagents that are not running.

## Supervisor

```bash
AGENOMIC_HERMES_ENDPOINT=https://agenomic.example \
AGENOMIC_HERMES_SUPERVISOR_TOKEN=agmhs_... \
AGENOMIC_HERMES_RUNTIME_TOKEN=agmhr_... \
agenomic-hermes-supervisor --skills-dir /srv/hermes-skills \
  --forbidden-host api.openai.com:443 --child-uid 1000 --child-gid 1000 \
  -- hermes gateway
```

- The child environment is built from an allowlist (`PATH`, `HOME`, locale,
  `HERMES_HOME`, `AGENOMIC_HERMES_CONFIG`, ... plus `--allow-env`). Every
  `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD` is removed even when
  allowlisted, except the runtime token variable. The supervisor token is
  never passed.
- Every heartbeat (`--interval-s`, default 15) sends the process state and an
  isolation self check: `provider_secrets_absent` (child environment),
  `egress_restricted` (true only if a TCP connect to every `--forbidden-host`
  fails), `config_readonly` and `skills_readonly` (mode bits and ownership of
  the files and their directories for the child uid, read only mounts),
  `docker_socket_absent`, `runs_as_non_root`. The checks run from the
  supervisor's own network namespace and filesystem view, which is the
  child's when both run in the same container.
- `quarantine` and `revoke`: SIGTERM to the child's process group, SIGKILL
  after `--grace-s`, restarts refused, `applied` with the exit code. `resume`
  allows restarts again and starts the child.
- Approved skills (`GET /skills/approved`) are written into `--skills-dir`
  after their digest (`sha256:` or `blake3:`) is checked; targets escaping the
  directory are rejected; files a previous sync wrote and that are no longer
  approved are removed. Mount that directory read only into the agent.
- SIGTERM or SIGINT stops the child and sends a final heartbeat.

## Guard

`agenomic-hermes-guard` reads the Hermes shell hook payload on stdin and the
status file the plugin keeps fresh, `$HERMES_HOME/agenomic/status.json`
(`loaded`, `instance_status`, `effective_state`, `updated_at`). It blocks when
the file is missing, unreadable, older than `AGENOMIC_HERMES_GUARD_MAX_AGE_S`
(default 120 s), dated in the future, not `loaded`, or says `paused`,
`quarantined` or `revoked`. Upstream allows a `fail_closed` hook that exits
non zero with an empty stdout, so every failure path, including internal
errors, prints `{"action": "block", "message": ...}` and exits 2. Allowing
prints nothing and exits 0. Shell hooks are registered by the Hermes CLI,
gateway and TUI; a bare programmatic `AIAgent` does not register them.

## What is not covered

- The plugin is not a boundary: code the agent runs (terminal, `execute_code`,
  scripts) can do anything the process can. `terminal` and `execute_code` are
  decided as one action on the whole command; nothing is promised about
  operations inside a script.
- Auxiliary model calls (titles, compression, vision, MoA, smart approvals,
  MCP sampling, plugin `ctx.llm`) and the iteration limit summary bypass the
  `llm_*` middleware. They are observed through `pre/post_auxiliary_call`
  when Hermes fires them and are only controllable at the Model Gateway, with
  egress restricted.
- `codex_app_server` and ACP providers run turns in external processes.
- Arguments changed by another plugin after the decision are detected after
  execution (`authorization.argument_mismatch`), not prevented; enforce is
  refused while such callbacks exist.
- A missing Python interpreter for the guard falls into the upstream gap
  (non zero exit, empty stdout is allowed). The guard and its status file live
  in the agent's environment and can be forged by code the agent runs.
- Cancel of a root session is cooperative and applies only when Hermes ends
  the session.
- For a float written with 17 significant digits, the local hash may differ
  from the server's `arguments_hash` (serde_json's default float parser is not
  correctly rounded). The adapter only compares its own hashes with each
  other and the server rehashes what it receives, so decisions are not
  affected; the local hash is not a proof of the server's.
- Hermes `HERMES_SAFE_MODE=1` disables plugins and shell hooks.

## Live demo and measurements

`examples/12_hermes_agent/live_demo/live_demo.py` (root, Linux, a running
agenomic-cloud gateway started with `AGENOMIC_HERMES_ALLOW_PRIVATE_UPSTREAMS=1`,
because the scripted model upstream listens on loopback) runs the real pinned Hermes with this adapter under
`agenomic-hermes-supervisor`, as uid 10001, in a network namespace whose only
route is the gateway port and a mount namespace without `/run`. It walks
through enrollment, observe, catalog approval, enforce, an allowed call, a
denied shell command, a denied write into the skills directory, a human
approval followed by a controlled retry, a delegation, a bypass attempt,
pause and quarantine, and saves the evidence. `HERMES_HOME` is root owned
with mode `1777`, and `config.yaml`, `.env`, `SOUL.md`,
`shell-hooks-allowlist.json`, `skills/`, `plugins/` and `hooks/` are root
owned: Hermes creates its state files but cannot replace those entries.
`bench.py` measures the runtime API (authorize, report, event batches)
against the same gateway. Results and conditions: agenomic-cloud
`docs/hermes/implementation-report.md`.

## Tests

```bash
uv run --extra dev pytest -q                      # offline, Hermes not needed
/home/user/upstream/hermes-venv/bin/python -m pytest -m hermes   # real Hermes
```

The second command needs a venv with the pinned Hermes clone and this
repository installed editable (`uv venv -p 3.11`, `uv pip install -e
<hermes-agent>`, `uv pip install -e "<agenomic-python>[dev,hermes]"`). Those
tests run a real `AIAgent` against a fake Agenomic server and assert external
effects: a file created exactly once on allow, absent on deny, the block
message seen by the model, the report with the permit, the correlation
header, the plugin loaded by its entry point, and the guard blocking when
the plugin is not loaded.
