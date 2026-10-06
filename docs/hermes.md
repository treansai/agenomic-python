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

The endpoint is validated first (absolute http(s) URL, no credentials, no
query or fragment, not even a bare `?` or `#`), as the adapter, the supervisor and the HTTP clients do, and so are the plugin
`settings`: unknown keys and literal values the adapter would refuse are
rejected, while `${env:VAR}` references are checked when the adapter loads
(a value made only of references is checked then; a reference mixed with
literal text is validated as written, so it fits text fields such as
`buffer.spool_path` but never a number or an enumeration; `timeouts`,
`buffer` and `capture` are mappings and
take references in their fields, never as a whole, and an unknown field is
refused even when its value is a reference). A
configuration Hermes could not load is never rendered. It produces:

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
  applying it; the plugin sends each staged write to Agenomic as a change
  proposal. A staged write whose content or diff carries a credential is never
  sent (a local `skill.proposal.refused` event, reason `credential_detected`).
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
| `runtime_token` | `${env:AGENOMIC_HERMES_RUNTIME_TOKEN}` | must be an `${env:VAR}` reference to a variable that is not the supervisor token, a provider key or another credential shaped name (`*_API_KEY`, `*_TOKEN`, `*_SECRET`, ... unless `AGENOMIC_`); its value must be a runtime token (`agmhr_...`). A refused name, a missing variable or another value is a `ConfigError` naming the variable, never its value |
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
The first `/hello` and the tool discovery happen when the plugin starts; a
failure there (gateway unreachable, an exception from a third party registry)
is never fatal: the heartbeat thread still starts and retries both on every
tick, and a discovery failure never skips that tick's heartbeat, so the guard
status and command polling stay alive. A tool whose name or toolset is
credential-shaped is not catalogued and its calls are never sent to
`/authorize` either (its identifier never leaves the process: blocked in
enforce, unchanged execution in shadow and observe), and schemas are redacted
before they are sent.

| Effective state | Adapter behaviour |
| --- | --- |
| `observe` | events only, `authorize` is never called (also when a `/hello` sent during an authorization switches to observe: the call proceeds without asking, and an authorization outage never blocks); local checks (protected paths, incompatible Hermes, argument mutators the gateway has not confirmed) emit one local `tool.call.decision` per call (`deny`, `local_mode: observe`, with a counterfactual) and the call proceeds; an approval required in enforce for the same action (session, tool, arguments hash) and still held locally is recorded the same way as `require_approval` (reason `approval_pending`, its `approval_id` in `extra`), without claiming, consuming or dropping it and without asking its status, so a later enforce retry still resumes under it; delegations are not reserved (a delegation refusal is only known by asking the gateway, so nothing is recorded for it) |
| `shadow` | `authorize` is called and recorded as counterfactual, nothing is blocked; an authorization outage emits `authorization.unavailable` and the call proceeds; local checks (protected paths, incompatible Hermes, unconfirmed mutators, refused delegation, arguments without a canonical form) emit a local `tool.call.decision` and the call proceeds; an approval still pending from an earlier enforce is not consulted (the call is asked afresh) |
| `enforce` | `allow` executes; `deny` and `require_approval` block; any error, timeout or invalid answer blocks, including a decision contradicting its HTTP status (`allow`/`observe` 200, `require_approval` 202, `deny` 403) and an `observe` decision whose `effective_mode` is not observe (or the reverse): it never downgrades enforcement |
| `enforce_blocked`, `paused`, `quarantined`, `revoked` | treated as enforce; the server denies |
| unknown (no answer yet) | treated as enforce: no valid decision, no execution |

Two local reasons block in every mode, shadow and observe included, because
they are operator commands rather than policy decisions: a pending `cancel` of
the session or of the subagent it runs as, and a local `pause`, `quarantine` or
`revoke` status (it makes the adapter treat the instance as enforce until
`resume`). Both are checked at each gate before an authorization cached by the
other gate is reused, so a command applied between the two gates still stops the
call. A call blocked by a pending cancel is recorded as a local
`tool.call.decision` (`deny`, reason code `cancel_pending`); one blocked by a
local `pause`, `quarantine` or `revoke`, with reason code `instance_stopped`.

Admission at the execution gate is atomic with server state updates: an
observe or shadow decision is never executed once enforce (or a blocking
state) has been applied, even when that answer lands after the gate's stale
check; the call is blocked and is decided again on retry. The observe path at
the execution gate, and `pre_tool_call` in every mode, make the same recheck
(local commands and the raw state, `enforce_blocked` included) before they let
a call through. In the agent loop order, a call the middleware admitted in
observe or shadow is blocked by the inner `pre_tool_call` if enforce applies
in between: an authorization made there would belong to no execution. A call
the middleware lets through without an authorization (observe, a shadow
fail-open) goes through the same locked recheck of local commands, blocking
states and the mode first. A local `pause`,
`quarantine`, `revoke` or `cancel` is written under the same lock and checked
again at admission, so one applied while the authorization was in flight
still stops the call.

## How a tool call is controlled

Hermes has two call orders, both verified in the pinned source:

- agent loop (`agent/tool_executor.py`): the `tool_execution` middleware runs
  **outside** `pre_tool_call`;
- direct dispatch (`model_tools.handle_function_call`): `pre_tool_call` runs
  first, then the middleware.

Whichever of the two runs first for a `tool_call_id` calls
`POST /sessions/:sid/actions/authorize` (`decision_s` budget); the other reuses
that decision. One authorization per call identity is in flight at a time: a
concurrent invocation with the same `tool_call_id` is blocked in enforce (it
would otherwise get its own permit and execute twice) and gets no permit in
shadow. The arguments sent to `/authorize` (and reported with the permit) are a
redacted copy: credential-named keys and credential-shaped values are masked,
so the gateway's policy, hash and permit bind that copy; the binding to the
arguments that actually execute is the adapter's own hash of the originals,
checked at every gate. The middleware never raises before `next_call` (Hermes would
skip the frame and execute: fail open); it returns
`{"error": "Agenomic: no valid authorization for this action"}` instead.

- `allow`: the adapter keeps `record_id`, the permit and the arguments hash,
  executes once, then posts `/actions/report` with the permit, `is_error`,
  `duration_ms` and `result_hash` (`result_preview` only with
  `redacted_preview`). A failed report emits `action.report_failed` with
  `external_state: unknown`; a transient failure (transport, timeout, 5xx,
  408, 425, 429) is retried by the heartbeat, a permanent one (any other 4xx,
  or an invalid permit) is not. The action is never executed again.
- `deny`: `Agenomic denied <tool>: <explanation> (decision <id>)`.
- `require_approval`: `Agenomic approval <id> required; the action was not
  executed. Retry the same call after approval.` The adapter remembers the
  identity (`logical_call_id`, `attempt`) under `(session, tool, arguments
  hash)`. When the model issues the same call again, the adapter reads
  `GET /approvals/:id`: still pending blocks again, rejected or expired blocks
  and forgets the identity, approved retries `authorize` with the original
  identity so the gateway resumes and consumes the approval.
- Identical calls that reach the gateway concurrently can each be issued their
  own approval. Every one is kept with its own identity, in issue order; a
  retry tries the unclaimed ones in that order, skips those still pending
  (and forgets rejected or expired ones), and resumes the identity of the
  first one granted, never a merge of two. Using or refusing one approval
  removes only that one; the others keep waiting for their own retry. When
  none is granted the call is blocked with the `still pending` message of the
  first one still waiting.
- A retry always arrives with a new `tool_call_id` (the model emits a new tool
  call; Hermes v2026.9.24 never reissues the blocked one), so a retry cannot be
  told apart from another identical call: every call with the same session,
  tool and arguments hash is treated as a retry of the pending action. One
  approval authorizes exactly one execution: the first retry claims the
  pending approval while it asks the gateway, and once it is allowed the
  identity is forgotten. Another identical call that arrives while the claim
  is held is blocked (`Agenomic approval <id> authorizes a single execution and
  another call is using it; ...`, recorded as a local `tool.call.decision`);
  an identical call after the approval was used asks the gateway as a new
  action and needs its own approval. While the approval is still pending,
  identical calls only see `still pending` and never execute. If the answer to
  the claimed retry is lost (timeout, transport error) the claim is released,
  so the next retry resumes the same identity and the gateway answers it.
- The arguments hash is `blake3` over the gateway's canonical JSON
  (`agenomic.canon/v1`, test vectors generated from agenomic-cloud in
  `tests/fixtures/hermes_canonical_vectors.json`). Arguments without a
  canonical form (`NaN`, a set, an object another plugin put there) cannot be
  authorized: the call is blocked in enforce and proceeds in shadow and
  observe; in every mode a local `tool.call.decision` (`deny`, reason
  `arguments_not_canonical`, with a counterfactual outside enforce) is
  recorded once per call. Arguments that change
  between authorization and execution are blocked in enforce;
  `post_tool_call` compares the executed arguments and emits
  `authorization.argument_mismatch` when another plugin modified them.
- `delegate_task` first reserves `POST /sessions/:sid/delegations` (`count` =
  number of tasks). `subagent_start` links the child session (it fires before
  the child's `on_session_start`), which is admitted with
  `parent_hermes_session_id`, `subagent_id` and `delegation_id`. A
  reservation is queued for children only once the action is allowed, and
  dropped when Hermes blocks the call afterwards or it fails. A reservation
  belongs to one invocation: a retry of the same action (after an approval or
  a transport error) reuses it, while a concurrent identical `delegate_task`
  reserves its own, so every allowed call queues exactly one reservation. A
  reservation made by a call that then required approval follows that
  approval: the retry resumed under it reuses it, and a rejected or expired
  approval drops it.
- Writes by `write_file`/`patch` into `$HERMES_HOME/skills`, `plugins`,
  `config.yaml`, `.env`, `/etc/hermes` or the profile's `protected_paths` are
  decided by the server and, in enforce, also denied locally.
- Other plugins' `pre_tool_call` callbacks and `tool_request`/`tool_execution`
  middleware can change arguments after the decision. They are reported in
  `/hello` as `foreign_mutators`; the server refuses enforce while any exists.
  Only the exact guard shell hook (`agenomic-hermes-guard`, or an absolute path
  to it) is excluded: a wrapper or compound command mentioning it is reported.
  The provider `base_url` in `/hello` has its userinfo and query values masked.
  The list is re-read before every authorization: when it differs from the
  one the server last confirmed, a `/hello` is sent first; if that hello is
  not delivered the call is blocked in enforce (in shadow it proceeds), and a
  local `tool.call.decision` with `foreign_mutators_unconfirmed` is emitted.

The adapter never answers a Hermes approval: approval hooks are observed only.

## Events

Events follow `agenomic.hermes.event/v1` (`event_id` ULID, per process `seq`,
`trace_id` = root Hermes session, `span_id` per API request and tool call,
`parent_span_id` for subagents). Content is reduced to `blake3:` hashes before
it reaches the queue or the spool. With `capture.content: redacted_preview`,
previews go through `redacted_preview` in `exporter.py` (credential keys masked at any depth,
including a content mapping's own top-level keys), credential pattern
masking and truncation. Pattern masking covers token shapes (`agmhr_`, `sk-`,
`ghp_`, `github_pat_`, `hf_`, `AKIA...`/`ASIA...`, `AIza...`, `glpat-`, `npm_`, `pypi-`, Stripe `sk_live_`/`rk_live_`, SendGrid `SG.`, Shopify `shpat_`, `xox?-`, JWTs, private key blocks), `Authorization`/`Proxy-Authorization`
values for every scheme (`Bearer`, `Basic`, `Digest`, `Token`, `ApiKey`,
`AWS4-HMAC-SHA256`, ...), cookie headers, `key=value` and `key: value` pairs
whose key names a credential (`api_key`, `x-api-key`, `password`, `token`,
`client_secret`, `Credential`, `Signature`, ...) and the password of URL user
info (`scheme://user:***@host`). A mapping key is a credential when, case and
separators removed, it ends in `token`, contains `password`, `passwd`,
`passphrase`, `secret`, `apikey`, `authorization`, `privatekey`, `cookie` or
`credential`, or is exactly `auth`, `pass`, `pwd`, `bearer`, `jwt`, `otp`,
`totp`, `csrf` or `xsrf`. Those short names only match whole, so `author`,
`authority`, `bypass`, `session_id`, `max_tokens` or `jwt_issuer` stay readable;
`password_policy` is masked (masking a non-secret is the safe failure). The scheme, header and key names stay
readable; prose such as "token budget" or `max_tokens=512` is not masked. In every capture
mode the event fields and the whole `extra` mapping, its top-level keys
included, get the same key and pattern masking. A value that is not a JSON
scalar (an exception, bytes, any other object) is replaced by its masked
`str()`, and a non-finite float by `null`, so nothing reaches serialization
unmasked and every event is strict JSON.

The exporter never blocks the agent: a bounded buffer, one daemon thread,
batches of at most 500 events and about 1 MiB to `POST /events` (each batch is
also capped by serialized size, with headroom for the envelope, under the
documented 1 MiB body limit; a batch always carries at least one event), 3
retries with backoff for transient failures (transport, timeout, 5xx, 408, 425,
429), deduplication by `event_id`, drop and count on overflow, and an optional
size capped JSONL spool for undelivered batches. A permanent refusal (another
4xx) is dropped and counted at once, never retried nor spooled, so it cannot
block later replays. A batch counts as delivered only on a consistent
acknowledgement (`accepted` + `duplicates` + `rejected` covering every event
sent); any other answer is handled as a transient failure and a replayed batch
stays in the spool. The spool is `0600` even
when the file already existed with a wider mode, and a directory the exporter
creates for it is `0700`. Every open of the spool, reads included, uses
`O_NOFOLLOW` and an owner check: a spool that is a symbolic link or belongs to
another user is never read, written, truncated or replaced; the exporter logs
an error and stops spooling for its lifetime (overflow is then dropped and
counted). The spool (like the status file below) is rewritten through a new
temporary file with an unpredictable name, created exclusively with mode
`0600`, so nothing planted next to it is opened or followed. The spool is
replayed one batch at a time when the buffer is idle and, under continuous
load, after every 10 live batches or once per `flush_interval_s`. A replayed
record must have the `agenomic.hermes.event/v1` shape (schema, string
`event_id` and `type`, no unknown top-level key) and goes through the
redaction walk again before it is sent; anything else is dropped and counted.
Replayed batches are capped by serialized size like live ones, and a replayed
record goes through the same per-event check as a submitted event: above the
gateway's 64 KiB per-event limit its `extra` becomes `{"truncated": true}`,
and an event still above the limit (submitted or replayed) is dropped and
counted, since the gateway would reject it. A spool that cannot be inspected
(a `stat`, read or rewrite failure) is logged once, counted, and replay backs
off for a few flush intervals; live delivery continues. Any unexpected error
in a delivery or replay step is logged and the worker thread carries on: it
never dies silently.
A replayed batch stays in the file until the server acknowledged it, then is
removed while events appended meanwhile are kept; a crash in between sends it
again, which the gateway absorbs because it deduplicates events by
`event_id` (counted as `duplicates`). Its stats
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
| `cancel` subagent | `tools.delegate_tool_registry.interrupt_subagent(id)` (cooperative); until the end is observed, further tool calls of the child session are blocked in every mode (the interrupt is asynchronous) | `subagent_stop`, `on_session_end` with `interrupted` or `on_session_finalize` of the child session is observed |
| `cancel` session | further tool calls in that session are blocked; a subagent session is interrupted | `on_session_end` with `interrupted`, `subagent_stop` or `on_session_finalize` is observed |
| `quarantine` | local status set; the process stop is the supervisor's | by the supervisor |

A root session exposes no interrupt handle to plugins, so a cancel of a root
session blocks its tools and waits for Hermes to end the session. An interrupted
turn, or the final end (`on_session_finalize`), of a session with a pending
cancel (of the session or of its subagent) is reported to Agenomic as `cancelled`, a
terminal state: the gateway applies a cancel only once its session has ended in the
control plane, never on the adapter's word alone. A terminal end report that
fails transiently is retried every heartbeat (up to 10 attempts) and the
cancels waiting for it are acknowledged only once it is reported. `refused` is
sent for unknown commands, missing targets and subagents that are not running.
A cancel naming an Agenomic session id the adapter does not know yet is held
while an admission is in flight, or while an active session's admission failed
(the gateway may have created it and lost the answer): it is decided once the
id is published by an admission or retry, and refused only once no such
session remains.
An acknowledgement that fails on a transport error, a timeout, a 5xx or a
transient 4xx (408, 425, 429) is queued and retried on the next heartbeat, by
the plugin and the supervisor alike.

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
  `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, and every provider
  credential (`AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`), is
  removed even when allowlisted, except the runtime token variable. The supervisor token is
  never passed.
- `AGENOMIC_HERMES_SUPERVISOR_TOKEN` must hold a supervisor token
  (`agmhs_...`); any other value (a provider key, a runtime token...) is
  refused with exit 2 before any request, so it is never sent to the endpoint.
  The error names the variable, never its value.
- `--runtime-token-env` must not name the supervisor token, a provider key
  (`*_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`) or
  another credential shaped variable (`*_TOKEN`, `*_SECRET`, `*_PASSWORD`, ...)
  unless it is an `AGENOMIC_` name; such a name is a settings error (exit 2).
  When the variable is set, its value must be a runtime token (`agmhr_...`),
  otherwise the supervisor refuses to start Hermes (exit 2). Only such a value
  is exempt from the scrub and from `provider_secrets_absent`.
- Every heartbeat (`--interval-s`, default 15) sends the process state and an
  isolation self check: `provider_secrets_absent` (child environment),
  `egress_restricted` (true only if a TCP connect to every `--forbidden-host`
  fails), `config_readonly` (every configuration source, additively:
  `$HERMES_HOME/config.yaml`, `$HERMES_HOME/.env`, each `--config-path` and the
  `AGENOMIC_HERMES_CONFIG` file passed to the child) and `skills_readonly` (mode bits and ownership of
  the files and their directories for the child uid, read only mounts;
  `skills_readonly` is also false when the supervisor refuses `--skills-dir`,
  see below), `docker_socket_absent`, `runs_as_non_root`. The checks run from the
  supervisor's own network namespace and filesystem view, which is the
  child's when both run in the same container.
- `quarantine` and `revoke`: SIGTERM to the child's process group, SIGKILL
  after `--grace-s`, restarts refused, `applied` with the exit code. The
  process group is tracked apart from the Hermes process: members that
  outlive it (a descendant ignoring SIGTERM) get SIGKILL once the grace
  period ends, and the command is `applied` only when the group is empty.
  Otherwise the process state is `stop_failed`, the command is not
  acknowledged as applied and is executed again when the gateway delivers it
  again; a supervisor leaving with such a group exits 1. `resume`
  allows restarts again and starts the child; it also rearms the restart
  budget of a supervisor that had given up after `--max-restarts`.
- When the Hermes process exits on its own, what remains of its process group
  is stopped the same way (SIGTERM, SIGKILL after `--grace-s`) before
  anything else: before a restart, before giving up, and also after a clean
  exit. While that group cannot be emptied the process state is `stop_failed`
  (reported in every heartbeat), no replacement is started (neither a restart
  nor `resume`) and the group stop is retried on every tick; without restarts
  the supervisor retries once more on the way out and exits 1.
- With `--skills-dir`, Hermes is started (at start-up and on `resume`) only
  after a sync of the approved skills that completed with no rejected entry;
  while the gateway or the directory is unavailable, or an entry could not be
  reconciled (bad digest, unsafe or linked destination, malformed entry, a
  directory or entry of the skills directory that cannot be inspected), it
  stays stopped and every tick retries the sync, so it never loads skills left
  over from before (possibly revoked since). Pending commands are fetched and
  applied before the first start, and a restart after a crash goes through the
  same tick (heartbeat, then sync, then start), so a quarantine or revoke
  queued meanwhile applies before any replacement runs. A missing manifest
  (first sync, or deleted by the agent) reconciles the whole directory; a
  present manifest is never authoritative either (the child may share the
  supervisor's uid and rewrite it): every sync reconciles the manifest and
  everything on disk,
  symbolic links, FIFOs, sockets and devices included (the node itself is
  removed, never what a link points to, never opened).
  A SIGTERM or SIGINT received while a heartbeat or sync is blocked means
  Hermes is not started once it returns.
  A periodic sync that fails or rejects an entry while Hermes runs stops it
  (it is started again once a sync reconciles every entry), so a revoked
  skill that could not be removed is never left loaded.
- Approved skills (`GET /skills/approved`) are written into `--skills-dir`
  after their digest (`sha256:` or `blake3:`) is checked; targets escaping the
  directory are rejected; files a previous sync wrote and that are no longer
  approved are removed. Skill files and the manifest of written files
  (`.agenomic_manifest.json`) are replaced atomically (temporary file, `fsync`,
  rename), and new files are added to the manifest before they are written,
  so an interrupted sync never forgets a file. The temporary file has an
  unpredictable name and is created exclusively (`O_EXCL`, `O_NOFOLLOW`, mode
  `0600` until complete). On POSIX the skills directory is created and opened
  once without following a link: its parent is opened from `/` one component
  at a time (`O_DIRECTORY|O_NOFOLLOW`, each relative to the previous
  descriptor; a relative `--skills-dir` is made absolute against the current
  directory first, and a `..` component is refused). Every directory on the
  way must belong to the supervisor's euid or root and, when group or other
  may write it, carry the sticky bit (like `/tmp`), so nobody else can rename
  or replace a component; missing components are created (mode `0755`). The
  only links followed on the way are system layout links directly under `/`
  that root owns, with `/` itself owned by root and writable by nobody else
  (macOS `/var` and `/tmp`, merged `/usr` links). The directory is created
  relative to the parent's descriptor, then opened with
  `O_DIRECTORY|O_NOFOLLOW` and must belong to the supervisor's euid. Everything else (subdirectories, each
  opened or created with `O_NOFOLLOW` and owner checked, reads, atomic writes,
  removals, the manifest) is relative to that descriptor and never goes
  through a path resolved again, so a link the agent swaps in at any moment
  is never followed. Windows has no directory descriptors and keeps explicit
  link checks. Symbolic links are never followed: a skills directory that is
  a link, has an ancestor that is a link (other than the system layout links
  above), an ancestor of another user or an ancestor writable by others
  without the sticky bit, or belongs to another user is refused
  as a whole (logged as an error, nothing written, `skills_readonly` false),
  a skill whose path
  goes through a link (the file or a directory) is rejected, and a stale file
  behind a link is left alone. When the manifest is
  unreadable or malformed anyway, the sync logs an error and removes every
  regular file of the directory that is not approved now: the directory
  belongs to the supervisor, and only approved skills may stay there. Mount
  that directory read only into the agent.
- SIGTERM or SIGINT stops the child and sends a final heartbeat. That
  heartbeat (also sent after the child exited without restarts, or after
  restarts were exhausted) is report only: commands it returns are neither
  executed nor acknowledged, so the gateway delivers them to the next
  supervisor.

## Guard

`agenomic-hermes-guard` reads the Hermes shell hook payload on stdin and the
status file the plugin keeps fresh, `$HERMES_HOME/agenomic/status.json`
(`loaded`, `instance_status`, `effective_state`, `updated_at`). It blocks when
the file is missing, unreadable, older than `AGENOMIC_HERMES_GUARD_MAX_AGE_S`
(default 120 s), dated in the future, not `loaded`, or says `paused`,
`quarantined` or `revoked`. It also blocks unless `effective_state` is
`observe`, `shadow` or `enforce`: an unknown state (before the first `/hello`,
or a value it does not know) and `enforce_blocked` never allow a tool. The plugin writes `loaded: true` only when both
gates, the `pre_tool_call` hook and the `tool_execution` middleware, are
registered: with one missing, an argument change after authorization could go
undetected in one of the two Hermes call orders, so the guard keeps blocking.
The status file is also bound to the Hermes process that wrote it: at load the
plugin draws a random epoch, puts it in that process's environment
(`AGENOMIC_HERMES_GUARD_EPOCH`, which Hermes passes to the shell hooks it
spawns) and writes it into the file. The guard blocks unless both match, so a
fresh `loaded: true` left by a Hermes killed with SIGKILL does not open the
guard of a Hermes restarted without the plugin. Limit: a Hermes that re-executes
itself in place (`exec`, same environment) and then fails to load the plugin is
covered only by the staleness deadline. When Hermes unloads the plugin (or the process
exits) the plugin writes `loaded: false`, so the guard blocks at once instead
of when the file goes stale; it then drains the exporter, closes its HTTP
client and removes its `atexit` hook, so reloads do not accumulate adapters or
connection pools. Shutdown is idempotent and a callback that still reaches an
unloaded adapter blocks in enforce. Upstream allows a `fail_closed` hook that exits
non zero with an empty stdout, so every failure path, including internal
errors, prints `{"action": "block", "message": ...}` and exits 2. Allowing
prints nothing and exits 0. `AGENOMIC_HERMES_GUARD_MAX_AGE_S` must be a
finite number of at least 3 s (`MIN_MAX_AGE_S`): the plugin refreshes the
status file every `min(heartbeat interval, max age / 3)` seconds and never
more often than once a second, so a shorter deadline is refused and every tool
call is blocked with a message naming the variable. Set the same value in the
Hermes process so the plugin paces its refreshes to it. Shell hooks are
registered by the Hermes CLI, gateway and TUI; a bare programmatic `AIAgent`
does not register them.

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
  refused while such callbacks exist. When observe or shadow lets such a call
  run, its authorization is detached: no permit-backed action report is sent
  for arguments that did not run.
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
