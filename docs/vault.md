# Agents Vault

> **Optional commercial module** of Agenomic Cloud/Enterprise. It needs the
> **Agents Vault add-on** on the workspace; it is not part of the open-source
> core, nothing in the offline SDK depends on it, and the SDK never decides
> whether you are entitled: it reports what the server answered.

The agent holds an **authorization**; the executor holds the **credential**.
Your agent asks Agenomic to perform one logical action (`crm.get_customer` on
the binding `crm-read`), the secure executor attaches the credential in
memory for the duration of the call, and the agent receives the business
result, already filtered by the server, and a receipt. No endpoint and no SDK
method returns a secret value.

Three different decisions are taken by three different components:

| decision | taken by | SDK surface |
| --- | --- | --- |
| storing a secret | a human with the secret permissions | `client.vault.secrets` (write-only) |
| the right to use it (binding and grant) | a human reviewer, never the proposer | `client.vault.bindings`, `client.vault.grants` |
| authorizing one action | the policy engine, plus an approval when a policy asks | `client.tools.execute` |

The server side (control plane, executor, ledger, entitlement) lives in
Agenomic Cloud; this page documents the Python client of its
`/v1/vault` contract. Operations added to the contract later (rotations,
revocation retry and lift, execution resolve, grant delegation) **require a
server version that includes them**; an older server answers 404 and the SDK
raises `VaultNotFound`.

## Quick start

An administrator sets the credential up once (this part is human work, run
it from a console or a provisioning script with an API key):

```python
from agenomic import Client
from agenomic.vault import (
    BearerAuth, BindingContent, Destination, PathParam, RequestTemplate, Sensitive,
)

admin = Client(api_key="agm_...", base_url="https://cloud.example")

provider = admin.vault.providers.list()[0]
secret = admin.vault.secrets.create(
    environment="prod", name="crm-api-key", secret_type="api_key",
    provider_id=provider.id,
    value=Sensitive.from_env("CRM_API_KEY"),      # write-only: never returned, never printed
)

binding = admin.vault.bindings.create(
    environment="prod", name="crm-read", agent_id="agent://acme/support",
    tool_name="crm.get_customer",
    content=BindingContent(
        secret_id=secret.secret.id, upstream_identity="crm-service",
        tool_contract_ref="crm.get_customer@1",
        destination=Destination(host="crm.example.com"),          # fixed server side
        auth=BearerAuth(),                                         # placement fixed server side
        request=RequestTemplate(method="GET", path="/customers/{id}",
                                path_params=[PathParam(name="id", pattern="^c_[0-9]+$")]),
        effect="read",
    ),
)
admin.vault.bindings.submit(binding.binding.id, 1)
# A human reviewer (not the proposer) approves in the console, then the version is activated:
# admin.vault.bindings.activate(binding.binding.id, 1)

issued = admin.vault.runtime_identities.issue(
    environment="prod", agent_id="agent://acme/support", label="support-bot", ttl_seconds=3600,
)
agent = issued.runtime_client("https://cloud.example")     # the token moves into this client
```

The agent runtime only ever holds a runtime token (`vrt_...`). It uses the
primary entry point:

```python
from agenomic.vault import VaultApprovalRequired, VaultNotEntitled, VaultOutcomeUnknown

try:
    out = agent.tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"},
    )
    print(out.result, out.receipt_id, out.status, out.action_id)
except VaultApprovalRequired as pending:
    print("waiting for", pending.approval_id)          # decide it, then execute again with pending.action_id
except VaultOutcomeUnknown as unknown:
    print("do not retry; verify the destination", unknown.action_id)
except VaultNotEntitled as locked:
    print("locked", locked.upgrade_hint, locked.required_plan)
```

`execute(tool=, binding=, arguments=, action_id=, deadline_ms=)` returns an
`ExecuteResult`: `result` (the filtered business answer), `receipt_id`,
`status` (`succeeded`), `status_code`, `limitations`, `replayed` and the
`action_id`. Use `aexecute` inside asyncio; the sync form blocks the loop.
A client built with `Client(runtime_token=...)` holds only that token: no API
key is needed, and the API key is never sent on a runtime route (nor the
token on an admin route).

## Idempotency, retries and `outcome_unknown`

`action_id` names one **logical action**. It is a UUID, generated when you do
not pass one, exposed on the result and on every error. Persist it before you
call (a graph node, a job row) so a restart resends the same action instead of
inventing a new one.

| what happens | the SDK |
| --- | --- |
| network failure, 429, 502, 503, 504 | retries up to the policy (`Client(vault_retry=RetryPolicy(...))`, default 3), **same `action_id`, same body**; the server de-duplicates, so the external effect is never repeated |
| retries exhausted | `VaultTransportError` / `VaultServerError` / `VaultRateLimited`, carrying the `action_id`; resend with the same one |
| `state: outcome_unknown` (200) or `vault_outcome_unknown` (409) | **never retried**: `VaultOutcomeUnknown(action_id, status_code, error_class, receipt_id)` |
| 409 `conflict` while the action is running | `VaultConflict`; poll `client.tools.get_execution(action_id)` instead of resending |
| anything else (4xx, `denied`, `refused`, `failed`) | raised immediately, not retried |

`outcome_unknown` means the request left the executor and its effect cannot
be established (a timeout after sending, a 5xx from a destination on a write).
Resending the same `action_id` returns the stored state again and never
repeats the effect; the SDK will not resend it for you. Verify at the
destination, then settle it, which records what you established and never
re-sends:

```python
admin.vault.executions.resolve(
    unknown.action_id, resolution="applied",          # or "not_applied"
    note="refund visible in the PSP dashboard",
)
```

Only a deliberate new `action_id` performs the action again.

Admin writes (`secrets.create`, `grants.request`, ...) are not idempotent on
the server: they are retried only when the server throttled them (429), never
after a network failure. Reads are retried like executions.

## Approvals

A policy can require a human approval before an action runs. `execute`
raises `VaultApprovalRequired` (HTTP 202) with `approval_id` and `action_id`
and executes nothing. Decide the approval with the Protect API (a human,
not the agent: runtime tokens cannot call it), then call `execute` again with
the **same `action_id`**:

```python
client.protect.approvals.decide(pending.approval_id, "approve", comment="checked")
agent.tools.execute(tool=..., binding=..., arguments=..., action_id=pending.action_id)
```

Material changes (arguments, binding version) invalidate an approval: change
the arguments and the new action needs its own.

A grant (bounded authority: `max_uses`, `ttl_seconds`, tied to a binding
version) is requested with `client.vault.runtime.request_grant(...)` or by an
administrator with `client.vault.grants.request(...)`, and approved by a
different human with a verified session. The SDK authenticates with an API
key, so `grants.approve`, `bindings.approve` and the other human-only calls are
exposed as the contract defines them but the server refuses them for an API
key with `VaultPermissionDenied`: approve in the console.
`runtime.delegate_grant(...)` hands a narrower slice of an approved grant to
another agent of the same environment.

## Locked add-on

Business operations (execute, create secret, binding, grant, rotation) need
the add-on; safety operations (revocation, kill switch, metadata and evidence
reads) never do. A locked workspace answers HTTP 403 with
`capability_not_entitled`, `capability_not_in_edition` or
`capability_disabled`, which is `VaultNotEntitled`:

```python
status = admin.vault.status()            # readable even when locked
if status.locked:
    show_banner(upgrade=status.upgrade_hint, plan=status.capability.required_plan)
try:
    ...
except VaultNotEntitled as locked:       # locked.locked is True
    show_banner(upgrade=locked.upgrade_hint, plan=locked.required_plan)
```

`upgrade_hint` mirrors the server's `reason` (`not_entitled`,
`not_in_edition`); `disabled` and `unavailable` are not upgrade cases. The SDK
carries the server's plan hint verbatim and never computes entitlement or
prices.

## Errors

Every vault error subclasses `agenomic.tools.ToolExecutionError`
(`code`, `status`) and so `CloudError` and `AgenomicError`; the vault ones add
`request_id` and `action_id`.

| server answer | exception | notes |
| --- | --- | --- |
| 403 `capability_*` | `VaultNotEntitled` | `.locked`, `.upgrade_hint`, `.reason`, `.capability`, `.required_plan` |
| 202 / `status: approval_required` | `VaultApprovalRequired` | `.approval_id`, `.action_id`; nothing executed |
| 403 / `status: denied` | `VaultPolicyDenied` | `.reason_codes`, `.explanation`; nothing executed |
| 200 `state: refused` | `VaultPolicyDenied` | stored denial, reasons from `error_class` |
| 409 `vault_grant_unusable` | `VaultGrantUnusable` | `.reason`: `not_found`, `not_approved`, `expired`, `exhausted`; `.missing`, `.exhausted` |
| 409 `vault_revoked` | `VaultRevoked` | binding, secret, grant or identity revoked or suspended |
| 200 `state: outcome_unknown` / 409 `vault_outcome_unknown` | `VaultOutcomeUnknown` | never retried; `.action_id` |
| 200 `state: failed` | `VaultExecutionFailed` | `.error_class`, `.status_code`, `.result` |
| 200 in-flight state | `VaultExecutionInProgress` | poll `get_execution` |
| 429 `too_many_requests` | `VaultRateLimited` | `.retry_after`; retried first |
| 400 `validation_error` | `VaultValidationError` | also raised locally before any request |
| 401 | `VaultAuthenticationError` | wrong plane or expired token |
| 403 `forbidden` / `vault_permission_denied` | `VaultPermissionDenied` | missing permission or human assurance |
| 404 | `VaultNotFound` | foreign ids answer 404 too |
| 409 `conflict` | `VaultConflict` | incl. action_id reused for another request |
| 409 `vault_fail_closed` | `VaultFailClosed` | a precondition could not be established |
| 409 `vault_backend_*`, `vault_destination_unavailable` | `VaultUnavailable` | not retried |
| 409 other `vault_*` / `status: refused` | `VaultRefused` | `.code` |
| 5xx | `VaultServerError` | |
| no answer | `VaultTransportError` | resend with the same `action_id` |
| missing key, token or `base_url` | `VaultNotConfigured` | raised before any request |
| no replay fixture | `ReplayFixtureMissing` | code `mock_unmatched` |

Unknown codes fall back to `VaultError` (or by HTTP status), never to a bare
`CloudError`. Exceptions are picklable and never carry arguments, bodies or
secret values; a server that echoes a submitted value in an error message has
it masked.

## Secrets are write-only

A value enters through `Sensitive` and leaves only in the request body that
stores it:

```python
Sensitive("s3cr3t-value")        # repr, str, format, %-format, pickle, copy, JSON, pydantic: "**********"
Sensitive.from_env("CRM_API_KEY")
```

- the mask is constant whatever the length; there is no public accessor, no
  `__dict__`, equality does not compare values;
- a plain `str` is refused by `secrets.create`, `secrets.add_version` and
  `secrets.rotate` (`TypeError`); a pydantic `SecretStr` is accepted;
- a `Sensitive` cannot ride in `execute(arguments=...)`: it is refused
  locally (`sensitive_not_allowed`);
- request bodies never appear in exceptions, logs (`agenomic.vault` logs the
  route and status only) or chained exceptions; response models drop unknown
  fields, so a value a server should never send cannot land in an object;
- the one-time runtime token is an `IssuedToken`: masked everywhere, taken once
  with `issued.token.consume()`, or moved into a client with
  `issued.runtime_client(base_url)`.

`secrets.rotate(...)` (same as `rotations.start`) writes the new version,
verifies it, keeps it pending until `rotations.activate(...)`, and keeps the
previous version for the `overlap_seconds` rollback window (`rotations.rollback`).

## Replay

Replay is **mock by default**. Attach a fixture set and
`client.tools.execute` is answered from it without touching the network; a
call no fixture matches raises `ReplayFixtureMissing` (code `mock_unmatched`,
the code of the tool mock engine). There is no switch that turns a miss into a
live call.

```python
from agenomic import Client
from agenomic.vault import ReplayFixture, ReplayFixtureMissing, ReplayOutcome, VaultReplay

replay = VaultReplay([
    ReplayFixture(fixture_id="fx-1", tool="crm.get_customer", binding="crm-read",
                  arguments={"id": "c_1"},
                  outcome=ReplayOutcome(result={"name": "Ada"}, receipt_id="rcpt-1")),
    ReplayFixture(fixture_id="fx-2", tool="payments.refund", binding="pay",
                  arguments={"amount_minor": 900},
                  outcome=ReplayOutcome(kind="outcome_unknown", status_code=504)),
])
client = Client(vault_replay=replay)        # no base_url, no token: nothing can go live
client.tools.execute(tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"})
```

- the match is exact on tool, binding and the canonical hash of the arguments;
- an outcome is `succeeded`, `failed`, `outcome_unknown`, `denied` or
  `approval_required` and raises exactly the exception a live execution
  raises, so error handling is tested offline;
- the same `action_id` replays the same outcome; another request on it is a
  `VaultConflict`, as on the server;
- `replay.calls` lists what was served; `VaultReplay.from_file(path)` /
  `replay.save(path)` use the `agenomic.vault_replay/v1` JSON document;
- `get_execution` serves stored outcomes; grants have no replay counterpart
  and raise `ReplayUnsupported` rather than going live;
- fixtures hold business results only. A vault execution never returns a
  credential, so there is none to record.

Replay applies to the execution path. The admin plane (`client.vault`) is
unaffected: build a client without `base_url` or API key if a test must be
unable to reach it. Replays driven by the Tool Gateway (`client.tools` runs)
are configured server side.

## LangGraph

`examples/12_vault_langgraph.py` runs offline from replay fixtures
(`python examples/12_vault_langgraph.py`) and against a real workspace when
`AGENOMIC_BASE_URL` and `AGENOMIC_VAULT_RUNTIME_TOKEN` are set. The rules it
follows:

- the state, and so every checkpoint, carries **opaque references only**: the
  binding name, the `action_id`, the `receipt_id`, the `approval_id` and a
  status string;
- a `plan` node fixes the `action_id` once and checkpoints it, so a resumed
  graph resends the same logical action and the server de-duplicates it;
- the `execute` node calls `client.tools.execute` and turns the typed errors
  into a status (`approval_pending`, `outcome_unknown`, `denied`, `locked`);
- the business result is read in-process with `client.tools.get_execution`
  by `action_id` instead of being stored in the state;
- resuming after an approval re-invokes the graph with the checkpointed
  `action_id`.

`tests/test_vault_langgraph.py` creates a secret with a canary value, runs the
graph with an `InMemorySaver` and serializes every checkpoint (LangGraph's
serializer, the SDK's canonical CBOR and JSON) asserting the canary is
absent, with a positive control proving the scan can see a leak.

## What is not supported

- **Reading a secret.** There is no `read`, `get_value`, `reveal` or export of
  a value, by contract. A test fails if such a public name appears.
- **Human-only operations with an API key.** Approving grants and binding
  versions needs a verified human session; the SDK authenticates with an API
  key only. Lifting a kill switch and settling an unknown outcome are also
  human actions in the contract.
- **Retrying or reconciling `outcome_unknown` automatically**, or forcing a
  resend. Settle it with `executions.resolve` or use a new `action_id` on
  purpose.
- **Waiting for approvals.** A runtime token cannot read approvals, so there
  is no polling helper; call `execute` again with the same `action_id`.
- **Verifying receipt signatures.** The SDK returns `receipt_id` and, to
  administrators, the receipt records (`client.vault.receipts`); it does not
  verify their signatures or the ledger chain.
- **Local mode.** Without `base_url` there is no vault; only a replay set
  answers.
- **Recording replays, wildcard fixtures, replaying grants.**
- **Pagination and server-side filters beyond the ones exposed**
  (`environment`, `provider_id`, `agent_id`, `binding_id`, `secret_id`,
  `state`, `action_id`, `limit`).
- **mTLS or SPIFFE runtime identities**: runtime identities are enrollment
  tokens, bearer-level, as the server documents.
- **Entitlement, plans and prices.** Reported by the server, never computed
  by the SDK.
- **Session or cookie authentication, and the MCP surface.**

## Contract coverage

Every operation of the `/v1/vault` contract (45 operations on 39 paths) has a
sync and an async method; `tests/test_vault_admin.py` fails when a route of
the contract snapshot is unreachable from the SDK.

| namespace | operations |
| --- | --- |
| `client.tools` | `execute`, `aexecute`, `get_execution`, `aget_execution` |
| `client.vault` | `status`, `kill_switch` |
| `client.vault.providers` | `list`, `get`, `create`, `health`, `set_state` |
| `client.vault.secrets` | `list`, `get`, `create`, `register_reference`, `add_version`, `rotate`, `revoke` |
| `client.vault.rotations` | `start`, `list`, `get`, `activate`, `rollback` |
| `client.vault.bindings` | `list`, `get`, `create`, `propose_version`, `submit`, `approve`, `reject`, `activate`, `revoke` |
| `client.vault.grants` | `list`, `request`, `approve`, `deny`, `revoke` |
| `client.vault.runtime_identities` | `list`, `issue`, `revoke` |
| `client.vault.executions` | `list`, `get`, `resolve` |
| `client.vault.receipts` | `list` |
| `client.vault.revocations` | `list`, `retry`, `lift` |
| `client.vault.runtime` | `execute`, `get_execution`, `list_grants`, `request_grant`, `delegate_grant` |

Async variants carry the `a` prefix (`alist`, `acreate`, `aexecute`, ...).
