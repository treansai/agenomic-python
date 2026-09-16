# Protect: policy enforcement on tool calls

A Protect run is a tool execution run whose configuration carries a
`protect` block. The Agenomic Cloud gateway evaluates the bound policies
before any tool executes; the SDK applies the decision and never executes a
call the gateway did not admit. There is exactly one policy evaluator and it
runs in the gateway: the local engine refuses a `protect` configuration with
`ToolExecutionError("protect_cloud_required", ...)`.

## Decisions on the per-call path

`router.call(...)` behaves as before for admitted calls. Two new outcomes
raise typed subclasses of `ToolExecutionError` and append the call to
`router.calls` with `status` `"pending"` or `"denied"`:

| gateway answer | SDK behaviour |
| --- | --- |
| `invoke` 202 / `local/authorize` `decision: pending` | `ToolApprovalPending(tool, approval_id, record_id, envelope)`, code `approval_pending`, status 202 |
| `invoke` 403 with the invoke envelope / `local/authorize` `decision: denied` | `ToolCallDenied(tool, envelope)`, code `policy_denied`, status 403 |
| any other `decision` string | treated as denied, never executed |

`envelope.protect` (a `ProtectDecision`: `decision_id`, `outcome`,
`effective_mode`, `reason_codes`, `approval_id`, `permit_ref`,
`policy_snapshot_digest`, `evaluated_at`), `envelope.transformation` and
`envelope.safe_explanation` mirror the gateway envelope. A transform
proposal never mutates arguments: submit a new call with the transformed
arguments and `parent_call_id` set.

Runtime-local functions receive a signed `permit` with the `local`
decision; the router forwards it verbatim to `report-local`
(`tools.report_local(..., permit=...)`). The gateway refuses a report
without a valid permit.

```python
from agenomic import Client
from agenomic.tools import ToolApprovalPending, ToolCallDenied

client = Client(api_key="agm_...", base_url="https://cloud.example")
router = client.tools.router(run_id, local_functions={"crm.update": crm_update})
try:
    router.call("payments.refund", {"amount_minor": 90000, "currency": "EUR"})
except ToolApprovalPending as pending:
    result = router.resume(pending, poll_interval=2.0, timeout=900.0)
except ToolCallDenied as denied:
    print(denied.code, denied.decision.reason_codes if denied.decision else [])
```

## Resume after approval

`router.resume(approval, *, poll_interval=2.0, timeout=900.0)` accepts the
`ToolApprovalPending` exception or the approval id, polls
`GET /v1/protect/approvals/{id}` and, once the status is `approved` or
`consumed`, re-issues the identical call identity exactly once (same tool,
arguments, `logical_call_id`, `repetition`, `attempt`, `parent_call_id` and
idempotency key) so the gateway resumes the pending claim exactly once.
A `consumed` approval means the action already executed: the replay
recovers its stored result, and a 409 answer raises
`ToolExecutionError("conflict", ..., 409)` because the evidence stands and
nothing is executed twice. `rejected`, `expired` or any other terminal
status raises `ToolCallDenied` with that status as `code`; a still pending
approval after `timeout` seconds raises
`ToolExecutionError("approval_timeout", ...)`. The router only resumes
approvals it recorded itself (`approval_unknown` otherwise).
`AsyncToolRouter.resume` awaits `asyncio.sleep` between polls.

## The `before_action` hook

`tools.router(run_id, before_action=fn)` and `tools.arouter(...)` call
`fn(identity)` with the identity dict (`tool`, `arguments`,
`logical_call_id`, `repetition`, `attempt`, `parent_call_id`) before any
request, on the local and the gateway path alike. Raising aborts the call
before anything is sent; the return value is ignored. The hook runs again
when a call is resumed.

## Instruction overlay for model calls

There is no model gateway: `model.call` is covered cooperatively. Fetch the
overlay computed for the run and let the instrumentation inject it in the
pre call window:

```python
from agenomic.integrations.openai import instrument_openai
from agenomic.integrations.anthropic import instrument_anthropic

# ProtectOverlay: version, digest, text, policies, truncated
overlay = client.protect.overlay(run_id)
openai_client = instrument_openai(OpenAI(), overlay=overlay)
anthropic_client = instrument_anthropic(Anthropic(), overlay=overlay.text)
```

OpenAI: the overlay is prepended as the first `system` message (skipped
when the first message already carries it). Anthropic: it becomes `system`
or prefixes the caller's system prompt, once. The injection is
deterministic, idempotent and happens before the request hash is computed.

## `client.protect`

Every method needs a cloud client and raises `ToolExecutionError` with the
server error code on refusal; async variants carry the `a` prefix
(`aoverlay`, `approvals.adecide`, ...).

| method | route |
| --- | --- |
| `overlay(run_id)`, `catalog(run_id)` | `GET /v1/protect/runs/{run_id}/overlay`, `/catalog` |
| `approvals.list(status=, run_id=)`, `approvals.get(id)`, `approvals.decide(id, decision, comment=)` | `/v1/protect/approvals` |
| `decisions.list(run_id=, outcome=, since=, limit=)`, `decisions.get(id)` | `/v1/protect/decisions` |
| `policies.register(document)`, `policies.get(policy_id, version)`, `policies.list()`, `policies.release(...)`, `policies.deprecate(...)`, `policies.simulate(policy_id, version, intents)`, `policies.diff(policy_id, version, against=)` | `/v1/policies`, `/v1/policies/{policy_id}@{version}/...` |
| `bindings.list(scope_kind=, scope_ref=)`, `bindings.create(policy_id, version, scope_kind, scope_ref=None, mode="enforce")`, `bindings.revoke(id, reason=)` | `/v1/protect/bindings` |
| `restrictions.list(status=)`, `restrictions.create(scope_kind=, kind=, reason=, ...)`, `restrictions.lift(id)` | `/v1/protect/restrictions` |
| `kill_switch(scope_kind, scope_ref=None, reason=)` | `POST /v1/protect/kill-switch` |
| `simulate(intents, policies=, policy_refs=, decisions_from_run=)` | `POST /v1/protect/simulate` |
| `coverage()`, `metrics_summary()` | `GET /v1/protect/coverage`, `/metrics/summary` |

The RMP Protect stage (`alerts`, `action_plan`, `recommendations`,
`notify`) stays on the same namespace.
