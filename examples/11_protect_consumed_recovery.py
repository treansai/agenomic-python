"""Protect consumed recovery contract, exercised against a real gateway.

Manual harness for closure criterion 3 of
`docs/protect/integration-test-report-2026-09-15.md`: the `approved` /
`consumed` resume contract is defined against an HTTP fixture only, so this
script replays it end to end against a running api-gateway, a real Postgres
and two distinct user owned API keys.

Environment:

    AGENOMIC_BASE_URL       gateway base URL
    AGENOMIC_API_KEY        caller, a user owned key with the write role
    AGENOMIC_REVIEWER_KEY   reviewer, a user owned key of a DIFFERENT user
    AGENOMIC_ADMIN_KEY      owner key (policy release, policy binding)
    AGENOMIC_TOOL_URL       counting HTTP tool endpoint (also serves /__count)
    AGENOMIC_ENV_PROFILE    tool-execution profile holding URL and allowing
                            the loopback destination
    AGENOMIC_APPROVAL_TTL_SECS  the gateway's protect.approval_ttl_secs

The reviewer_distinct_from_principal obligation refuses a self approval and
refuses a credential with no user, so caller and reviewer must both be user
owned keys of two different users.

Scenarios. Each pending call is issued by TWO routers over the same identity:
the gateway answers 202 twice with one approval, so the second router still
holds the pending call once the first has resumed. That is how the consumed
recovery path is reachable at all: `resume` drops its pending entry before
re-issuing, so a second `resume` on the same router never reaches the gateway.

    A gateway executed HTTP tool: pending, approve, resume, consumed resume
    B runtime local function:     pending, approve, resume, consumed resume
    C rejected approval:          denial, nothing ran
    D expired approval:           denial, nothing ran
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import uuid

from agenomic import Client
from agenomic.tools import ToolApprovalPending, ToolCallDenied, ToolExecutionError

BASE = os.environ["AGENOMIC_BASE_URL"]
TOOL_URL = os.environ["AGENOMIC_TOOL_URL"]
PROFILE = os.environ.get("AGENOMIC_ENV_PROFILE", "sdk-real")
TTL = int(os.environ.get("AGENOMIC_APPROVAL_TTL_SECS", "45"))
AGENT = "agent://sdkreal/python"
POLICY_ID = "sdk-real-python"
VERSION = os.environ.get("AGENOMIC_POLICY_VERSION", "1.0.0")

caller = Client(base_url=BASE, api_key=os.environ["AGENOMIC_API_KEY"])
reviewer = Client(base_url=BASE, api_key=os.environ["AGENOMIC_REVIEWER_KEY"])
admin = Client(base_url=BASE, api_key=os.environ["AGENOMIC_ADMIN_KEY"])

POLICY = f"""
policy_id: {POLICY_ID}
version: {VERSION}
schema_version: agenomic.policy/v1
status: draft
scope:
  tools: [crm.update_customer, crm.set_flag]
  agent_id: {AGENT}
  environment: production
default_decision: deny
rules:
  - rule_id: update-needs-review
    match:
      action_type: tool.call
      tool_id: crm.update_customer
      arguments: [{{ field: fields.credit_limit, op: exists }}]
    decision: require_approval
    obligations: [{{ kind: reviewer_distinct_from_principal }}]
    reason: credit limit changes need a human reviewer
  - rule_id: set-flag-needs-review
    match:
      action_type: tool.call
      tool_id: crm.set_flag
      arguments: [{{ field: flag, op: exists }}]
    decision: require_approval
    obligations: [{{ kind: reviewer_distinct_from_principal }}]
    reason: flag changes need a human reviewer
"""

CONFIG = f"""
schema_version: agenomic.tool_execution/v1
mode: live
default_mode: mock
on_unmatched: error
environment_profile: {PROFILE}
allowed_env: [URL]
limits: {{ max_live_calls: 20, max_concurrency: 2, timeout_ms: 15000 }}
safety: {{ live_writes: allow, require_approved_bindings: true, allow_implicit_fallback: false }}
recording: {{ enabled: false }}
protect:
  agent_id: {AGENT}
  environment: production
  mode: enforce
  policy_refs: [{POLICY_ID}@{VERSION}]
bindings:
  crm.update_customer:
    mode: live
    adapter: http
    transport: http
    endpoint: ${{env:URL}}/crm/update
    method: POST
    effect: reversible_write
    health_path: /health
  crm.set_flag:
    mode: live
    adapter: local
    transport: in_process
    function: crm.set_flag
    effect: reversible_write
"""

OBSERVED: list[dict[str, object]] = []
LOCAL_CALLS: list[dict[str, object]] = []


def note(step: str, **fields: object) -> None:
    entry = {"step": step, **fields}
    OBSERVED.append(entry)
    print(json.dumps(entry, default=str))


def check(label: str, ok: bool, detail: object = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail}")
    if not ok:
        note("assertion_failed", label=label, detail=detail)
        raise SystemExit(1)


def endpoint_count() -> int:
    with urllib.request.urlopen(f"{TOOL_URL}/__count") as response:
        return int(json.load(response)["count"])


def set_flag(**arguments: object) -> dict[str, object]:
    LOCAL_CALLS.append(dict(arguments))
    return {"flag_set": True, "call_number": len(LOCAL_CALLS)}


def settled(run_id: str, logical_call_id: str) -> list[dict[str, object]]:
    export = caller.tools.export(run_id)
    return [
        invocation
        for invocation in export.get("invocations", [])
        if invocation.get("logical_call_id") == logical_call_id
    ]


def http_status(error: Exception) -> object:
    return getattr(error, "status", None)


def pending_twice(router_a, router_b, tool: str, arguments: dict, call_id: str):
    """Both routers issue the identical identity; the gateway answers 202 twice."""
    first = second = None
    try:
        router_a.call(tool, arguments, logical_call_id=call_id)
    except ToolApprovalPending as pending:
        first = pending
    try:
        router_b.call(tool, arguments, logical_call_id=call_id)
    except ToolApprovalPending as pending:
        second = pending
    check(f"{call_id}: router A raised ToolApprovalPending", first is not None)
    check(f"{call_id}: router B raised ToolApprovalPending", second is not None)
    check(
        f"{call_id}: one approval for both routers",
        first.approval_id == second.approval_id,
        first.approval_id,
    )
    return first


def main() -> None:
    admin.protect.policies.register(POLICY)
    admin.protect.policies.release(POLICY_ID, VERSION)
    binding = admin.protect.bindings.create(POLICY_ID, VERSION, "agent", AGENT, mode="enforce")
    note("binding", id=binding["id"], status=binding["status"])

    run = caller.tools.create_run(
        name=f"protect-consumed-py-{uuid.uuid4().hex[:8]}", config_text=CONFIG
    )
    caller.tools.approve_run(run["id"], plan_hash=run["plan_hash"])
    caller.tools.start_run(run["id"])
    run_id = run["id"]
    note("run", id=run_id)

    locals_map = {"crm.set_flag": set_flag}
    router_a = caller.tools.router(run_id, local_functions=locals_map)
    router_b = caller.tools.router(run_id, local_functions=locals_map)

    before = endpoint_count()
    args = {"customer_id": "c-1", "fields": {"credit_limit": 5000}}
    pending = pending_twice(router_a, router_b, "crm.update_customer", args, "gw-1")
    check("A: the tool endpoint was not called", endpoint_count() == before)
    note("A.pending", approval_id=pending.approval_id, endpoint_calls=endpoint_count())

    try:
        caller.protect.approvals.decide(pending.approval_id, "approve", comment="self approval")
        check("A: self approval refused", False, "the caller approved its own call")
    except Exception as error:  # noqa: BLE001
        note("A.self_approve", refused=type(error).__name__, message=str(error)[:120])

    decided = reviewer.protect.approvals.decide(pending.approval_id, "approve", comment="reviewed")
    check("A: reviewer approved", decided["status"] == "approved", decided["status"])

    result = router_a.resume(pending, poll_interval=0.2, timeout=30)
    note("A.resume", result=result, endpoint_calls=endpoint_count())
    check("A: recovered result", result.get("updated") is True, result)
    check("A: exactly one endpoint call", endpoint_count() == before + 1, endpoint_count())
    check("A: exactly one settled invocation", len(settled(run_id, "gw-1")) == 1)

    status = caller.protect.approvals.get(pending.approval_id)["status"]
    check("A: approval is consumed", status == "consumed", status)

    outcome: dict[str, object]
    try:
        replayed = router_b.resume(pending.approval_id, poll_interval=0.2, timeout=30)
        outcome = {"kind": "recovered", "result": replayed}
    except ToolExecutionError as error:
        outcome = {
            "kind": "error",
            "code": error.code,
            "status": http_status(error),
            "message": str(error),
        }
    note("A.consumed_resume", **outcome, endpoint_calls=endpoint_count())
    check(
        "A: the consumed resume recovers the stored result on the gateway path",
        outcome["kind"] == "recovered",
        outcome,
    )
    if outcome["kind"] == "recovered":
        check(
            "A: the replayed result is the stored one",
            outcome["result"] == result,
            outcome["result"],
        )
    check("A: the consumed resume ran no second effect", endpoint_count() == before + 1)
    check("A: still exactly one settled invocation", len(settled(run_id, "gw-1")) == 1)

    router_c = caller.tools.router(run_id, local_functions=locals_map)
    router_d = caller.tools.router(run_id, local_functions=locals_map)
    pending = pending_twice(router_c, router_d, "crm.set_flag", {"flag": "vip"}, "loc-1")
    check("B: the local function did not run", len(LOCAL_CALLS) == 0, LOCAL_CALLS)
    note("B.pending", approval_id=pending.approval_id, local_calls=len(LOCAL_CALLS))

    reviewer.protect.approvals.decide(pending.approval_id, "approve", comment="reviewed")
    result = router_c.resume(pending, poll_interval=0.2, timeout=30)
    note("B.resume", result=result, local_calls=len(LOCAL_CALLS))
    check("B: recovered result", result.get("flag_set") is True, result)
    check("B: the local function ran exactly once", len(LOCAL_CALLS) == 1, LOCAL_CALLS)
    check("B: exactly one settled invocation", len(settled(run_id, "loc-1")) == 1)

    try:
        replayed = router_d.resume(pending.approval_id, poll_interval=0.2, timeout=30)
        outcome = {"kind": "recovered", "result": replayed}
    except ToolExecutionError as error:
        outcome = {
            "kind": "error",
            "code": error.code,
            "status": http_status(error),
            "message": str(error),
        }
    note("B.consumed_resume", **outcome, local_calls=len(LOCAL_CALLS))
    check(
        "B: the consumed resume raises conflict on the runtime local path",
        outcome["kind"] == "error" and outcome.get("code") == "conflict",
        outcome,
    )
    check("B: the local function still ran once", len(LOCAL_CALLS) == 1, LOCAL_CALLS)

    before = endpoint_count()
    router_e = caller.tools.router(run_id, local_functions=locals_map)
    try:
        router_e.call(
            "crm.update_customer",
            {"customer_id": "c-2", "fields": {"credit_limit": 9000}},
            logical_call_id="gw-2",
        )
        check("C: pending raised", False)
    except ToolApprovalPending as pending_c:
        pending = pending_c
    rejected = reviewer.protect.approvals.decide(pending.approval_id, "reject", comment="too high")
    note("C.rejected", status=rejected["status"])
    try:
        router_e.resume(pending, poll_interval=0.2, timeout=30)
        check("C: rejected raises a denial", False)
    except ToolCallDenied as denied:
        note("C.resume", code=denied.code, endpoint_calls=endpoint_count())
        check("C: denial carries the rejected status", denied.code == "rejected", denied.code)
    check("C: nothing ran", endpoint_count() == before, endpoint_count())

    router_f = caller.tools.router(run_id, local_functions=locals_map)
    try:
        router_f.call(
            "crm.update_customer",
            {"customer_id": "c-3", "fields": {"credit_limit": 1000}},
            logical_call_id="gw-3",
        )
        check("D: pending raised", False)
    except ToolApprovalPending as pending_d:
        pending = pending_d
    note("D.pending", approval_id=pending.approval_id, ttl_secs=TTL)
    started = time.monotonic()
    try:
        router_f.resume(pending, poll_interval=1.0, timeout=TTL + 60)
        check("D: expired raises a denial", False)
    except ToolCallDenied as denied:
        note("D.resume", code=denied.code, waited_secs=round(time.monotonic() - started, 1))
        check("D: denial carries the expired status", denied.code == "expired", denied.code)
    check("D: nothing ran", endpoint_count() == before, endpoint_count())

    caller.tools.complete_run(run_id)
    out = os.environ.get("AGENOMIC_OBSERVED_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(OBSERVED, handle, indent=2, default=str)
    print(f"\nprotect consumed recovery: python SDK, run {run_id}, {len(OBSERVED)} observations")


if __name__ == "__main__":
    sys.exit(main())
