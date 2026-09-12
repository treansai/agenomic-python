"""Hybrid replay: one real MCP tool, one mocked scenario, one recorded fixture.

Runs against a live Agenomic Cloud. Set:

    AGENOMIC_BASE_URL=http://localhost:8080
    AGENOMIC_API_KEY=agm_...
    CRM_MCP_URL=http://crm-staging.internal/mcp   (value stored write-only)
    CRM_API_TOKEN=...                              (value stored write-only)

The agent loop below is a stand-in for your own runtime: only the tool calls
are routed through Agenomic; the model strategy stays yours.
"""

from __future__ import annotations

import os
import sys

from agenomic import Client
from agenomic.tools import ToolCallError

CONFIG = """
schema_version: agenomic.tool_execution/v1
mode: hybrid
default_mode: mock
on_unmatched: error
environment_profile: replay-staging
allowed_env: [CRM_MCP_URL, CRM_API_TOKEN]
limits: { max_live_calls: 20, max_concurrency: 4, timeout_ms: 15000 }
safety: { live_writes: deny, require_approved_bindings: true, allow_implicit_fallback: false }
mock_engine:
  strict: true
  seed: 42
  clock: { mode: virtual, start: "2026-01-01T00:00:00Z" }
  external_tool_network: deny
recording: { enabled: false, capture_policy_ref: replay-safe-v1 }
bindings:
  crm.get_customer:
    mode: live
    contract_ref: crm.get_customer@1
    adapter: mcp
    transport: streamable_http
    endpoint: "${env:CRM_MCP_URL}"
    headers: { Authorization: "Bearer ${env:CRM_API_TOKEN}" }
    policy_ref: staging-crm-readonly
    effect: read
  email.send:
    mode: mock
    contract_ref: email.send@1
    strategy: scenario
    scenario_ref: email-delivery@1
  documents.extract:
    mode: mock
    contract_ref: documents.extract@2
    strategy: recorded
    fixture_set_ref: document-extraction-fixtures@3
"""


def main() -> int:
    base_url = os.environ.get("AGENOMIC_BASE_URL")
    api_key = os.environ.get("AGENOMIC_API_KEY")
    if not base_url or not api_key:
        print("set AGENOMIC_BASE_URL and AGENOMIC_API_KEY to run this example")
        return 0
    client = Client(api_key=api_key, base_url=base_url)
    tools = client.tools

    profiles = {p["name"]: p for p in tools.list_profiles()}
    profile = (
        profiles.get("replay-staging")
        or tools.create_profile(
            name="replay-staging",
            environment="staging",
            allowed_env=["CRM_MCP_URL", "CRM_API_TOKEN"],
        )["profile"]
    )
    for name in ("CRM_MCP_URL", "CRM_API_TOKEN"):
        if os.environ.get(name):
            tools.set_variable(profile["id"], name, os.environ[name])
    print("variables:", [(v["name"], v["available"]) for v in tools.variable_status(profile["id"])])

    plan = tools.preflight(config_text=CONFIG, repetitions=3)
    print("live tools:", plan["plan"]["live_tools"], "mock tools:", plan["plan"]["mock_tools"])
    if not plan["runnable"]:
        print(
            "plan not runnable:",
            plan["plan"]["errors"],
            [t["errors"] for t in plan["plan"]["tools"]],
        )
        return 1

    run = tools.create_run(name="hybrid-demo", config_text=CONFIG, repetitions=3)
    if run["status"] == "planned":
        run = tools.approve_run(run["id"], plan_hash=run["plan_hash"])
    run = tools.start_run(run["id"])

    for repetition in range(1, 4):
        router = tools.router(run["id"], repetition=repetition)
        try:
            customer = router.call("crm.get_customer", {"id": "c_1"})
            router.call("email.send", {"to": "ops@example.test", "subject": f"hello {customer}"})
            router.call("documents.extract", {"doc": "invoice-1"})
        except ToolCallError as error:
            print(f"repetition {repetition}: {error}")
        print(f"repetition {repetition}:", router.summary())

    tools.complete_run(run["id"])
    report = tools.report(run["id"])["report"]
    print("by source:", report["calls_by_source"])
    print("warning:", report["warning"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
