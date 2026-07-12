"""Review · Monitor · Protect: one pass around the safety loop, offline.

With no ``base_url`` everything is buffered locally in the spec's snake_case
``agenomic.rmp/v0.1`` shapes; point the client at Agenomic Cloud to drive the
``/v1/rmp|review|monitor|protect`` endpoints instead.

    python examples/09_rmp.py
"""

from __future__ import annotations

from agenomic import Client


def main() -> None:
    # base_url=... switches to cloud mode (no silent insecure fallback).
    client = Client()
    agent = "agent://treans/claims-agent"

    # Umbrella session tying the Review + Monitor + Protect pass together.
    rmp = client.rmp.start(agent=agent, release_id="release_123", ledger=True)
    print("rmp session:", rmp["session_id"])

    # Review: register a scenario and run a pre-release pass.
    scenario = client.review.add_scenario({"agent_id": agent, "title": "refund over limit"})
    outcome = client.review.run(agent=agent)
    print("review:", outcome["result"], "scenarios:", outcome["scenario_ids"])
    assert scenario["scenario_id"] in outcome["scenario_ids"]

    # Monitor: observe the production run (events are stamped snake_case dicts).
    monitor = client.monitor.start(agent=agent, release_id="release_123")
    monitor.event({"type": "tool.call.completed", "tool": {"name": "claims_db.lookup"}})
    monitor.event({"type": "loop.detected"})
    monitor.stop()
    findings = client.monitor.findings(monitor.session_id)
    print("monitor findings:", [f["kind"] for f in findings])

    # Protect: alerts, an ordered remediation plan, and routing.
    print("alerts:", client.protect.alerts(monitor.session_id))
    plan = client.protect.action_plan("alr_demo", session_id=monitor.session_id)
    print("action plan phases:", [step["phase"] for step in plan["steps"]])
    client.protect.notify("alr_demo", session_id=monitor.session_id)

    # Close the loop: a human approves the enrichment proposal back into Review.
    proposal_id = plan["scenario_proposal_ids"][0]
    approved = client.review.approve_scenario_enrichment(
        proposal_id, session_id=monitor.session_id, reviewer="ops@treans.example"
    )
    print("proposal:", approved["proposal_id"], "->", approved["status"])

    # And report on the whole pass.
    report = client.rmp.report(rmp["session_id"])
    print("report:", report["report_version"], report["result"])


if __name__ == "__main__":
    main()
