# Review · Monitor · Protect (RMP)

RMP is Agenomic's continuous safety loop: **Review** an agent before
release, **Monitor** it in production, **Protect** operators with alerts and
remediation plans, and feed what Protect learns back into Review as scenario
enrichment proposals.

Like the rest of the SDK it is **local-first**: with no `base_url` sessions,
scenarios and proposals are buffered in memory (detection runs in
`agenomic-cli` or Agenomic Cloud). Pass `base_url` to drive the cloud
`/v1/rmp|review|monitor|protect` endpoints — there is no silent fallback.
All shapes are snake_case dicts stamped with `spec_version`
`agenomic.rmp/v0.1`, identical to the Rust `agenomic-rmp` engines.

## The umbrella session

```python
from agenomic import Client

client = Client()  # or Client(api_key="...", base_url="https://cloud.example.com")

rmp = client.rmp.start(
    agent="agent://acme/claims-agent",
    release_id="release_123",
    environment="production",
    ledger=True,
)

client.rmp.get(rmp["session_id"])
client.rmp.list()
report = client.rmp.report(rmp["session_id"])   # report_version agenomic.rmp.report/v0.1
```

## Review — before release

```python
scenario = client.review.add_scenario(
    {"agent_id": "agent://acme/claims-agent", "title": "refund over limit"}
)
client.review.list_scenarios(agent="agent://acme/claims-agent")

outcome = client.review.run(agent="agent://acme/claims-agent")
assert outcome["result"] in ("pass", "warn", "fail")
```

## Monitor — in production

```python
session = client.monitor.start(agent="agent://acme/claims-agent")

# Events are plain snake_case dicts with a required `type`; the session
# stamps event_id, timestamp, sequence_number, session_id and agent_id.
session.event({"type": "tool.call.completed", "tool": {"name": "claims_db.lookup"}})
session.event({"type": "loop.detected"})
session.stop()  # idempotent

client.monitor.findings(session.session_id)
```

## Protect — alerts, plans, recommendations

```python
alerts = client.protect.alerts(session.session_id)
plan = client.protect.action_plan("alr_1", session_id=session.session_id)
client.protect.recommendations(session.session_id)
client.protect.notify("alr_1", session_id=session.session_id)
```

## Closing the loop — enrichment approval

Action plans carry scenario enrichment proposals. Proposals are **never
applied automatically** — a human approves them back into the Review
scenario suite:

```python
proposal_id = plan["scenario_proposal_ids"][0]
client.review.approve_scenario_enrichment(
    proposal_id,
    session_id=session.session_id,
    reviewer="ops@acme.example",
)
```

See `examples/09_rmp.py` for a runnable offline walkthrough.
