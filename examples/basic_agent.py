import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlock import AgentLockClient, Redactor, TraceRecorder, trace_agent_run

client = AgentLockClient()
redactor = Redactor(
    redact=["customer.email", "customer.phone"],
    mode="mask",
)


@trace_agent_run(
    agent_id="claims-agent",
    release="dev",
    client=client,
    redactor=redactor,
)
def handle_claim(
    payload: dict[str, object],
    trace: TraceRecorder | None = None,
) -> dict[str, object]:
    if trace is not None:
        trace.add_model_call(
            name="local-review-model",
            provider="example",
            request={"task": "review claim"},
            response={"risk_score": 0.08},
        )
        trace.add_tool_call(
            name="policy_lookup",
            input={"policy_id": payload["policy_id"]},
            output={"active": True, "tier": "gold"},
        )

    return {
        "claim_id": payload["claim_id"],
        "decision": "approve",
        "reason": "policy active and risk score below threshold",
        "customer": payload["customer"],
    }


if __name__ == "__main__":
    result = handle_claim(
        {
            "claim_id": "clm_123",
            "policy_id": "pol_123",
            "customer": {
                "email": "casey@example.com",
                "phone": "+1-555-0100",
            },
        }
    )
    print(result)
    print(client.local_traces[0].model_dump_json(indent=2))
