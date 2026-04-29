import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


if __name__ == "__main__":
    from agentlock import AgentLockClient, TraceRecorder

    client = AgentLockClient()

    traces = [
        TraceRecorder(
            agent_id="claims-agent",
            release="dev",
            input={"claim_id": "clm_1"},
        ).complete(final_output={"decision": "approve"}),
        TraceRecorder(
            agent_id="claims-agent",
            release="dev",
            input={"claim_id": "clm_2"},
        ).complete(final_output={"decision": "deny"}),
    ]

    output_path = Path(__file__).with_name("traces.jsonl")
    client.export_jsonl(str(output_path), traces)
    print(output_path)
