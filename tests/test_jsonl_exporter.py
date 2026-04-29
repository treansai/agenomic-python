import json
from pathlib import Path

from agentlock import AgentLockClient, TraceRecorder


def test_export_jsonl_writes_valid_json_lines(tmp_path: Path) -> None:
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

    output_path = tmp_path / "traces.jsonl"
    result = client.export_jsonl(str(output_path), traces)

    assert result == str(output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    documents = [json.loads(line) for line in lines]
    assert documents[0]["agent_id"] == "claims-agent"
    assert documents[1]["final_output"]["decision"] == "deny"
