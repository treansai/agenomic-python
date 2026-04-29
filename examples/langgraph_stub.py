import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlock import AgentLockClient, instrument_langgraph


class FakeGraph:
    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "echo": payload}


if __name__ == "__main__":
    client = AgentLockClient()
    graph = instrument_langgraph(
        FakeGraph(),
        client=client,
        agent_id="langgraph-demo",
        release="dev",
    )

    result = graph.invoke({"message": "hello"})
    print(result)
    print(client.local_traces[0].model_dump_json(indent=2))
