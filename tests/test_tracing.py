import json

import httpx
import pytest

from agentlock import AgentLockClient, HTTPTraceExporter, Redactor, TraceRecorder, trace_agent_run


def test_trace_agent_run_emits_redacted_trace() -> None:
    client = AgentLockClient()
    redactor = Redactor(redact=["customer.email", "customer.phone"], mode="mask")

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
        assert trace is not None
        trace.add_tool_call(
            name="policy_lookup",
            input={"policy_id": payload["policy_id"]},
            output={"active": True},
        )
        return {"customer": payload["customer"], "decision": "approve"}

    result = handle_claim(
        {
            "policy_id": "pol_123",
            "customer": {
                "email": "casey@example.com",
                "phone": "+1-555-0100",
            },
        }
    )

    assert result["decision"] == "approve"
    assert len(client.local_traces) == 1

    trace = client.local_traces[0]
    assert trace.metadata["success"] is True
    assert isinstance(trace.metadata["input_hash"], str)
    assert isinstance(trace.metadata["output_hash"], str)
    assert trace.input["customer"]["email"] == "***REDACTED***"
    assert trace.final_output["customer"]["phone"] == "***REDACTED***"
    assert trace.tool_calls[0].name == "policy_lookup"


def test_trace_agent_run_records_failures() -> None:
    client = AgentLockClient()

    @trace_agent_run(agent_id="claims-agent", release="dev", client=client)
    def fail_claim(_: dict[str, object]) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fail_claim({"claim_id": "clm_404"})

    trace = client.local_traces[0]
    assert trace.metadata["success"] is False
    assert trace.run_completed is not None
    assert trace.run_completed.error == "ValueError"


def test_client_emits_trace_over_http() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(202, json={"accepted": True})

    client = AgentLockClient(
        api_key="agentlock_test_key",
        endpoint="https://example.test/v1/traces",
    )
    client._http_exporter = HTTPTraceExporter(
        endpoint=client.endpoint or "",
        api_key=client.api_key,
        transport=httpx.MockTransport(handler),
    )

    envelope = TraceRecorder(
        agent_id="claims-agent",
        release="dev",
        input={"claim_id": "clm_123"},
    ).complete(final_output={"decision": "approve"})

    response = client.emit_trace(envelope)

    assert response.status_code == 202
    assert captured["authorization"] == "Bearer agentlock_test_key"
    assert captured["payload"]["agent_id"] == "claims-agent"
