"""Tests for the Review · Monitor · Protect (RMP) SDK surface."""

from __future__ import annotations

import json

import pytest

from agenomic import Client
from agenomic.rmp import (
    ENRICHMENT_VERSION,
    RMP_REPORT_VERSION,
    RMP_SPEC_VERSION,
    MonitorSession,
)

# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------


def test_local_rmp_session_lifecycle() -> None:
    client = Client()  # no base_url => local mode
    assert not client.is_cloud

    session = client.rmp.start(
        agent="agent://treans/claims-agent",
        release_id="release_123",
        environment="production",
        ledger=True,
        genome_hash="blake3:" + "0" * 64,
    )
    session_id = session["session_id"]
    assert session_id.startswith("rmp_")
    assert session["spec_version"] == RMP_SPEC_VERSION
    assert session["agent_id"] == "agent://treans/claims-agent"
    assert session["release_id"] == "release_123"
    assert session["ledger_enabled"] is True
    assert session["status"] == "active"

    assert client.rmp.get(session_id) == session
    assert session in client.rmp.list()

    report = client.rmp.report(session_id)
    assert report["report_version"] == RMP_REPORT_VERSION
    assert report["session_id"] == session_id
    assert report["result"] == "pass"
    assert report["findings"] == []


def test_local_rmp_get_unknown_session_raises() -> None:
    client = Client()
    with pytest.raises(KeyError):
        client.rmp.get("rmp_missing")


def test_local_rmp_start_reuses_active_session_per_agent_and_environment() -> None:
    client = Client()

    first = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
    second = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
    assert first["session_id"] == second["session_id"]
    assert len(client.rmp.list()) == 1

    other_env = client.rmp.start(agent="agent://treans/claims-agent", environment="production")
    assert other_env["session_id"] != first["session_id"]
    assert len(client.rmp.list()) == 2


def test_local_rmp_concurrent_starts_share_one_session() -> None:
    import threading

    client = Client()
    workers = 8
    barrier = threading.Barrier(workers)
    ids: list[str] = []
    lock = threading.Lock()

    def start() -> None:
        barrier.wait()
        session = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
        with lock:
            ids.append(session["session_id"])

    threads = [threading.Thread(target=start) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ids) == workers
    assert len(set(ids)) == 1
    assert len(client.rmp.list()) == 1


def test_local_rmp_stop_twice_keeps_the_completion_timestamp() -> None:
    client = Client()
    session = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
    first = dict(client.rmp.stop(session["session_id"]))
    second = client.rmp.stop(session["session_id"])
    assert second["status"] == "completed"
    assert second["ended_at"] == first["ended_at"]
    assert second == first


def test_local_rmp_stop_frees_the_slot_for_a_new_session() -> None:
    client = Client()

    first = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
    stopped = client.rmp.stop(first["session_id"])
    assert stopped["status"] == "completed"
    assert client.rmp.get(first["session_id"])["status"] == "completed"

    second = client.rmp.start(agent="agent://treans/claims-agent", environment="development")
    assert second["session_id"] != first["session_id"]
    assert second["status"] == "active"


def test_local_monitor_buffers_and_stamps_events() -> None:
    client = Client()
    session = client.monitor.start(agent="agent://treans/claims-agent")
    assert isinstance(session, MonitorSession)
    assert session.session_id.startswith("mon_")

    first = session.event({"type": "tool.call.completed", "tool": {"name": "claims_db.lookup"}})
    second = client.monitor.event(session.session_id, {"type": "loop.detected"})

    events = session.events
    assert len(events) == 2
    assert [e["sequence_number"] for e in events] == [0, 1]
    assert all(e["spec_version"] == RMP_SPEC_VERSION for e in events)
    assert all(e["agent_id"] == "agent://treans/claims-agent" for e in events)
    assert all(e["session_id"] == session.session_id for e in events)
    assert all(e["timestamp"].endswith("Z") for e in events)
    assert first["event_id"] != second["event_id"]
    assert first["tool"] == {"name": "claims_db.lookup"}

    # buffered detection events project into local findings
    findings = client.monitor.findings(session.session_id)
    assert len(findings) == 1
    assert findings[0]["kind"] == "loop"
    assert findings[0]["loop_stage"] == "monitor"
    assert findings[0]["evidence_refs"] == [second["event_id"]]


def test_local_monitor_event_requires_type() -> None:
    client = Client()
    session = client.monitor.start(agent="agent://acme/a")
    with pytest.raises(ValueError):
        session.event({"tool": {"name": "x"}})


def test_local_monitor_stop_is_idempotent_and_refuses_events() -> None:
    client = Client()
    session = client.monitor.start(agent="agent://acme/a")
    session.event({"type": "agent.started"})
    session.stop()
    session.stop()  # idempotent
    with pytest.raises(RuntimeError):
        session.event({"type": "agent.completed"})


def test_local_monitor_unknown_session_raises() -> None:
    client = Client()
    with pytest.raises(KeyError):
        client.monitor.event("mon_missing", {"type": "agent.started"})
    with pytest.raises(KeyError):
        client.monitor.findings("mon_missing")


def test_local_review_scenarios_add_and_list() -> None:
    client = Client()
    scenario = client.review.add_scenario(
        {"agent_id": "agent://acme/a", "title": "refund over limit"}
    )
    assert scenario["scenario_id"].startswith("sc_")
    assert scenario["spec_version"] == RMP_SPEC_VERSION

    other = client.review.add_scenario({"agent_id": "agent://acme/b", "title": "other"})
    assert client.review.list_scenarios() == [scenario, other]
    assert client.review.list_scenarios(agent="agent://acme/a") == [scenario]


def test_local_review_run_returns_stub_outcome() -> None:
    client = Client()
    scenario = client.review.add_scenario({"agent_id": "agent://acme/a", "title": "t"})
    outcome = client.review.run(agent="agent://acme/a")
    assert outcome["session_id"].startswith("rev_")
    assert outcome["result"] == "pass"
    assert outcome["findings"] == []
    assert outcome["scenario_ids"] == [scenario["scenario_id"]]


def test_local_protect_alerts_empty_and_action_plan_closes_loop() -> None:
    client = Client()
    assert client.protect.alerts("mon_x") == []
    assert client.protect.recommendations("mon_x") == []

    plan = client.protect.action_plan("alr_1", session_id="mon_x")
    assert plan["plan_id"].startswith("apl_")
    assert plan["alert_id"] == "alr_1"
    assert [s["phase"] for s in plan["steps"]] == ["investigate", "mitigate", "verify"]

    # the plan's enrichment proposal is buffered on client.review …
    [proposal_id] = plan["scenario_proposal_ids"]
    [buffered] = client.review.proposals
    assert buffered["proposal_id"] == proposal_id
    assert buffered["spec_version"] == ENRICHMENT_VERSION
    assert buffered["status"] == "proposed"

    # … and approving it closes the Protect → Review feedback loop
    approved = client.review.approve_scenario_enrichment(
        proposal_id, session_id="mon_x", reviewer="ops@acme.test"
    )
    assert approved["status"] == "approved"
    assert approved["reviewer"] == "ops@acme.test"
    assert client.review.proposals[0]["status"] == "approved"

    # the approved scenario is materialized into the Review suite, so
    # subsequent list_scenarios() / run() calls include the enrichment
    scenario = approved["proposed_scenario"]
    suite = client.review.list_scenarios()
    assert [s["scenario_id"] for s in suite] == [scenario["scenario_id"]]
    assert suite[0]["source"] == "protect_derived"

    # approving again does not duplicate the scenario
    client.review.approve_scenario_enrichment(proposal_id)
    assert len(client.review.list_scenarios()) == 1


def test_local_approve_unknown_proposal_raises() -> None:
    client = Client()
    with pytest.raises(KeyError):
        client.review.approve_scenario_enrichment("sep_missing")


def test_local_protect_notify_returns_stub_route() -> None:
    client = Client()
    record = client.protect.notify("alr_1", session_id="mon_x")
    assert record["alert_id"] == "alr_1"
    assert record["routes"][0]["channel"] == "stdout"


def test_monitor_session_context_manager_stops() -> None:
    client = Client()
    with client.monitor.start(agent="agent://acme/a") as session:
        session.event({"type": "agent.started"})
    assert session._stopped is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# Cloud mode (mocked, no network)
# ---------------------------------------------------------------------------


def test_cloud_rmp_start_posts_session(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/sessions",
        json={"session": {"session_id": "rmp_1", "agent_id": "agent://treans/claims-agent"}},
    )
    client = Client(api_key="key_123", base_url="https://api.test")
    session = client.rmp.start(agent="agent://treans/claims-agent", release_id="r1", ledger=True)
    assert session["session_id"] == "rmp_1"

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer key_123"
    body = json.loads(request.content)
    assert body == {
        "spec_version": RMP_SPEC_VERSION,
        "agent_id": "agent://treans/claims-agent",
        "environment": "production",
        "ledger_enabled": True,
        "release_id": "r1",
    }


def test_cloud_rmp_start_without_session_id_raises(httpx_mock) -> None:
    from agenomic.exceptions import CloudError

    httpx_mock.add_response(
        method="POST", url="https://api.test/v1/rmp/sessions", json={"session": {}}
    )
    client = Client(base_url="https://api.test")
    with pytest.raises(CloudError):
        client.rmp.start(agent="agent://a/b")


def test_cloud_monitor_event_posts_stamped_event(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/monitor/sessions",
        json={"session": {"session_id": "mon_1"}},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/monitor/sessions/mon_1/events",
        json={},
    )
    client = Client(base_url="https://api.test")
    session = client.monitor.start(agent="agent://treans/claims-agent")
    assert session.session_id == "mon_1"

    session.event({"type": "tool.call.completed", "tool": {"name": "claims_db.lookup"}})

    body = json.loads(httpx_mock.get_requests()[1].content)
    assert body["type"] == "tool.call.completed"
    assert body["tool"] == {"name": "claims_db.lookup"}
    assert body["session_id"] == "mon_1"
    assert body["agent_id"] == "agent://treans/claims-agent"
    assert body["sequence_number"] == 0
    assert body["spec_version"] == RMP_SPEC_VERSION
    assert body["event_id"]
    assert body["timestamp"].endswith("Z")
    # cloud mode does not buffer locally
    assert session.events == []


def test_cloud_protect_alerts_gets_by_session(httpx_mock) -> None:
    alert = {"alert_id": "alr_1", "severity": "high", "status": "open"}
    httpx_mock.add_response(
        method="GET",
        url="https://api.test/v1/protect/alerts?session_id=mon_1",
        json={"alerts": [alert]},
    )
    client = Client(base_url="https://api.test")
    assert client.protect.alerts("mon_1") == [alert]


def test_cloud_protect_action_plan_posts(httpx_mock) -> None:
    plan = {"plan_id": "apl_1", "alert_id": "alr_1", "steps": []}
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/protect/alerts/alr_1/action-plan",
        json={"action_plan": plan},
    )
    client = Client(base_url="https://api.test")
    assert client.protect.action_plan("alr_1", session_id="mon_1") == plan

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == {"session_id": "mon_1"}


def test_cloud_review_approve_scenario_enrichment_posts(httpx_mock) -> None:
    proposal = {"proposal_id": "sep_1", "status": "approved"}
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/review/proposals/sep_1/approve",
        json={"proposal": proposal},
    )
    client = Client(base_url="https://api.test")
    approved = client.review.approve_scenario_enrichment(
        "sep_1", session_id="mon_1", reviewer="ops@acme.test"
    )
    assert approved == proposal

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == {"session_id": "mon_1", "reviewer": "ops@acme.test"}


def test_cloud_error_is_wrapped(httpx_mock) -> None:
    from agenomic.exceptions import CloudError

    httpx_mock.add_response(method="POST", url="https://api.test/v1/rmp/sessions", status_code=500)
    client = Client(base_url="https://api.test")
    with pytest.raises(CloudError):
        client.rmp.start(agent="agent://a/b")


def test_cloud_rmp_stop_posts_to_session_stop_route(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.test/v1/rmp/sessions/rmp_1/stop",
        json={"session": {"session_id": "rmp_1", "status": "completed"}},
    )
    client = Client(api_key="key_123", base_url="https://api.test")
    session = client.rmp.stop("rmp_1")
    assert session["status"] == "completed"

    request = httpx_mock.get_requests()[0]
    assert request.url == "https://api.test/v1/rmp/sessions/rmp_1/stop"
    assert request.headers["Authorization"] == "Bearer key_123"
