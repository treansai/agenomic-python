"""Correction loop client: every route, bodies, typed models and the cloud-only rule."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agenomic import Client
from agenomic.correction import CorrectionSettings, PolicyChange
from agenomic.tools import ToolExecutionError

BASE = "https://api.test"
API = f"{BASE}/v1/correction"
INCIDENT = "11111111-0000-4000-8000-000000000001"
CANDIDATE = "22222222-0000-4000-8000-000000000002"
REPORT = "33333333-0000-4000-8000-000000000003"
DIGEST = "sha256:abc"


def _cloud() -> Client:
    return Client(api_key="key", base_url=BASE)


def _incident(status: str = "open") -> dict[str, Any]:
    return {
        "id": INCIDENT,
        "org_id": "o",
        "source": "protect_decision",
        "source_ref": "dec_1",
        "agent_id": "agent://support",
        "tool": "payments.refund",
        "outcome": "deny",
        "effective_mode": "enforce",
        "severity": "high",
        "title": "refund denied",
        "reason_codes": ["family_default"],
        "evidence_refs": ["protect_decision:dec_1"],
        "dedup_key": "k",
        "occurrence_count": 1,
        "status": status,
        "opened_by": "user",
        "suggestion": None,
        "first_seen_at": "2026-10-15T10:00:00Z",
        "last_seen_at": "2026-10-15T10:00:00Z",
        "updated_at": "2026-10-15T10:00:00Z",
    }


def _candidate(status: str = "proposed") -> dict[str, Any]:
    return {
        "id": CANDIDATE,
        "org_id": "o",
        "incident_id": INCIDENT,
        "kind": "policy_change",
        "title": "tighten refunds",
        "rationale": "",
        "payload": {
            "policy_id": "refunds",
            "candidate_version": "2",
            "scope_kind": "org",
            "scope_ref": "",
            "mode": "shadow",
        },
        "content_digest": DIGEST,
        "scenario_digest": "sha256:def",
        "proposal_id": "prop_1",
        "claims_weakening": False,
        "weakening": "none",
        "weakening_reasons": [],
        "weakening_acknowledged": False,
        "status": status,
        "created_at": "2026-10-15T10:00:00Z",
        "updated_at": "2026-10-15T10:00:00Z",
    }


def _report(verdict: str = "passed") -> dict[str, Any]:
    return {
        "id": REPORT,
        "org_id": "o",
        "candidate_id": CANDIDATE,
        "content_digest": DIGEST,
        "scenario_digest": "sha256:def",
        "bindings_digest": "sha256:ghi",
        "report_digest": "sha256:jkl",
        "verdict": verdict,
        "report": {"summary": {"fixed": 1}},
        "created_at": "2026-10-15T10:00:00Z",
    }


def _sent(httpx_mock: Any) -> dict[str, Any]:
    (request,) = httpx_mock.get_requests()
    return json.loads(request.content) if request.content else {}


def test_cloud_required() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        Client().correction.candidates.get(CANDIDATE)
    assert exc.value.code == "cloud_required"


def test_open_incident_created_and_joined(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/incidents",
        status_code=201,
        json={"incident": _incident(), "created": True},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/incidents",
        status_code=200,
        json={"incident": _incident(), "created": False},
    )
    incidents = _cloud().correction.incidents
    assert incidents.open("protect_decision", "dec_1").created is True
    assert incidents.open("protect_decision", "dec_1").created is False
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "source": "protect_decision",
        "source_ref": "dec_1",
    }


def test_list_incidents_query(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/incidents?status=open&agent_id=agent%3A%2F%2Fsupport&limit=5",
        json={"incidents": [_incident()]},
    )
    found = _cloud().correction.incidents.list(status="open", agent_id="agent://support", limit=5)
    assert [i.severity for i in found] == ["high"]


def test_incident_get_scenario_dismiss(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/incidents/{INCIDENT}", json=_incident())
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/incidents/{INCIDENT}/scenario",
        json=_incident("scenario_ready"),
    )
    httpx_mock.add_response(
        method="POST", url=f"{API}/incidents/{INCIDENT}/dismiss", json=_incident("dismissed")
    )
    incidents = _cloud().correction.incidents
    assert incidents.get(INCIDENT).status == "open"
    assert incidents.to_scenario(INCIDENT).status == "scenario_ready"
    assert incidents.dismiss(INCIDENT, reason="duplicate").status == "dismissed"
    assert json.loads(httpx_mock.get_requests()[2].content) == {"reason": "duplicate"}


def test_create_candidate_serializes_a_typed_payload(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/candidates", status_code=201, json=_candidate()
    )
    candidate = _cloud().correction.candidates.create(
        INCIDENT,
        kind="policy_change",
        title="tighten refunds",
        payload=PolicyChange(
            policy_id="refunds", candidate_version="2", scope_kind="org", mode="shadow"
        ),
    )
    assert candidate.status == "proposed"
    assert _sent(httpx_mock) == {
        "incident_id": INCIDENT,
        "kind": "policy_change",
        "title": "tighten refunds",
        "rationale": "",
        "payload": {
            "policy_id": "refunds",
            "candidate_version": "2",
            "scope_kind": "org",
            "mode": "shadow",
        },
        "claims_weakening": False,
    }


def test_list_get_transitions_replay(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/candidates?status=verified&incident_id={INCIDENT}",
        json={"candidates": [_candidate("verified")]},
    )
    httpx_mock.add_response(method="GET", url=f"{API}/candidates/{CANDIDATE}", json=_candidate())
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/candidates/{CANDIDATE}/transitions",
        json={
            "transitions": [
                {
                    "id": "t1",
                    "org_id": "o",
                    "entity": "candidate",
                    "entity_id": CANDIDATE,
                    "from_status": "proposed",
                    "to_status": "proposed",
                    "actor_role": "maintainer",
                    "acknowledged_weakening": False,
                    "evidence": {"verdict": "failed"},
                    "created_at": "2026-10-15T10:00:00Z",
                }
            ]
        },
    )
    httpx_mock.add_response(
        method="GET", url=f"{API}/candidates/{CANDIDATE}/replay", json=_report("failed")
    )
    candidates = _cloud().correction.candidates
    assert len(candidates.list(status="verified", incident_id=INCIDENT)) == 1
    assert candidates.get(CANDIDATE).weakening == "none"
    assert candidates.transitions(CANDIDATE)[0].from_status == "proposed"
    assert candidates.replay(CANDIDATE).verdict == "failed"


@pytest.mark.parametrize(
    ("method", "step", "kwargs", "expected_body", "to_status"),
    [
        ("verify", "verify", {"content_digest": DIGEST}, {"content_digest": DIGEST}, None),
        (
            "review",
            "review",
            {"content_digest": DIGEST, "report_id": REPORT, "note": "read it"},
            {"content_digest": DIGEST, "report_id": REPORT, "note": "read it"},
            "reviewed",
        ),
        (
            "approve",
            "approve",
            {"content_digest": DIGEST, "report_id": REPORT, "acknowledge_weakening": True},
            {"content_digest": DIGEST, "report_id": REPORT, "acknowledge_weakening": True},
            "approved",
        ),
        ("reject", "reject", {"reason": "no"}, {"reason": "no"}, "rejected"),
        (
            "promote",
            "promote",
            {"content_digest": DIGEST},
            {"content_digest": DIGEST},
            "staged",
        ),
        ("enforce", "enforce", {"content_digest": DIGEST}, {"content_digest": DIGEST}, "promoted"),
        ("rollback", "rollback", {"reason": "bad"}, {"reason": "bad"}, "rolled_back"),
    ],
)
def test_every_transition(
    httpx_mock: Any,
    method: str,
    step: str,
    kwargs: dict[str, Any],
    expected_body: dict[str, Any],
    to_status: str | None,
) -> None:
    body = (
        {"candidate": _candidate("verified"), "report": _report()}
        if method == "verify"
        else _candidate(to_status or "proposed")
    )
    httpx_mock.add_response(method="POST", url=f"{API}/candidates/{CANDIDATE}/{step}", json=body)
    result = getattr(_cloud().correction.candidates, method)(CANDIDATE, **kwargs)
    assert _sent(httpx_mock) == expected_body
    if method == "verify":
        assert result.candidate.status == "verified"
        assert result.report.verdict == "passed"
    else:
        assert result.status == to_status


def test_failed_verification_is_returned_not_raised(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/candidates/{CANDIDATE}/verify",
        json={"candidate": _candidate("proposed"), "report": _report("inconclusive")},
    )
    outcome = _cloud().correction.candidates.verify(CANDIDATE, content_digest=DIGEST)
    assert outcome.candidate.status == "proposed"
    assert outcome.report.verdict == "inconclusive"


def test_refusals_carry_the_server_code(httpx_mock: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/candidates/{CANDIDATE}/promote",
        status_code=403,
        json={"error": {"code": "forbidden", "message": "owner only"}},
    )
    with pytest.raises(ToolExecutionError) as exc:
        _cloud().correction.candidates.promote(CANDIDATE, content_digest=DIGEST)
    assert (exc.value.code, exc.value.status) == ("forbidden", 403)


def test_settings_and_sweep(httpx_mock: Any) -> None:
    settings = {
        "trigger_enabled": True,
        "deny_threshold": 3,
        "window_secs": 600,
        "max_incidents_per_hour": 2,
        "min_shadow_secs": 0,
    }
    httpx_mock.add_response(
        method="GET", url=f"{API}/settings", json=CorrectionSettings().model_dump()
    )
    httpx_mock.add_response(method="PUT", url=f"{API}/settings", json=settings)
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/triggers/sweep",
        json={"enabled": True, "groups_over_threshold": 1, "opened": [INCIDENT], "rate_limited": 0},
    )
    correction = _cloud().correction
    assert correction.settings().trigger_enabled is False
    updated = correction.put_settings(CorrectionSettings(**settings))
    assert updated.min_shadow_secs == 0
    assert json.loads(httpx_mock.get_requests()[1].content) == settings
    assert correction.sweep().opened == [INCIDENT]


async def test_async_path(httpx_mock: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/incidents", json={"incidents": []})
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/candidates/{CANDIDATE}/rollback",
        json=_candidate("rolled_back"),
    )
    correction = _cloud().correction
    assert await correction.incidents.alist() == []
    rolled = await correction.candidates.arollback(CANDIDATE, reason="regression")
    assert rolled.status == "rolled_back"
