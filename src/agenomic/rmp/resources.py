"""Review · Monitor · Protect (RMP) — the continuous safety loop.

``client.rmp.start(...)`` opens an umbrella session tying one Review +
Monitor + Protect pass together for a single agent/release/environment;
``client.review`` / ``client.monitor`` / ``client.protect`` drive the
individual loop stages. In cloud mode (the client has a ``base_url``) calls
hit the ``/v1/rmp``, ``/v1/review``, ``/v1/monitor`` and ``/v1/protect``
endpoints; in local mode sessions, scenarios and proposals are buffered in
memory — there is no silent fallback. The wire format is the spec's
snake_case ``agenomic.rmp/v0.1`` shapes, matching the Rust ``agenomic-rmp``
engines, so cloud and CLI consume the SDK's output unchanged.

Detection and remediation planning run server-side or in the CLI; the SDK's
job is faithful, redaction-safe instrumentation and the human approval
workflow that closes the loop (Protect findings → scenario enrichment
proposals → Review).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import ulid

from agenomic.exceptions import CloudError

#: Version stamped on every RMP wire type.
RMP_SPEC_VERSION = "agenomic.rmp/v0.1"
#: Version stamped on RMP reports.
RMP_REPORT_VERSION = "agenomic.rmp.report/v0.1"
#: Version stamped on scenario enrichment proposals.
ENRICHMENT_VERSION = "agenomic.rmp.enrichment/v0.1"

#: ``spec_version`` stamped on RMP test scenarios.
SCENARIO_VERSION = "agenomic.rmp.scenario/v0.1"

#: Monitor event types projected into local findings (type → finding kind).
_LOCAL_FINDING_KINDS = {
    "drift.detected": ("drift", "medium"),
    "loop.detected": ("loop", "medium"),
    "harness.violation": ("harness_violation", "high"),
    "agent.failed": ("failure", "high"),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{ulid.new().str}"


def _unwrap(response: Any, key: str) -> dict[str, Any]:
    """Return ``response[key]`` when it is a dict, else the response itself."""
    if not isinstance(response, dict):
        return {}
    inner = response.get(key)
    if isinstance(inner, dict):
        return inner
    return response


def _items(response: Any, key: str) -> list[dict[str, Any]]:
    """Return the ``key`` list of a cloud list response (empty otherwise)."""
    items = response.get(key, []) if isinstance(response, dict) else []
    return list(items) if isinstance(items, list) else []


def _session_from(response: Any, stage: str) -> dict[str, Any]:
    """Extract and validate the session object of a cloud start response."""
    session = _unwrap(response, "session")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise CloudError(f"{stage} start response did not include a session_id")
    return session


class RmpResource:
    """The ``client.rmp`` namespace — umbrella Review·Monitor·Protect sessions.

    Example:
        >>> from agenomic import Client
        >>> session = Client().rmp.start(agent="agent://acme/demo")
        >>> session["session_id"].startswith("rmp_")
        True
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._sessions: dict[str, dict[str, Any]] = {}

    def start(
        self,
        *,
        agent: str,
        release_id: Optional[str] = None,
        environment: str = "production",
        ledger: bool = False,
        genome_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """Open a new RMP session (snake_case ``rmp-session`` wire shape)."""
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {
                "spec_version": RMP_SPEC_VERSION,
                "agent_id": agent,
                "environment": environment,
                "ledger_enabled": ledger,
            }
            if release_id:
                body["release_id"] = release_id
            if genome_hash:
                body["genome_hash"] = genome_hash
            response = self._client._post("/v1/rmp/sessions", body)
            return _session_from(response, "rmp")
        # At most one active session per (agent, environment): reuse it
        # instead of piling up duplicates, mirroring the cloud service.
        for existing in self._sessions.values():
            if (
                existing["agent_id"] == agent
                and existing["environment"] == environment
                and existing["status"] == "active"
            ):
                return existing
        session: dict[str, Any] = {
            "spec_version": RMP_SPEC_VERSION,
            "session_id": _new_id("rmp"),
            "agent_id": agent,
            "environment": environment,
            "mode": "durable_low_latency",
            "ledger_enabled": ledger,
            "status": "active",
            "started_at": _now_iso(),
        }
        if release_id:
            session["release_id"] = release_id
        if genome_hash:
            session["genome_hash"] = genome_hash
        self._sessions[session["session_id"]] = session
        return session

    def stop(self, session_id: str) -> dict[str, Any]:
        """End a local RMP session.

        Frees its (agent, environment) slot so a later ``start()`` opens a
        fresh session instead of reusing this one. Cloud sessions have no
        stop endpoint yet; call :meth:`report` there instead.
        """
        if getattr(self._client, "is_cloud", False):
            raise CloudError("cloud RMP sessions cannot be stopped from the SDK yet")
        session = self.get(session_id)
        session["status"] = "completed"
        session["ended_at"] = _now_iso()
        return session

    def get(self, session_id: str) -> dict[str, Any]:
        """Fetch one RMP session by id."""
        if getattr(self._client, "is_cloud", False):
            response = self._client._get(f"/v1/rmp/sessions/{session_id}")
            return _unwrap(response, "session")
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown RMP session: {session_id!r}") from None

    def list(self) -> list[dict[str, Any]]:
        """List RMP sessions (locally started ones in local mode)."""
        if getattr(self._client, "is_cloud", False):
            response = self._client._get("/v1/rmp/sessions")
            return _items(response, "sessions")
        return list(self._sessions.values())

    def report(self, session_id: str) -> dict[str, Any]:
        """Generate the loop report for a session.

        Cloud mode asks the RMP engine; local mode builds a minimal
        ``agenomic.rmp.report/v0.1`` document (detection runs in the CLI or
        cloud, so a purely local session reports no findings).
        """
        if getattr(self._client, "is_cloud", False):
            report: dict[str, Any] = self._client._post(f"/v1/rmp/sessions/{session_id}/report", {})
            return report
        session = self.get(session_id)
        return {
            "report_version": RMP_REPORT_VERSION,
            "session_id": session_id,
            "agent_id": session["agent_id"],
            "environment": session["environment"],
            "generated_at": _now_iso(),
            "result": "pass",
            "findings": [],
            "alerts": [],
            "recommendations": [],
            "summary": {"findings": 0, "alerts": 0, "recommendations": 0},
        }


class ReviewResource:
    """The ``client.review`` namespace — pre-release review passes.

    Example:
        >>> from agenomic import Client
        >>> review = Client().review
        >>> review.run(agent="agent://acme/demo")["result"]
        'pass'
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._scenarios: list[dict[str, Any]] = []
        self._proposals: dict[str, dict[str, Any]] = {}

    @property
    def proposals(self) -> list[dict[str, Any]]:
        """Scenario enrichment proposals buffered in local mode."""
        return list(self._proposals.values())

    def run(
        self,
        *,
        agent: str,
        scenarios: Optional[list[dict[str, Any]]] = None,
        risk_matrix: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one Review pass over an agent.

        Cloud mode submits the run to the Review engine; local mode returns a
        stub outcome (scenario execution and risk scoring run in the CLI).
        """
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {
                "spec_version": RMP_SPEC_VERSION,
                "agent_id": agent,
            }
            if scenarios is not None:
                body["scenarios"] = scenarios
            if risk_matrix is not None:
                body["risk_matrix"] = risk_matrix
            response = self._client._post("/v1/review/runs", body)
            return _unwrap(response, "run")
        selected = scenarios if scenarios is not None else self.list_scenarios(agent=agent)
        now = _now_iso()
        return {
            "spec_version": RMP_SPEC_VERSION,
            "session_id": _new_id("rev"),
            "agent_id": agent,
            "status": "completed",
            "result": "pass",
            "findings": [],
            "scenario_ids": [s["scenario_id"] for s in selected if "scenario_id" in s],
            "started_at": now,
            "ended_at": now,
        }

    def list_scenarios(self, agent: Optional[str] = None) -> list[dict[str, Any]]:
        """List test scenarios, optionally filtered by agent id."""
        if getattr(self._client, "is_cloud", False):
            path = "/v1/review/scenarios"
            if agent:
                path += f"?agent_id={quote(agent, safe='')}"
            response = self._client._get(path)
            return _items(response, "scenarios")
        return [s for s in self._scenarios if agent is None or s.get("agent_id") == agent]

    def add_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Register a test scenario (stamps ``scenario_id`` when missing)."""
        record = dict(scenario)
        record.setdefault("spec_version", RMP_SPEC_VERSION)
        record.setdefault("scenario_id", _new_id("sc"))
        if getattr(self._client, "is_cloud", False):
            response = self._client._post("/v1/review/scenarios", record)
            unwrapped = _unwrap(response, "scenario")
            return unwrapped if unwrapped.get("scenario_id") else record
        self._scenarios.append(record)
        return record

    def approve_scenario_enrichment(
        self,
        proposal_id: str,
        *,
        session_id: Optional[str] = None,
        reviewer: Optional[str] = None,
    ) -> dict[str, Any]:
        """Approve a scenario enrichment proposal (never applied automatically).

        This is the human-in-the-loop step that feeds Protect output back into
        the Review scenario suite. ``reviewer`` is recorded for the audit trail.
        """
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {}
            if session_id:
                body["session_id"] = session_id
            if reviewer:
                body["reviewer"] = reviewer
            response = self._client._post(f"/v1/review/proposals/{proposal_id}/approve", body)
            return _unwrap(response, "proposal")
        try:
            proposal = self._proposals[proposal_id]
        except KeyError:
            raise KeyError(f"unknown scenario enrichment proposal: {proposal_id!r}") from None
        proposal["status"] = "approved"
        proposal["approved_at"] = _now_iso()
        if reviewer:
            proposal["reviewer"] = reviewer
        if session_id:
            proposal["session_id"] = session_id
        # Materialize the approved scenario into the Review suite so the
        # Protect -> Review loop actually closes offline: subsequent
        # list_scenarios() / run() calls include the enrichment.
        scenario = proposal.get("proposed_scenario")
        if isinstance(scenario, dict):
            scenario_id = scenario.get("scenario_id")
            already = any(s.get("scenario_id") == scenario_id for s in self._scenarios)
            if not already:
                self.add_scenario(dict(scenario))
        return proposal

    def _buffer_proposal(self, proposal: dict[str, Any]) -> None:
        """Buffer a locally generated enrichment proposal for later approval."""
        self._proposals[proposal["proposal_id"]] = proposal


class MonitorSession:
    """A live Monitor-stage session (production observation).

    In cloud mode each event is POSTed; in local mode events are buffered on
    the session so the CLI engines can analyze them offline.
    """

    def __init__(self, client: Any, session_id: str, agent_id: str, environment: str) -> None:
        self._client = client
        self.session_id = session_id
        self.agent_id = agent_id
        self.environment = environment
        self._seq = 0
        self._events: list[dict[str, Any]] = []
        self._stopped = False

    @property
    def cloud(self) -> bool:
        return bool(getattr(self._client, "is_cloud", False))

    @property
    def events(self) -> list[dict[str, Any]]:
        """Events buffered in local mode (empty in cloud mode)."""
        return list(self._events)

    def event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record one runtime event (a snake_case dict with a ``type`` key).

        The session stamps the wire envelope — ``event_id``, ``timestamp``,
        ``sequence_number``, ``session_id``, ``agent_id``, ``spec_version`` —
        and returns the stamped event.
        """
        if self._stopped:
            raise RuntimeError("monitor session already stopped")
        if "type" not in event:
            raise ValueError("monitor event requires a 'type' key")
        stamped: dict[str, Any] = dict(event)
        stamped["spec_version"] = RMP_SPEC_VERSION
        stamped["event_id"] = ulid.new().str
        stamped["session_id"] = self.session_id
        stamped["timestamp"] = _now_iso()
        stamped["sequence_number"] = self._seq
        stamped["agent_id"] = self.agent_id
        self._seq += 1
        if self.cloud:
            self._client._post(f"/v1/monitor/sessions/{self.session_id}/events", stamped)
        else:
            self._events.append(stamped)
        return stamped

    def stop(self) -> None:
        """Finalize the session. Idempotent."""
        if self._stopped:
            return
        # Mark stopped only after a successful stop so a failed cloud POST
        # stays retryable and the remote session isn't orphaned.
        if self.cloud:
            self._client._post(f"/v1/monitor/sessions/{self.session_id}/stop", {})
        self._stopped = True

    def __enter__(self) -> MonitorSession:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


class MonitorResource:
    """The ``client.monitor`` namespace — the Monitor stage of the loop.

    Example:
        >>> from agenomic import Client
        >>> session = Client().monitor.start(agent="agent://acme/demo")
        >>> _ = session.event({"type": "tool.call.completed"})
        >>> session.stop()
        >>> len(session.events)
        1
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._sessions: dict[str, MonitorSession] = {}

    def start(
        self,
        *,
        agent: str,
        release_id: Optional[str] = None,
        environment: str = "production",
        ledger: bool = False,
    ) -> MonitorSession:
        """Start a new Monitor session for a production agent."""
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {
                "spec_version": RMP_SPEC_VERSION,
                "agent_id": agent,
                "environment": environment,
                "ledger_enabled": ledger,
            }
            if release_id:
                body["release_id"] = release_id
            response = self._client._post("/v1/monitor/sessions", body)
            session_id: str = _session_from(response, "monitor")["session_id"]
        else:
            session_id = _new_id("mon")
        session = MonitorSession(self._client, session_id, agent, environment)
        self._sessions[session_id] = session
        return session

    def event(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """Record an event against a session by id (see :meth:`MonitorSession.event`)."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session.event(event)
        if getattr(self._client, "is_cloud", False):
            # The session was started elsewhere; stamp what we can and POST.
            if "type" not in event:
                raise ValueError("monitor event requires a 'type' key")
            stamped: dict[str, Any] = dict(event)
            stamped.setdefault("spec_version", RMP_SPEC_VERSION)
            stamped.setdefault("event_id", ulid.new().str)
            stamped.setdefault("timestamp", _now_iso())
            stamped["session_id"] = session_id
            self._client._post(f"/v1/monitor/sessions/{session_id}/events", stamped)
            return stamped
        raise KeyError(f"unknown monitor session: {session_id!r}")

    def findings(self, session_id: str) -> list[dict[str, Any]]:
        """Fetch Monitor findings for a session.

        Cloud mode asks the Monitor engine. Local mode projects buffered
        detection events (``drift.detected``, ``loop.detected``,
        ``harness.violation``, ``agent.failed``) into finding dicts and is
        otherwise empty — full detection runs in the CLI or cloud.
        """
        if getattr(self._client, "is_cloud", False):
            response = self._client._get(f"/v1/monitor/sessions/{session_id}/findings")
            return _items(response, "findings")
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown monitor session: {session_id!r}")
        findings: list[dict[str, Any]] = []
        for event in session.events:
            projected = _LOCAL_FINDING_KINDS.get(event.get("type", ""))
            if projected is None:
                continue
            kind, severity = projected
            findings.append(
                {
                    "spec_version": RMP_SPEC_VERSION,
                    "finding_id": _new_id("fnd"),
                    "loop_stage": "monitor",
                    "session_id": session_id,
                    "agent_id": session.agent_id,
                    "kind": kind,
                    "severity": severity,
                    "title": f"{event['type']} observed",
                    "message": f"event {event['event_id']} reported {event['type']}",
                    "created_at": _now_iso(),
                    "evidence_refs": [event["event_id"]],
                }
            )
        return findings


class ProtectResource:
    """The ``client.protect`` namespace — the Protect stage of the loop.

    Example:
        >>> from agenomic import Client
        >>> Client().protect.alerts("mon_missing")
        []
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._alerts: dict[str, list[dict[str, Any]]] = {}
        self._recommendations: dict[str, list[dict[str, Any]]] = {}

    def alerts(self, session_id: str) -> list[dict[str, Any]]:
        """List operator-facing alerts raised by Protect for a session."""
        if getattr(self._client, "is_cloud", False):
            response = self._client._get(
                f"/v1/protect/alerts?session_id={quote(session_id, safe='')}"
            )
            return _items(response, "alerts")
        return list(self._alerts.get(session_id, []))

    def action_plan(self, alert_id: str, *, session_id: Optional[str] = None) -> dict[str, Any]:
        """Generate an ordered remediation plan for an alert.

        Local mode builds a minimal deterministic plan and buffers its
        scenario enrichment proposal on ``client.review`` so
        :meth:`ReviewResource.approve_scenario_enrichment` can close the loop.
        """
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {}
            if session_id:
                body["session_id"] = session_id
            response = self._client._post(f"/v1/protect/alerts/{alert_id}/action-plan", body)
            return _unwrap(response, "action_plan")
        now = _now_iso()
        proposal: dict[str, Any] = {
            "spec_version": ENRICHMENT_VERSION,
            "proposal_id": _new_id("sep"),
            "alert_id": alert_id,
            "status": "proposed",
            "requires_human_approval": True,
            "created_at": now,
            # The Review artifact this proposal materializes on approval —
            # approve_scenario_enrichment() adds it to the scenario suite.
            "proposed_scenario": {
                "spec_version": SCENARIO_VERSION,
                "scenario_id": _new_id("sc"),
                "title": f"Regression scenario for alert {alert_id}",
                "source": "protect_derived",
                "severity": "high",
                "created_at": now,
                "evidence_source_refs": [alert_id],
            },
        }
        if session_id:
            proposal["session_id"] = session_id
        steps = [
            {
                "step_id": _new_id("stp"),
                "order": 1,
                "phase": "investigate",
                "title": "Review alert evidence",
                "requires_human_approval": False,
                "status": "pending",
            },
            {
                "step_id": _new_id("stp"),
                "order": 2,
                "phase": "mitigate",
                "title": "Apply recommended guardrail",
                "requires_human_approval": True,
                "status": "pending",
            },
            {
                "step_id": _new_id("stp"),
                "order": 3,
                "phase": "verify",
                "title": "Replay enriched review scenarios",
                "requires_human_approval": False,
                "status": "pending",
            },
        ]
        plan: dict[str, Any] = {
            "spec_version": RMP_SPEC_VERSION,
            "plan_id": _new_id("apl"),
            "alert_id": alert_id,
            "created_at": now,
            "steps": steps,
            "scenario_proposal_ids": [proposal["proposal_id"]],
        }
        if session_id:
            plan["session_id"] = session_id
        review = getattr(self._client, "review", None)
        if isinstance(review, ReviewResource):
            review._buffer_proposal(proposal)
        return plan

    def recommendations(self, session_id: str) -> list[dict[str, Any]]:
        """List structural / prompt / policy recommendations for a session."""
        if getattr(self._client, "is_cloud", False):
            response = self._client._get(
                f"/v1/protect/recommendations?session_id={quote(session_id, safe='')}"
            )
            return _items(response, "recommendations")
        return list(self._recommendations.get(session_id, []))

    def notify(self, alert_id: str, *, session_id: Optional[str] = None) -> dict[str, Any]:
        """Route an alert to its configured channels.

        Local mode has no delivery integrations; it returns a stub routing
        record on the ``stdout`` channel.
        """
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {}
            if session_id:
                body["session_id"] = session_id
            routed: dict[str, Any] = self._client._post(
                f"/v1/protect/alerts/{alert_id}/route", body
            )
            return routed
        record: dict[str, Any] = {
            "spec_version": RMP_SPEC_VERSION,
            "alert_id": alert_id,
            "routes": [{"target": "local", "channel": "stdout", "routed_at": _now_iso()}],
        }
        if session_id:
            record["session_id"] = session_id
        return record
