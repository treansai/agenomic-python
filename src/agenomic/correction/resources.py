"""``client.correction``: incidents, scenario conversion and reviewed candidate fixes.

The gateway computes every verdict (replay, weakening, freshness) and enforces
the roles: opening, converting, dismissing, creating, verifying, reviewing and
rejecting need the write role; approving, promoting, enforcing, rolling back,
changing settings and sweeping need the owner. The SDK does not check roles
and never decides a transition. Nothing is promoted automatically. Cloud mode
only; a deployment without Postgres refuses every route.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union
from urllib.parse import urlencode

from pydantic import BaseModel

from agenomic.correction.models import (
    Candidate,
    CandidateKind,
    CandidateStatus,
    ConfigChange,
    CorrectionSettings,
    Incident,
    IncidentSource,
    IncidentStatus,
    OpenedIncident,
    PolicyChange,
    PromptChange,
    ReplayReport,
    SweepOutcome,
    ToolContractChange,
    Transition,
    VerifyOutcome,
)
from agenomic.tools.models import ToolExecutionError
from agenomic.tools.resources import atyped_request, typed_request

Payload = Union[PromptChange, PolicyChange, ToolContractChange, ConfigChange, Mapping[str, Any]]

Incidents = list[Incident]
Candidates = list[Candidate]
Transitions = list[Transition]


def _query(**params: Any) -> str:
    present = {key: value for key, value in params.items() if value is not None}
    return f"?{urlencode(present)}" if present else ""


def _items(response: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    items = response.get(key)
    if not isinstance(items, list):
        raise ToolExecutionError("invalid_response", f"list response carries no {key!r} array", 0)
    return [dict(item) for item in items]


def _note(body: dict[str, Any], note: Optional[str]) -> dict[str, Any]:
    if note is not None:
        body["note"] = note
    return body


def _reason(reason: Optional[str]) -> dict[str, Any]:
    return {} if reason is None else {"reason": reason}


class _Base:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _require_cloud(self) -> None:
        if not getattr(self._client, "is_cloud", False):
            raise ToolExecutionError(
                "cloud_required", "the correction loop is run by the Agenomic Cloud gateway", 0
            )

    def _request(
        self, method: str, path: str, body: Optional[Mapping[str, Any]] = None
    ) -> dict[str, Any]:
        self._require_cloud()
        return typed_request(self._client, method, path, body)

    async def _arequest(
        self, method: str, path: str, body: Optional[Mapping[str, Any]] = None
    ) -> dict[str, Any]:
        self._require_cloud()
        return await atyped_request(self._client, method, path, body)


class IncidentsResource(_Base):
    """Incidents: a Protect decision, an RMP finding or a failed run, named by source and ref.

    The caller only names the source; severity and text are read from the stored
    record, scrubbed and capped by the gateway.
    """

    def list(
        self,
        *,
        status: Optional[IncidentStatus] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Incidents:
        """List incidents of the organization (read)."""
        path = f"/v1/correction/incidents{_query(status=status, agent_id=agent_id, limit=limit)}"
        return [Incident.model_validate(i) for i in _items(self._request("GET", path), "incidents")]

    async def alist(
        self,
        *,
        status: Optional[IncidentStatus] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Incidents:
        """Async counterpart of :meth:`list`."""
        path = f"/v1/correction/incidents{_query(status=status, agent_id=agent_id, limit=limit)}"
        items = _items(await self._arequest("GET", path), "incidents")
        return [Incident.model_validate(i) for i in items]

    def open(self, source: IncidentSource, source_ref: str) -> OpenedIncident:
        """Open an incident (write role); a duplicate of a live one raises its count (``created`` false)."""
        body = {"source": source, "source_ref": source_ref}
        return OpenedIncident.model_validate(
            self._request("POST", "/v1/correction/incidents", body)
        )

    async def aopen(self, source: IncidentSource, source_ref: str) -> OpenedIncident:
        """Async counterpart of :meth:`open`."""
        body = {"source": source, "source_ref": source_ref}
        return OpenedIncident.model_validate(
            await self._arequest("POST", "/v1/correction/incidents", body)
        )

    def get(self, incident_id: str) -> Incident:
        """Read one incident."""
        return Incident.model_validate(
            self._request("GET", f"/v1/correction/incidents/{incident_id}")
        )

    async def aget(self, incident_id: str) -> Incident:
        """Async counterpart of :meth:`get`."""
        return Incident.model_validate(
            await self._arequest("GET", f"/v1/correction/incidents/{incident_id}")
        )

    def to_scenario(self, incident_id: str) -> Incident:
        """Turn the incident into a regression scenario proposal (write role).

        The scenario joins the corpus only through the RMP proposal approval.
        """
        path = f"/v1/correction/incidents/{incident_id}/scenario"
        return Incident.model_validate(self._request("POST", path))

    async def ato_scenario(self, incident_id: str) -> Incident:
        """Async counterpart of :meth:`to_scenario`."""
        path = f"/v1/correction/incidents/{incident_id}/scenario"
        return Incident.model_validate(await self._arequest("POST", path))

    def dismiss(self, incident_id: str, *, reason: Optional[str] = None) -> Incident:
        """Dismiss the incident (write role)."""
        path = f"/v1/correction/incidents/{incident_id}/dismiss"
        return Incident.model_validate(self._request("POST", path, _reason(reason)))

    async def adismiss(self, incident_id: str, *, reason: Optional[str] = None) -> Incident:
        """Async counterpart of :meth:`dismiss`."""
        path = f"/v1/correction/incidents/{incident_id}/dismiss"
        return Incident.model_validate(await self._arequest("POST", path, _reason(reason)))


def _create_body(
    incident_id: str,
    kind: CandidateKind,
    title: str,
    payload: Payload,
    rationale: str,
    claims_weakening: bool,
) -> dict[str, Any]:
    raw = (
        payload.model_dump(exclude_unset=True) if isinstance(payload, BaseModel) else dict(payload)
    )
    return {
        "incident_id": incident_id,
        "kind": kind,
        "title": title,
        "rationale": rationale,
        "payload": raw,
        "claims_weakening": claims_weakening,
    }


class CandidatesResource(_Base):
    """Candidates: a digest-bound proposed fix and its verify, review, approve, promote path.

    Every step after creation carries the ``content_digest`` the actor looked at;
    a changed digest, a stale scenario or a stale binding set is a 409. The path
    is ``proposed``, ``verified``, ``reviewed``, ``approved``, ``staged`` (enforce
    payloads, shadow soak), ``promoted``, then ``rolled_back`` from ``staged`` or
    ``promoted``; ``rejected`` before staging. Only ``policy_change`` can be
    verified conclusively and promoted.
    """

    def list(
        self,
        *,
        status: Optional[CandidateStatus] = None,
        incident_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Candidates:
        """List candidates (read)."""
        path = f"/v1/correction/candidates{_query(status=status, incident_id=incident_id, limit=limit)}"
        return [
            Candidate.model_validate(c) for c in _items(self._request("GET", path), "candidates")
        ]

    async def alist(
        self,
        *,
        status: Optional[CandidateStatus] = None,
        incident_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Candidates:
        """Async counterpart of :meth:`list`."""
        path = f"/v1/correction/candidates{_query(status=status, incident_id=incident_id, limit=limit)}"
        items = _items(await self._arequest("GET", path), "candidates")
        return [Candidate.model_validate(c) for c in items]

    def create(
        self,
        incident_id: str,
        *,
        kind: CandidateKind,
        title: str,
        payload: Payload,
        rationale: str = "",
        claims_weakening: bool = False,
    ) -> Candidate:
        """Propose a fix for an incident (write role); the payload is immutable afterwards.

        ``payload`` is validated by the gateway per kind (unknown fields are
        refused). ``claims_weakening`` can only raise the weakening verdict.
        """
        body = _create_body(incident_id, kind, title, payload, rationale, claims_weakening)
        return Candidate.model_validate(self._request("POST", "/v1/correction/candidates", body))

    async def acreate(
        self,
        incident_id: str,
        *,
        kind: CandidateKind,
        title: str,
        payload: Payload,
        rationale: str = "",
        claims_weakening: bool = False,
    ) -> Candidate:
        """Async counterpart of :meth:`create`."""
        body = _create_body(incident_id, kind, title, payload, rationale, claims_weakening)
        return Candidate.model_validate(
            await self._arequest("POST", "/v1/correction/candidates", body)
        )

    def get(self, candidate_id: str) -> Candidate:
        """Read one candidate."""
        return Candidate.model_validate(
            self._request("GET", f"/v1/correction/candidates/{candidate_id}")
        )

    async def aget(self, candidate_id: str) -> Candidate:
        """Async counterpart of :meth:`get`."""
        return Candidate.model_validate(
            await self._arequest("GET", f"/v1/correction/candidates/{candidate_id}")
        )

    def transitions(self, candidate_id: str) -> Transitions:
        """The append-only transition log of a candidate."""
        path = f"/v1/correction/candidates/{candidate_id}/transitions"
        return [
            Transition.model_validate(t) for t in _items(self._request("GET", path), "transitions")
        ]

    async def atransitions(self, candidate_id: str) -> Transitions:
        """Async counterpart of :meth:`transitions`."""
        path = f"/v1/correction/candidates/{candidate_id}/transitions"
        items = _items(await self._arequest("GET", path), "transitions")
        return [Transition.model_validate(t) for t in items]

    def replay(self, candidate_id: str) -> ReplayReport:
        """The replay report currently bound to the candidate."""
        path = f"/v1/correction/candidates/{candidate_id}/replay"
        return ReplayReport.model_validate(self._request("GET", path))

    async def areplay(self, candidate_id: str) -> ReplayReport:
        """Async counterpart of :meth:`replay`."""
        path = f"/v1/correction/candidates/{candidate_id}/replay"
        return ReplayReport.model_validate(await self._arequest("GET", path))

    def verify(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> VerifyOutcome:
        """Run the comparative replay (write role); the verdict is computed server side.

        Only a ``passed`` report moves the candidate to ``verified``. A failed or
        inconclusive attempt keeps it ``proposed`` and is still returned.
        """
        path = f"/v1/correction/candidates/{candidate_id}/verify"
        body = _note({"content_digest": content_digest}, note)
        return VerifyOutcome.model_validate(self._request("POST", path, body))

    async def averify(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> VerifyOutcome:
        """Async counterpart of :meth:`verify`."""
        path = f"/v1/correction/candidates/{candidate_id}/verify"
        body = _note({"content_digest": content_digest}, note)
        return VerifyOutcome.model_validate(await self._arequest("POST", path, body))

    def review(
        self,
        candidate_id: str,
        *,
        content_digest: str,
        report_id: str,
        note: Optional[str] = None,
    ) -> Candidate:
        """Record a review of the report (write role); the reviewer must differ from the author."""
        path = f"/v1/correction/candidates/{candidate_id}/review"
        body = _note({"content_digest": content_digest, "report_id": report_id}, note)
        return Candidate.model_validate(self._request("POST", path, body))

    async def areview(
        self,
        candidate_id: str,
        *,
        content_digest: str,
        report_id: str,
        note: Optional[str] = None,
    ) -> Candidate:
        """Async counterpart of :meth:`review`."""
        path = f"/v1/correction/candidates/{candidate_id}/review"
        body = _note({"content_digest": content_digest, "report_id": report_id}, note)
        return Candidate.model_validate(await self._arequest("POST", path, body))

    def approve(
        self,
        candidate_id: str,
        *,
        content_digest: str,
        report_id: str,
        acknowledge_weakening: bool = False,
        note: Optional[str] = None,
    ) -> Candidate:
        """Approve (owner); a weakening or unknown verdict needs ``acknowledge_weakening`` and a note."""
        path = f"/v1/correction/candidates/{candidate_id}/approve"
        body = _note(
            {
                "content_digest": content_digest,
                "report_id": report_id,
                "acknowledge_weakening": acknowledge_weakening,
            },
            note,
        )
        return Candidate.model_validate(self._request("POST", path, body))

    async def aapprove(
        self,
        candidate_id: str,
        *,
        content_digest: str,
        report_id: str,
        acknowledge_weakening: bool = False,
        note: Optional[str] = None,
    ) -> Candidate:
        """Async counterpart of :meth:`approve`."""
        path = f"/v1/correction/candidates/{candidate_id}/approve"
        body = _note(
            {
                "content_digest": content_digest,
                "report_id": report_id,
                "acknowledge_weakening": acknowledge_weakening,
            },
            note,
        )
        return Candidate.model_validate(await self._arequest("POST", path, body))

    def reject(self, candidate_id: str, *, reason: Optional[str] = None) -> Candidate:
        """Reject before staging (write role); a rejected candidate is never reopened."""
        path = f"/v1/correction/candidates/{candidate_id}/reject"
        return Candidate.model_validate(self._request("POST", path, _reason(reason)))

    async def areject(self, candidate_id: str, *, reason: Optional[str] = None) -> Candidate:
        """Async counterpart of :meth:`reject`."""
        path = f"/v1/correction/candidates/{candidate_id}/reject"
        return Candidate.model_validate(await self._arequest("POST", path, _reason(reason)))

    def promote(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> Candidate:
        """Promote an approved candidate (owner); refused while a kill switch covers it.

        An ``enforce`` payload goes to ``staged`` (the candidate policy is bound
        in shadow); a ``shadow`` payload is applied at once. A stale candidate
        invalidates its approval and must be proposed again.
        """
        path = f"/v1/correction/candidates/{candidate_id}/promote"
        return Candidate.model_validate(
            self._request("POST", path, _note({"content_digest": content_digest}, note))
        )

    async def apromote(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> Candidate:
        """Async counterpart of :meth:`promote`."""
        path = f"/v1/correction/candidates/{candidate_id}/promote"
        return Candidate.model_validate(
            await self._arequest("POST", path, _note({"content_digest": content_digest}, note))
        )

    def enforce(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> Candidate:
        """Move a staged candidate to enforce after the shadow soak (owner).

        The soak is a time delay only; nothing reads the shadow decisions.
        """
        path = f"/v1/correction/candidates/{candidate_id}/enforce"
        return Candidate.model_validate(
            self._request("POST", path, _note({"content_digest": content_digest}, note))
        )

    async def aenforce(
        self, candidate_id: str, *, content_digest: str, note: Optional[str] = None
    ) -> Candidate:
        """Async counterpart of :meth:`enforce`."""
        path = f"/v1/correction/candidates/{candidate_id}/enforce"
        return Candidate.model_validate(
            await self._arequest("POST", path, _note({"content_digest": content_digest}, note))
        )

    def rollback(self, candidate_id: str, *, reason: str) -> Candidate:
        """Undo a staged or promoted candidate (owner, idempotent, ignores the kill switch)."""
        path = f"/v1/correction/candidates/{candidate_id}/rollback"
        return Candidate.model_validate(self._request("POST", path, {"reason": reason}))

    async def arollback(self, candidate_id: str, *, reason: str) -> Candidate:
        """Async counterpart of :meth:`rollback`."""
        path = f"/v1/correction/candidates/{candidate_id}/rollback"
        return Candidate.model_validate(await self._arequest("POST", path, {"reason": reason}))


class CorrectionResource(_Base):
    """The ``client.correction`` namespace.

    Example:
        >>> from agenomic import Client
        >>> Client().correction.incidents.list()
        Traceback (most recent call last):
        ...
        agenomic.tools.models.ToolExecutionError: cloud_required: the correction loop is run by the Agenomic Cloud gateway
    """

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.incidents = IncidentsResource(client)
        self.candidates = CandidatesResource(client)

    def settings(self) -> CorrectionSettings:
        """Trigger and soak settings (read)."""
        return CorrectionSettings.model_validate(self._request("GET", "/v1/correction/settings"))

    async def asettings(self) -> CorrectionSettings:
        """Async counterpart of :meth:`settings`."""
        return CorrectionSettings.model_validate(
            await self._arequest("GET", "/v1/correction/settings")
        )

    def put_settings(self, settings: CorrectionSettings) -> CorrectionSettings:
        """Replace the settings (owner); the full object is sent, ranges are checked server side."""
        body = settings.model_dump()
        return CorrectionSettings.model_validate(
            self._request("PUT", "/v1/correction/settings", body)
        )

    async def aput_settings(self, settings: CorrectionSettings) -> CorrectionSettings:
        """Async counterpart of :meth:`put_settings`."""
        body = settings.model_dump()
        return CorrectionSettings.model_validate(
            await self._arequest("PUT", "/v1/correction/settings", body)
        )

    def sweep(self) -> SweepOutcome:
        """Run the proactive trigger once (owner); it opens incidents and never a candidate."""
        return SweepOutcome.model_validate(self._request("POST", "/v1/correction/triggers/sweep"))

    async def asweep(self) -> SweepOutcome:
        """Async counterpart of :meth:`sweep`."""
        return SweepOutcome.model_validate(
            await self._arequest("POST", "/v1/correction/triggers/sweep")
        )


__all__ = ["CandidatesResource", "CorrectionResource", "IncidentsResource"]
