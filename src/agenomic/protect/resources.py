"""``client.protect``: proactive policy enforcement on top of the RMP Protect stage.

Every method talks to Agenomic Cloud through the typed tool-execution
transport: refusals raise :class:`agenomic.tools.ToolExecutionError` with
the server error code. Nothing here evaluates a policy locally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Union
from urllib.parse import urlencode

from agenomic.protect.models import (
    DelegationBudget,
    DelegationScope,
    ProtectExemption,
    ProtectOverlay,
    ProtectSettings,
    SignedDelegation,
)
from agenomic.rmp.resources import ProtectResource as _LoopProtectResource
from agenomic.tools.models import ToolExecutionError
from agenomic.tools.resources import atyped_request, typed_request


def _query(**params: Any) -> str:
    present = {key: value for key, value in params.items() if value is not None}
    return f"?{urlencode(present)}" if present else ""


def _policy_version(policy_id: str, version: str) -> str:
    return f"{policy_id}@{version}"


Items = list[dict[str, Any]]


def _items(response: Mapping[str, Any], key: str) -> Items:
    items = response.get(key)
    if not isinstance(items, list):
        raise ToolExecutionError("invalid_response", f"list response carries no {key!r} array", 0)
    return [dict(item) for item in items]


def _page(response: Mapping[str, Any], key: str) -> dict[str, Any]:
    cursor = response.get("next_cursor")
    return {key: _items(response, key), "next_cursor": str(cursor) if cursor else None}


def _register_body(document: Union[Mapping[str, Any], str]) -> dict[str, Any]:
    return {"document_text": document} if isinstance(document, str) else dict(document)


class _Namespace:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _require_cloud(self) -> None:
        if not getattr(self._client, "is_cloud", False):
            raise ToolExecutionError(
                "cloud_required",
                "protect policies are evaluated by the Agenomic Cloud gateway only",
                0,
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


class ApprovalsResource(_Namespace):
    """``client.protect.approvals``: pending human approvals."""

    def list(self, *, status: Optional[str] = None, run_id: Optional[str] = None) -> Items:
        return _items(
            self._request("GET", "/v1/protect/approvals" + _query(status=status, run_id=run_id)),
            "approvals",
        )

    async def alist(self, *, status: Optional[str] = None, run_id: Optional[str] = None) -> Items:
        return _items(
            await self._arequest(
                "GET", "/v1/protect/approvals" + _query(status=status, run_id=run_id)
            ),
            "approvals",
        )

    def get(self, approval_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/protect/approvals/{approval_id}")

    async def aget(self, approval_id: str) -> dict[str, Any]:
        return await self._arequest("GET", f"/v1/protect/approvals/{approval_id}")

    def decide(
        self, approval_id: str, decision: str, *, comment: Optional[str] = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": decision}
        if comment is not None:
            body["comment"] = comment
        return self._request("POST", f"/v1/protect/approvals/{approval_id}/decide", body)

    async def adecide(
        self, approval_id: str, decision: str, *, comment: Optional[str] = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": decision}
        if comment is not None:
            body["comment"] = comment
        return await self._arequest("POST", f"/v1/protect/approvals/{approval_id}/decide", body)


class DecisionsResource(_Namespace):
    """``client.protect.decisions``: recorded admission decisions.

    ``list`` is the only paginated read: it returns
    ``{"decisions": [...], "next_cursor": str | None}`` and ``cursor`` continues a page.
    """

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        outcome: Optional[str] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        query = _query(run_id=run_id, outcome=outcome, since=since, limit=limit, cursor=cursor)
        return _page(self._request("GET", "/v1/protect/decisions" + query), "decisions")

    async def alist(
        self,
        *,
        run_id: Optional[str] = None,
        outcome: Optional[str] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        query = _query(run_id=run_id, outcome=outcome, since=since, limit=limit, cursor=cursor)
        return _page(await self._arequest("GET", "/v1/protect/decisions" + query), "decisions")

    def get(self, decision_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/protect/decisions/{decision_id}")

    async def aget(self, decision_id: str) -> dict[str, Any]:
        return await self._arequest("GET", f"/v1/protect/decisions/{decision_id}")


class PoliciesResource(_Namespace):
    """``client.protect.policies``: the policy registry (draft, release, deprecate, simulate)."""

    def register(self, document: Union[Mapping[str, Any], str]) -> dict[str, Any]:
        """Register a draft: a policy document (bare JSON) or its YAML/JSON text."""
        return self._request("POST", "/v1/policies", _register_body(document))

    async def aregister(self, document: Union[Mapping[str, Any], str]) -> dict[str, Any]:
        return await self._arequest("POST", "/v1/policies", _register_body(document))

    def get(self, policy_id: str, version: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/policies/{_policy_version(policy_id, version)}")

    async def aget(self, policy_id: str, version: str) -> dict[str, Any]:
        return await self._arequest("GET", f"/v1/policies/{_policy_version(policy_id, version)}")

    def list(self) -> Items:
        return _items(self._request("GET", "/v1/policies"), "policies")

    async def alist(self) -> Items:
        return _items(await self._arequest("GET", "/v1/policies"), "policies")

    def release(self, policy_id: str, version: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/policies/{_policy_version(policy_id, version)}/release", {}
        )

    async def arelease(self, policy_id: str, version: str) -> dict[str, Any]:
        return await self._arequest(
            "POST", f"/v1/policies/{_policy_version(policy_id, version)}/release", {}
        )

    def deprecate(self, policy_id: str, version: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/policies/{_policy_version(policy_id, version)}/deprecate", {}
        )

    async def adeprecate(self, policy_id: str, version: str) -> dict[str, Any]:
        return await self._arequest(
            "POST", f"/v1/policies/{_policy_version(policy_id, version)}/deprecate", {}
        )

    def simulate(
        self, policy_id: str, version: str, intents: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/policies/{_policy_version(policy_id, version)}/simulate",
            {"intents": [dict(intent) for intent in intents]},
        )

    async def asimulate(
        self, policy_id: str, version: str, intents: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return await self._arequest(
            "POST",
            f"/v1/policies/{_policy_version(policy_id, version)}/simulate",
            {"intents": [dict(intent) for intent in intents]},
        )

    def diff(self, policy_id: str, version: str, *, against: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/policies/{_policy_version(policy_id, version)}/diff" + _query(against=against),
        )

    async def adiff(self, policy_id: str, version: str, *, against: str) -> dict[str, Any]:
        return await self._arequest(
            "GET",
            f"/v1/policies/{_policy_version(policy_id, version)}/diff" + _query(against=against),
        )


def _binding_body(
    policy_id: str, version: str, scope_kind: str, scope_ref: Optional[str], mode: str
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "policy_id": policy_id,
        "version": version,
        "scope_kind": scope_kind,
        "mode": mode,
    }
    if scope_ref is not None:
        body["scope_ref"] = scope_ref
    return body


class BindingsResource(_Namespace):
    """``client.protect.bindings``: released policy versions bound to a scope."""

    def list(
        self,
        *,
        scope_kind: Optional[str] = None,
        scope_ref: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Items:
        query = _query(scope_kind=scope_kind, scope_ref=scope_ref, status=status)
        return _items(self._request("GET", "/v1/protect/bindings" + query), "bindings")

    async def alist(
        self,
        *,
        scope_kind: Optional[str] = None,
        scope_ref: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Items:
        query = _query(scope_kind=scope_kind, scope_ref=scope_ref, status=status)
        return _items(await self._arequest("GET", "/v1/protect/bindings" + query), "bindings")

    def create(
        self,
        policy_id: str,
        version: str,
        scope_kind: str,
        scope_ref: Optional[str] = None,
        *,
        mode: str = "enforce",
    ) -> dict[str, Any]:
        body = _binding_body(policy_id, version, scope_kind, scope_ref, mode)
        return self._request("POST", "/v1/protect/bindings", body)

    async def acreate(
        self,
        policy_id: str,
        version: str,
        scope_kind: str,
        scope_ref: Optional[str] = None,
        *,
        mode: str = "enforce",
    ) -> dict[str, Any]:
        body = _binding_body(policy_id, version, scope_kind, scope_ref, mode)
        return await self._arequest("POST", "/v1/protect/bindings", body)

    def revoke(self, binding_id: str, *, reason: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/protect/bindings/{binding_id}/revoke", {"reason": reason}
        )

    async def arevoke(self, binding_id: str, *, reason: str) -> dict[str, Any]:
        return await self._arequest(
            "POST", f"/v1/protect/bindings/{binding_id}/revoke", {"reason": reason}
        )


def _restriction_body(
    scope_kind: str,
    kind: str,
    reason: str,
    scope_ref: Optional[str],
    parameters: Optional[Mapping[str, Any]],
    expires_at: Optional[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {"scope_kind": scope_kind, "kind": kind, "reason": reason}
    if scope_ref is not None:
        body["scope_ref"] = scope_ref
    if parameters is not None:
        body["parameters"] = dict(parameters)
    if expires_at is not None:
        body["expires_at"] = expires_at
    return body


class RestrictionsResource(_Namespace):
    """``client.protect.restrictions``: suspensions, tool blocks and caps."""

    def list(self, *, status: Optional[str] = None) -> Items:
        return _items(
            self._request("GET", "/v1/protect/restrictions" + _query(status=status)),
            "restrictions",
        )

    async def alist(self, *, status: Optional[str] = None) -> Items:
        return _items(
            await self._arequest("GET", "/v1/protect/restrictions" + _query(status=status)),
            "restrictions",
        )

    def create(
        self,
        *,
        scope_kind: str,
        kind: str,
        reason: str,
        scope_ref: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> dict[str, Any]:
        body = _restriction_body(scope_kind, kind, reason, scope_ref, parameters, expires_at)
        return self._request("POST", "/v1/protect/restrictions", body)

    async def acreate(
        self,
        *,
        scope_kind: str,
        kind: str,
        reason: str,
        scope_ref: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> dict[str, Any]:
        body = _restriction_body(scope_kind, kind, reason, scope_ref, parameters, expires_at)
        return await self._arequest("POST", "/v1/protect/restrictions", body)

    def lift(self, restriction_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/protect/restrictions/{restriction_id}/lift", {})

    async def alift(self, restriction_id: str) -> dict[str, Any]:
        return await self._arequest("POST", f"/v1/protect/restrictions/{restriction_id}/lift", {})


def _exemption_body(
    scope_kind: str, reason: str, expires_at: str, scope_ref: Optional[str]
) -> dict[str, Any]:
    body: dict[str, Any] = {"scope_kind": scope_kind, "reason": reason, "expires_at": expires_at}
    if scope_ref is not None:
        body["scope_ref"] = scope_ref
    return body


Exemptions = list[ProtectExemption]


class ExemptionsResource(_Namespace):
    """``client.protect.exemptions``: time-boxed exemptions from mandatory protection.

    Creating and lifting an exemption is reserved to organization owners.
    """

    def list(self, *, status: Optional[str] = None) -> Exemptions:
        response = self._request("GET", "/v1/protect/exemptions" + _query(status=status))
        return [ProtectExemption.model_validate(item) for item in _items(response, "exemptions")]

    async def alist(self, *, status: Optional[str] = None) -> Exemptions:
        response = await self._arequest("GET", "/v1/protect/exemptions" + _query(status=status))
        return [ProtectExemption.model_validate(item) for item in _items(response, "exemptions")]

    def create(
        self,
        *,
        scope_kind: str,
        reason: str,
        expires_at: str,
        scope_ref: Optional[str] = None,
    ) -> ProtectExemption:
        body = _exemption_body(scope_kind, reason, expires_at, scope_ref)
        return ProtectExemption.model_validate(
            self._request("POST", "/v1/protect/exemptions", body)
        )

    async def acreate(
        self,
        *,
        scope_kind: str,
        reason: str,
        expires_at: str,
        scope_ref: Optional[str] = None,
    ) -> ProtectExemption:
        body = _exemption_body(scope_kind, reason, expires_at, scope_ref)
        return ProtectExemption.model_validate(
            await self._arequest("POST", "/v1/protect/exemptions", body)
        )

    def lift(self, exemption_id: str) -> ProtectExemption:
        return ProtectExemption.model_validate(
            self._request("POST", f"/v1/protect/exemptions/{exemption_id}/lift", {})
        )

    async def alift(self, exemption_id: str) -> ProtectExemption:
        return ProtectExemption.model_validate(
            await self._arequest("POST", f"/v1/protect/exemptions/{exemption_id}/lift", {})
        )


def _kill_switch_body(scope_kind: str, scope_ref: Optional[str], reason: str) -> dict[str, Any]:
    body: dict[str, Any] = {"scope_kind": scope_kind, "reason": reason}
    if scope_ref is not None:
        body["scope_ref"] = scope_ref
    return body


def _settings_body(protect_mandatory: bool, expected_revision: Optional[int]) -> dict[str, Any]:
    body: dict[str, Any] = {"protect_mandatory": protect_mandatory}
    if expected_revision is not None:
        body["expected_revision"] = expected_revision
    return body


def _simulate_body(
    intents: Sequence[Mapping[str, Any]],
    policies: Optional[Sequence[Mapping[str, Any]]],
    policy_refs: Optional[Sequence[str]],
    decisions_from_run: Optional[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {"intents": [dict(intent) for intent in intents]}
    if policies is not None:
        body["policies"] = [dict(policy) for policy in policies]
    if policy_refs is not None:
        body["policy_refs"] = list(policy_refs)
    if decisions_from_run is not None:
        body["decisions_from_run"] = decisions_from_run
    return body


def _delegation_body(
    delegate_agent_id: str,
    scope: Union[DelegationScope, Mapping[str, Any]],
    budget: Union[DelegationBudget, Mapping[str, Any]],
    expires_at: Union[datetime, str],
    parent_id: Optional[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "delegate_agent_id": delegate_agent_id,
        "scope": scope.model_dump() if isinstance(scope, DelegationScope) else dict(scope),
        "budget": budget.model_dump() if isinstance(budget, DelegationBudget) else dict(budget),
        "expires_at": expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at,
    }
    if parent_id is not None:
        body["parent_id"] = parent_id
    return body


class DelegationsResource(_Namespace):
    """Signed delegation links: user or API key, then agent, then sub-agent.

    A link can only narrow its parent (tools, families, effect ceiling, action
    budget, expiry); the gateway checks this at issuance and again at every
    admission. ``issued_by`` comes from the credential, never from the body.
    A run opts in with ``protect.delegation_id`` (the last link of the chain) in
    its run config; a run that does not declare it is not attenuated. There is
    no revocation, only expiry and the kill switch.
    """

    def issue(
        self,
        delegate_agent_id: str,
        *,
        scope: Union[DelegationScope, Mapping[str, Any]],
        budget: Union[DelegationBudget, Mapping[str, Any]],
        expires_at: Union[datetime, str],
        parent_id: Optional[str] = None,
    ) -> SignedDelegation:
        """Issue a link (write role); the root is opened by the caller, a child names ``parent_id``."""
        body = _delegation_body(delegate_agent_id, scope, budget, expires_at, parent_id)
        return SignedDelegation.model_validate(
            self._request("POST", "/v1/protect/delegations", body)
        )

    async def aissue(
        self,
        delegate_agent_id: str,
        *,
        scope: Union[DelegationScope, Mapping[str, Any]],
        budget: Union[DelegationBudget, Mapping[str, Any]],
        expires_at: Union[datetime, str],
        parent_id: Optional[str] = None,
    ) -> SignedDelegation:
        """Async counterpart of :meth:`issue`."""
        body = _delegation_body(delegate_agent_id, scope, budget, expires_at, parent_id)
        return SignedDelegation.model_validate(
            await self._arequest("POST", "/v1/protect/delegations", body)
        )

    def get(self, delegation_id: str) -> SignedDelegation:
        """Read one signed link."""
        return SignedDelegation.model_validate(
            self._request("GET", f"/v1/protect/delegations/{delegation_id}")
        )

    async def aget(self, delegation_id: str) -> SignedDelegation:
        """Async counterpart of :meth:`get`."""
        return SignedDelegation.model_validate(
            await self._arequest("GET", f"/v1/protect/delegations/{delegation_id}")
        )


class ProtectResource(_LoopProtectResource, _Namespace):
    """The ``client.protect`` namespace: RMP Protect stage plus policy enforcement.

    Example:
        >>> from agenomic import Client
        >>> protect = Client().protect
        >>> protect.alerts("mon_missing")
        []
        >>> protect.coverage()
        Traceback (most recent call last):
        ...
        agenomic.tools.models.ToolExecutionError: cloud_required: protect policies are evaluated by the Agenomic Cloud gateway only
    """

    def __init__(self, client: Any) -> None:
        _LoopProtectResource.__init__(self, client)
        self.approvals = ApprovalsResource(client)
        self.decisions = DecisionsResource(client)
        self.policies = PoliciesResource(client)
        self.bindings = BindingsResource(client)
        self.restrictions = RestrictionsResource(client)
        self.exemptions = ExemptionsResource(client)
        self.delegations = DelegationsResource(client)

    def overlay(self, run_id: str) -> ProtectOverlay:
        """Instruction overlay of a run, to inject as the first system message."""
        return ProtectOverlay.model_validate(
            self._request("GET", f"/v1/protect/runs/{run_id}/overlay")
        )

    async def aoverlay(self, run_id: str) -> ProtectOverlay:
        """Async counterpart of :meth:`overlay`."""
        return ProtectOverlay.model_validate(
            await self._arequest("GET", f"/v1/protect/runs/{run_id}/overlay")
        )

    def catalog(self, run_id: str) -> dict[str, Any]:
        """Tools currently allowed by the effective policies of a run."""
        return self._request("GET", f"/v1/protect/runs/{run_id}/catalog")

    async def acatalog(self, run_id: str) -> dict[str, Any]:
        """Async counterpart of :meth:`catalog`."""
        return await self._arequest("GET", f"/v1/protect/runs/{run_id}/catalog")

    def kill_switch(
        self, scope_kind: str, scope_ref: Optional[str] = None, *, reason: str
    ) -> dict[str, Any]:
        """Suspend a scope (org, agent, run or tool) and cancel its running protect runs."""
        body = _kill_switch_body(scope_kind, scope_ref, reason)
        return self._request("POST", "/v1/protect/kill-switch", body)

    async def akill_switch(
        self, scope_kind: str, scope_ref: Optional[str] = None, *, reason: str
    ) -> dict[str, Any]:
        """Async counterpart of :meth:`kill_switch`."""
        body = _kill_switch_body(scope_kind, scope_ref, reason)
        return await self._arequest("POST", "/v1/protect/kill-switch", body)

    def simulate(
        self,
        intents: Sequence[Mapping[str, Any]],
        *,
        policies: Optional[Sequence[Mapping[str, Any]]] = None,
        policy_refs: Optional[Sequence[str]] = None,
        decisions_from_run: Optional[str] = None,
    ) -> dict[str, Any]:
        """Evaluate intents against documents or released refs; no permit, no side effect.

        Intents are posted as given, including ``families``, ``family_tiers`` and
        ``delegation_chain``; the gateway re-derives the action families from
        the intent and can only add one, so posting an empty ``families`` list
        does not drop a family. A classified call that no rule names the family
        of gets the family default (``family_default`` in ``reason_codes``).
        """
        body = _simulate_body(intents, policies, policy_refs, decisions_from_run)
        return self._request("POST", "/v1/protect/simulate", body)

    async def asimulate(
        self,
        intents: Sequence[Mapping[str, Any]],
        *,
        policies: Optional[Sequence[Mapping[str, Any]]] = None,
        policy_refs: Optional[Sequence[str]] = None,
        decisions_from_run: Optional[str] = None,
    ) -> dict[str, Any]:
        """Async counterpart of :meth:`simulate`."""
        body = _simulate_body(intents, policies, policy_refs, decisions_from_run)
        return await self._arequest("POST", "/v1/protect/simulate", body)

    def settings(self) -> ProtectSettings:
        """Organization protect settings (mandatory flag, entitlement, revision)."""
        return ProtectSettings.model_validate(self._request("GET", "/v1/protect/settings"))

    async def asettings(self) -> ProtectSettings:
        """Async counterpart of :meth:`settings`."""
        return ProtectSettings.model_validate(await self._arequest("GET", "/v1/protect/settings"))

    def set_mandatory(
        self, protect_mandatory: bool, *, expected_revision: Optional[int] = None
    ) -> ProtectSettings:
        """Turn mandatory protection on or off; ``expected_revision`` guards a concurrent edit."""
        body = _settings_body(protect_mandatory, expected_revision)
        return ProtectSettings.model_validate(self._request("PUT", "/v1/protect/settings", body))

    async def aset_mandatory(
        self, protect_mandatory: bool, *, expected_revision: Optional[int] = None
    ) -> ProtectSettings:
        """Async counterpart of :meth:`set_mandatory`."""
        body = _settings_body(protect_mandatory, expected_revision)
        return ProtectSettings.model_validate(
            await self._arequest("PUT", "/v1/protect/settings", body)
        )

    def coverage(self) -> dict[str, Any]:
        """Coverage matrix per action family, plus the organization block (entitlement, exemptions)."""
        return self._request("GET", "/v1/protect/coverage")

    async def acoverage(self) -> dict[str, Any]:
        """Async counterpart of :meth:`coverage`."""
        return await self._arequest("GET", "/v1/protect/coverage")

    def metrics_summary(self) -> dict[str, Any]:
        """Decision counts by outcome, pending approvals and latency percentiles."""
        return self._request("GET", "/v1/protect/metrics/summary")

    async def ametrics_summary(self) -> dict[str, Any]:
        """Async counterpart of :meth:`metrics_summary`."""
        return await self._arequest("GET", "/v1/protect/metrics/summary")


__all__ = [
    "ApprovalsResource",
    "BindingsResource",
    "DecisionsResource",
    "DelegationsResource",
    "ExemptionsResource",
    "PoliciesResource",
    "ProtectResource",
    "RestrictionsResource",
]
