"""``client.benchmarks``: plan, preflight, launch and follow RMP benchmark runs.

Cloud-only: every call hits ``/v1/rmp/benchmarks``. Starting an RMP session
(``client.rmp.start``) never launches a benchmark; launching is the explicit
``launch`` call on a plan whose preflight passed. In local mode the methods
raise :class:`~agenomic.exceptions.CloudError` instead of pretending.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from agenomic.exceptions import CloudError

#: Version stamped on benchmark wire types.
BENCHMARKS_SPEC_VERSION = "agenomic.rmp.benchmarks/v0.1"

_ENCODE = "utf-8"


def _q(value: str) -> str:
    return quote(value, safe="")


class BenchmarksResource:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _require_cloud(self) -> None:
        if not getattr(self._client, "is_cloud", False):
            raise CloudError("benchmarks need a cloud client (base_url); nothing runs locally")

    # ── Catalogue ───────────────────────────────────────────────────────

    def catalog(
        self, *, agent: Optional[str] = None, release_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """The five benchmark cards with adapter availability and, when an
        agent is given, bridge compatibility."""
        self._require_cloud()
        query = []
        if agent:
            query.append(f"agent_id={_q(agent)}")
        if release_id:
            query.append(f"release_id={_q(release_id)}")
        path = "/v1/rmp/benchmarks/catalog" + (f"?{'&'.join(query)}" if query else "")
        response = self._client._get(path)
        return list(response.get("benchmarks", [])) if isinstance(response, dict) else []

    def card(self, benchmark_id: str) -> dict[str, Any]:
        self._require_cloud()
        card: dict[str, Any] = self._client._get(f"/v1/rmp/benchmarks/catalog/{_q(benchmark_id)}")
        return card

    # ── Plans ───────────────────────────────────────────────────────────

    def create_plan(
        self,
        session_id: str,
        selections: list[dict[str, Any]],
        *,
        target: str = "customer_agent",
        budget: Optional[dict[str, Any]] = None,
        baseline_plan_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a draft plan under an RMP session. ``selections`` are
        ``{"benchmark_id", "benchmark_version", "scope", "profile", ...}``
        dicts as described in ``docs/benchmarks.md``."""
        self._require_cloud()
        body: dict[str, Any] = {"target": target, "selections": selections, "budget": budget or {}}
        if baseline_plan_id:
            body["baseline_plan_id"] = baseline_plan_id
        response = self._client._post(f"/v1/rmp/sessions/{_q(session_id)}/benchmarks/plans", body)
        return _unwrap(response, "plan")

    def list_plans(self, session_id: str) -> list[dict[str, Any]]:
        self._require_cloud()
        response = self._client._get(f"/v1/rmp/sessions/{_q(session_id)}/benchmarks/plans")
        return list(response.get("plans", [])) if isinstance(response, dict) else []

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        """Plan, its runs and the overall verdict (``view`` envelope)."""
        self._require_cloud()
        response = self._client._get(f"/v1/rmp/benchmarks/plans/{_q(plan_id)}")
        return _unwrap(response, "view")

    def update_plan(
        self,
        plan_id: str,
        selections: list[dict[str, Any]],
        *,
        budget: Optional[dict[str, Any]] = None,
        target: Optional[str] = None,
    ) -> dict[str, Any]:
        self._require_cloud()
        body: dict[str, Any] = {"selections": selections}
        if budget is not None:
            body["budget"] = budget
        if target is not None:
            body["target"] = target
        response = self._client._put(f"/v1/rmp/benchmarks/plans/{_q(plan_id)}", body)
        return _unwrap(response, "plan")

    def preflight(self, plan_id: str) -> dict[str, Any]:
        """Server-side checks. The returned plan carries ``preflight`` with
        every check, its corrective action and the frozen trial counts."""
        self._require_cloud()
        response = self._client._post(f"/v1/rmp/benchmarks/plans/{_q(plan_id)}/preflight", {})
        return _unwrap(response, "plan")

    def launch(self, plan_id: str) -> dict[str, Any]:
        """Idempotent asynchronous launch: returns ``{plan, runs, already_launched}``."""
        self._require_cloud()
        outcome: dict[str, Any] = self._client._post(
            f"/v1/rmp/benchmarks/plans/{_q(plan_id)}/launch", {}
        )
        return outcome

    def cancel_plan(self, plan_id: str) -> dict[str, Any]:
        self._require_cloud()
        response = self._client._post(f"/v1/rmp/benchmarks/plans/{_q(plan_id)}/cancel", {})
        return _unwrap(response, "view")

    def compare(self, plan_id: str, baseline_plan_id: str) -> dict[str, Any]:
        self._require_cloud()
        response = self._client._get(
            f"/v1/rmp/benchmarks/plans/{_q(plan_id)}/compare?baseline={_q(baseline_plan_id)}"
        )
        return _unwrap(response, "comparison")

    # ── Runs ────────────────────────────────────────────────────────────

    def list_runs(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        self._require_cloud()
        response = self._client._get(
            f"/v1/rmp/benchmarks/runs?session_id={_q(session_id)}&limit={int(limit)}"
        )
        return list(response.get("runs", [])) if isinstance(response, dict) else []

    def get_run(self, run_id: str, *, after: int = 0, limit: int = 200) -> dict[str, Any]:
        """Run, trials and events after the given event cursor."""
        self._require_cloud()
        detail: dict[str, Any] = self._client._get(
            f"/v1/rmp/benchmarks/runs/{_q(run_id)}?after={int(after)}&limit={int(limit)}"
        )
        return detail

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        self._require_cloud()
        response = self._client._post(f"/v1/rmp/benchmarks/runs/{_q(run_id)}/cancel", {})
        return _unwrap(response, "run")

    # ── Protect policies ────────────────────────────────────────────────

    def list_policies(
        self, *, agent: Optional[str] = None, plan_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        self._require_cloud()
        query = []
        if agent:
            query.append(f"agent_id={_q(agent)}")
        if plan_id:
            query.append(f"plan_id={_q(plan_id)}")
        path = "/v1/rmp/benchmarks/policies" + (f"?{'&'.join(query)}" if query else "")
        response = self._client._get(path)
        return list(response.get("policies", [])) if isinstance(response, dict) else []

    def propose_policy(self, proposal: dict[str, Any]) -> dict[str, Any]:
        self._require_cloud()
        response = self._client._post("/v1/rmp/benchmarks/policies", proposal)
        return _unwrap(response, "policy")

    def decide_policy(
        self,
        proposal_id: str,
        to: str,
        *,
        manifest_hash: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Move a proposal along proposed -> reviewed -> shadow -> approved -> active
        (or reject / roll back). Approval and activation must cite the exact
        manifest hash the proposal was built from."""
        self._require_cloud()
        body: dict[str, Any] = {"to": to}
        if manifest_hash is not None:
            body["manifest_hash"] = manifest_hash
        if note is not None:
            body["note"] = note
        response = self._client._post(f"/v1/rmp/benchmarks/policies/{_q(proposal_id)}/decide", body)
        return _unwrap(response, "policy")


def _unwrap(response: Any, key: str) -> dict[str, Any]:
    if isinstance(response, dict) and isinstance(response.get(key), dict):
        inner: dict[str, Any] = response[key]
        return inner
    return dict(response) if isinstance(response, dict) else {}
