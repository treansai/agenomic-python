"""``client.vault.runtime``: what an agent runtime may do with its runtime-identity token.

An agent holds an authorization, never a credential. It asks for one logical
action (:meth:`RuntimeResource.execute`), reads the stored outcome of an
action, and requests or delegates grants that a human must approve. The
primary entry point is ``client.tools.execute``, which delegates here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from pydantic import JsonValue

from agenomic.vault import ops
from agenomic.vault.errors import ReplayUnsupported
from agenomic.vault.execution import ExecuteRequest, normalize_action_id, settle
from agenomic.vault.models import ExecuteResult, ExecutionStatus, Grant
from agenomic.vault.transport import Namespace, asend, send


class RuntimeResource(Namespace):
    """Runtime plane of the vault. Needs ``Client(runtime_token=...)`` or a replay set.

    Example:
        >>> from agenomic import Client
        >>> from agenomic.vault import ReplayFixture, ReplayOutcome, VaultReplay
        >>> replay = VaultReplay([ReplayFixture(fixture_id="f", tool="t", binding="b",
        ...     outcome=ReplayOutcome(result={"ok": True}))])
        >>> Client(vault_replay=replay).vault.runtime.execute(tool="t", binding="b").result
        {'ok': True}
    """

    def _replay_guard(self, operation: str) -> None:
        if self._client._vault_replay is not None:
            raise ReplayUnsupported(
                "replay_unsupported",
                f"{operation} has no replay counterpart and is not sent live while replaying",
                0,
            )

    def execute(
        self,
        *,
        tool: str,
        binding: str,
        arguments: Optional[Mapping[str, JsonValue]] = None,
        action_id: Optional[str] = None,
        deadline_ms: Optional[int] = None,
    ) -> ExecuteResult:
        """Execute one authorized business action; the credential never reaches this process.

        ``action_id`` identifies the logical action (a UUID is generated when
        absent and exposed on the result and on every error). Technical retries
        reuse it, so the server de-duplicates; an ``outcome_unknown`` outcome is
        never retried and raises :class:`VaultOutcomeUnknown`.
        """
        request = ExecuteRequest.of(tool, binding, arguments, action_id, deadline_ms)
        if self._client._vault_replay is not None:
            return self._client._vault_replay.execute(request)
        return settle(send(self._client, request.call()), request.action_id)

    async def aexecute(
        self,
        *,
        tool: str,
        binding: str,
        arguments: Optional[Mapping[str, JsonValue]] = None,
        action_id: Optional[str] = None,
        deadline_ms: Optional[int] = None,
    ) -> ExecuteResult:
        """Async counterpart of :meth:`execute`."""
        request = ExecuteRequest.of(tool, binding, arguments, action_id, deadline_ms)
        if self._client._vault_replay is not None:
            return self._client._vault_replay.execute(request)
        return settle(await asend(self._client, request.call()), request.action_id)

    def get_execution(self, action_id: str) -> ExecutionStatus:
        """The stored state of one of this identity's executions; reading never raises on a state."""
        ident = normalize_action_id(action_id)
        if self._client._vault_replay is not None:
            return self._client._vault_replay.status(ident)
        return self._run(ops.runtime_execution_get(ident))

    async def aget_execution(self, action_id: str) -> ExecutionStatus:
        """Async counterpart of :meth:`get_execution`."""
        ident = normalize_action_id(action_id)
        if self._client._vault_replay is not None:
            return self._client._vault_replay.status(ident)
        return await self._arun(ops.runtime_execution_get(ident))

    def list_grants(
        self, *, binding_id: Optional[str] = None, state: Optional[str] = None
    ) -> list[Grant]:
        """The grants of this identity."""
        self._replay_guard("list_grants")
        return self._run(ops.runtime_grants_list(binding_id, state))

    async def alist_grants(
        self, *, binding_id: Optional[str] = None, state: Optional[str] = None
    ) -> list[Grant]:
        """Async counterpart of :meth:`list_grants`."""
        self._replay_guard("list_grants")
        return await self._arun(ops.runtime_grants_list(binding_id, state))

    def request_grant(
        self,
        *,
        binding_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
        version: Optional[int] = None,
    ) -> Grant:
        """Ask for a grant on a binding; a human with ``grant.approve`` must decide it."""
        self._replay_guard("request_grant")
        return self._run(
            ops.runtime_grant_request(binding_id, max_uses, ttl_seconds, reason, version)
        )

    async def arequest_grant(
        self,
        *,
        binding_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
        version: Optional[int] = None,
    ) -> Grant:
        """Async counterpart of :meth:`request_grant`."""
        self._replay_guard("request_grant")
        return await self._arun(
            ops.runtime_grant_request(binding_id, max_uses, ttl_seconds, reason, version)
        )

    def delegate_grant(
        self,
        grant_id: str,
        *,
        delegate_agent_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
    ) -> Grant:
        """Delegate a narrower slice of an approved grant to another agent of the same environment."""
        self._replay_guard("delegate_grant")
        return self._run(
            ops.runtime_grant_delegate(grant_id, delegate_agent_id, max_uses, ttl_seconds, reason)
        )

    async def adelegate_grant(
        self,
        grant_id: str,
        *,
        delegate_agent_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
    ) -> Grant:
        """Async counterpart of :meth:`delegate_grant`."""
        self._replay_guard("delegate_grant")
        return await self._arun(
            ops.runtime_grant_delegate(grant_id, delegate_agent_id, max_uses, ttl_seconds, reason)
        )
