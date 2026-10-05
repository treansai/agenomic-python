"""``client.vault``: the administrative and metadata plane of Agents Vault.

Authenticated with the Agenomic API key. Secret values go in through
:class:`Sensitive` and are write-only: no method of this module returns one.
Operations that need a human session (approving a grant or a binding version,
lifting a kill switch, settling an unknown outcome) are exposed as the API
defines them; the server refuses them for an API key with
``VaultPermissionDenied``. Async variants carry the ``a`` prefix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from agenomic.vault import ops
from agenomic.vault.models import (
    Binding,
    BindingDetail,
    BindingVersion,
    ExecutionStatus,
    ExecutionSummary,
    Grant,
    IssuedIdentity,
    Provider,
    Receipt,
    Revocation,
    RotationJob,
    RuntimeIdentity,
    Secret,
    SecretDetail,
    VaultStatus,
)
from agenomic.vault.runtime import RuntimeResource
from agenomic.vault.transport import Namespace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agenomic._client import Client

ProviderList = list[Provider]
SecretList = list[Secret]
RotationList = list[RotationJob]
BindingList = list[Binding]
GrantList = list[Grant]
IdentityList = list[RuntimeIdentity]
ExecutionList = list[ExecutionSummary]
ReceiptList = list[Receipt]
RevocationList = list[Revocation]
KillTarget = Literal["binding", "agent", "session", "workspace", "provider", "secret", "grant"]


class ProvidersResource(Namespace):
    """``client.vault.providers``: the vault backends the executor reaches."""

    def list(self) -> ProviderList:
        return self._run(ops.providers_list())

    async def alist(self) -> ProviderList:
        return await self._arun(ops.providers_list())

    def get(self, provider_id: str) -> Provider:
        return self._run(ops.provider_get(provider_id))

    async def aget(self, provider_id: str) -> Provider:
        return await self._arun(ops.provider_get(provider_id))

    def create(self, *, name: str, mode: str, descriptor: ops.Descriptor) -> Provider:
        """Connect a provider (``mode``: ``managed``, ``customer_hosted`` or ``byov``)."""
        return self._run(ops.provider_create(name, mode, descriptor))

    async def acreate(self, *, name: str, mode: str, descriptor: ops.Descriptor) -> Provider:
        return await self._arun(ops.provider_create(name, mode, descriptor))

    def health(self, provider_id: str) -> Provider:
        """Probe a managed provider and return it with its refreshed health."""
        return self._run(ops.provider_health(provider_id))

    async def ahealth(self, provider_id: str) -> Provider:
        return await self._arun(ops.provider_health(provider_id))

    def set_state(self, provider_id: str, *, state: str, reason: str) -> Provider:
        """Disable or revoke a provider (``active``, ``disabled``, ``revoked``)."""
        return self._run(ops.provider_state(provider_id, state, reason))

    async def aset_state(self, provider_id: str, *, state: str, reason: str) -> Provider:
        return await self._arun(ops.provider_state(provider_id, state, reason))


class SecretsResource(Namespace):
    """``client.vault.secrets``: metadata, write-only values, references and revocation.

    Example:
        >>> from agenomic.vault import Sensitive
        >>> repr(Sensitive("a value that must never be printed"))
        "Sensitive('**********')"
    """

    def list(
        self,
        *,
        environment: Optional[str] = None,
        provider_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> SecretList:
        """Secret metadata. There is no value in it."""
        return self._run(ops.secrets_list(environment, provider_id, state))

    async def alist(
        self,
        *,
        environment: Optional[str] = None,
        provider_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> SecretList:
        return await self._arun(ops.secrets_list(environment, provider_id, state))

    def get(self, secret_id: str) -> SecretDetail:
        """Metadata and versions of one secret."""
        return self._run(ops.secret_get(secret_id))

    async def aget(self, secret_id: str) -> SecretDetail:
        return await self._arun(ops.secret_get(secret_id))

    def create(
        self,
        *,
        environment: str,
        name: str,
        secret_type: str,
        provider_id: str,
        value: ops.SecretValue,
        classification: Optional[str] = None,
        provider_ref: Optional[str] = None,
    ) -> SecretDetail:
        """Import a secret value. ``value`` must be a :class:`Sensitive`; it is never returned."""
        return self._run(
            ops.secret_create(
                environment, name, secret_type, provider_id, value, classification, provider_ref
            )
        )

    async def acreate(
        self,
        *,
        environment: str,
        name: str,
        secret_type: str,
        provider_id: str,
        value: ops.SecretValue,
        classification: Optional[str] = None,
        provider_ref: Optional[str] = None,
    ) -> SecretDetail:
        return await self._arun(
            ops.secret_create(
                environment, name, secret_type, provider_id, value, classification, provider_ref
            )
        )

    def register_reference(
        self,
        *,
        environment: str,
        name: str,
        secret_type: str,
        provider_id: str,
        provider_ref: str,
        provider_version: str,
        classification: Optional[str] = None,
    ) -> SecretDetail:
        """Register a secret that lives in a customer vault (no value is sent)."""
        return self._run(
            ops.secret_reference(
                environment,
                name,
                secret_type,
                provider_id,
                provider_ref,
                provider_version,
                classification,
            )
        )

    async def aregister_reference(
        self,
        *,
        environment: str,
        name: str,
        secret_type: str,
        provider_id: str,
        provider_ref: str,
        provider_version: str,
        classification: Optional[str] = None,
    ) -> SecretDetail:
        return await self._arun(
            ops.secret_reference(
                environment,
                name,
                secret_type,
                provider_id,
                provider_ref,
                provider_version,
                classification,
            )
        )

    def add_version(self, secret_id: str, *, value: ops.SecretValue) -> SecretDetail:
        """Add a value version directly (write-only). Prefer :meth:`rotate`, which verifies it."""
        return self._run(ops.secret_add_version(secret_id, value))

    async def aadd_version(self, secret_id: str, *, value: ops.SecretValue) -> SecretDetail:
        return await self._arun(ops.secret_add_version(secret_id, value))

    def rotate(
        self, secret_id: str, *, value: ops.SecretValue, overlap_seconds: Optional[int] = None
    ) -> RotationJob:
        """Start a rotation: the new value is verified and kept pending until activated.

        Same operation as ``client.vault.rotations.start``.
        """
        return self._run(ops.rotation_start(secret_id, value, overlap_seconds))

    async def arotate(
        self, secret_id: str, *, value: ops.SecretValue, overlap_seconds: Optional[int] = None
    ) -> RotationJob:
        return await self._arun(ops.rotation_start(secret_id, value, overlap_seconds))

    def revoke(self, secret_id: str, *, reason: str) -> Revocation:
        """Revoke a secret; Agenomic and provider state are reported separately."""
        return self._run(ops.secret_revoke(secret_id, reason))

    async def arevoke(self, secret_id: str, *, reason: str) -> Revocation:
        return await self._arun(ops.secret_revoke(secret_id, reason))


class RotationsResource(Namespace):
    """``client.vault.rotations``: verified rotations with an overlap and a rollback window."""

    def start(
        self, secret_id: str, *, value: ops.SecretValue, overlap_seconds: Optional[int] = None
    ) -> RotationJob:
        """Write the new version, verify it by read-back, keep it pending."""
        return self._run(ops.rotation_start(secret_id, value, overlap_seconds))

    async def astart(
        self, secret_id: str, *, value: ops.SecretValue, overlap_seconds: Optional[int] = None
    ) -> RotationJob:
        return await self._arun(ops.rotation_start(secret_id, value, overlap_seconds))

    def list(self, *, secret_id: Optional[str] = None, state: Optional[str] = None) -> RotationList:
        return self._run(ops.rotations_list(secret_id, state))

    async def alist(
        self, *, secret_id: Optional[str] = None, state: Optional[str] = None
    ) -> RotationList:
        return await self._arun(ops.rotations_list(secret_id, state))

    def get(self, rotation_id: str) -> RotationJob:
        return self._run(ops.rotation_get(rotation_id))

    async def aget(self, rotation_id: str) -> RotationJob:
        return await self._arun(ops.rotation_get(rotation_id))

    def activate(self, rotation_id: str) -> RotationJob:
        """Activate the prepared version; the previous one stays for the rollback window."""
        return self._run(ops.rotation_activate(rotation_id))

    async def aactivate(self, rotation_id: str) -> RotationJob:
        return await self._arun(ops.rotation_activate(rotation_id))

    def rollback(self, rotation_id: str, *, reason: str) -> RotationJob:
        """Return to the previous version and retire the new one."""
        return self._run(ops.rotation_rollback(rotation_id, reason))

    async def arollback(self, rotation_id: str, *, reason: str) -> RotationJob:
        return await self._arun(ops.rotation_rollback(rotation_id, reason))


class BindingsResource(Namespace):
    """``client.vault.bindings``: what an agent may do with a secret, and the review of it.

    A binding version goes ``draft`` -> ``submit`` -> ``approve`` -> ``activate``.
    """

    def list(
        self,
        *,
        environment: Optional[str] = None,
        agent_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> BindingList:
        return self._run(ops.bindings_list(environment, agent_id, state))

    async def alist(
        self,
        *,
        environment: Optional[str] = None,
        agent_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> BindingList:
        return await self._arun(ops.bindings_list(environment, agent_id, state))

    def get(self, binding_id: str) -> BindingDetail:
        return self._run(ops.binding_get(binding_id))

    async def aget(self, binding_id: str) -> BindingDetail:
        return await self._arun(ops.binding_get(binding_id))

    def create(
        self, *, environment: str, name: str, agent_id: str, tool_name: str, content: ops.Content
    ) -> BindingDetail:
        """Create a binding with its first draft version."""
        return self._run(ops.binding_create(environment, name, agent_id, tool_name, content))

    async def acreate(
        self, *, environment: str, name: str, agent_id: str, tool_name: str, content: ops.Content
    ) -> BindingDetail:
        return await self._arun(ops.binding_create(environment, name, agent_id, tool_name, content))

    def propose_version(self, binding_id: str, *, content: ops.Content) -> BindingVersion:
        return self._run(ops.binding_propose(binding_id, content))

    async def apropose_version(self, binding_id: str, *, content: ops.Content) -> BindingVersion:
        return await self._arun(ops.binding_propose(binding_id, content))

    def submit(self, binding_id: str, version: int) -> BindingVersion:
        """Submit a draft version for review."""
        return self._run(ops.binding_submit(binding_id, version))

    async def asubmit(self, binding_id: str, version: int) -> BindingVersion:
        return await self._arun(ops.binding_submit(binding_id, version))

    def approve(self, binding_id: str, version: int) -> BindingVersion:
        """Approve a submitted version. Needs a human session that is not the proposer."""
        return self._run(ops.binding_decide(binding_id, version, True))

    async def aapprove(self, binding_id: str, version: int) -> BindingVersion:
        return await self._arun(ops.binding_decide(binding_id, version, True))

    def reject(self, binding_id: str, version: int) -> BindingVersion:
        """Reject a submitted version."""
        return self._run(ops.binding_decide(binding_id, version, False))

    async def areject(self, binding_id: str, version: int) -> BindingVersion:
        return await self._arun(ops.binding_decide(binding_id, version, False))

    def activate(self, binding_id: str, version: int) -> BindingDetail:
        """Activate an approved version."""
        return self._run(ops.binding_activate(binding_id, version))

    async def aactivate(self, binding_id: str, version: int) -> BindingDetail:
        return await self._arun(ops.binding_activate(binding_id, version))

    def revoke(self, binding_id: str, *, reason: str) -> Revocation:
        return self._run(ops.binding_revoke(binding_id, reason))

    async def arevoke(self, binding_id: str, *, reason: str) -> Revocation:
        return await self._arun(ops.binding_revoke(binding_id, reason))


class GrantsResource(Namespace):
    """``client.vault.grants``: bounded authority to perform logical actions on a binding."""

    def list(
        self,
        *,
        binding_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> GrantList:
        return self._run(ops.grants_list(binding_id, agent_id, state))

    async def alist(
        self,
        *,
        binding_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> GrantList:
        return await self._arun(ops.grants_list(binding_id, agent_id, state))

    def request(
        self,
        *,
        binding_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
        version: Optional[int] = None,
    ) -> Grant:
        """Request a grant on behalf of an agent; a different human must approve it."""
        return self._run(ops.grant_request(binding_id, max_uses, ttl_seconds, reason, version))

    async def arequest(
        self,
        *,
        binding_id: str,
        max_uses: int,
        ttl_seconds: int,
        reason: str,
        version: Optional[int] = None,
    ) -> Grant:
        return await self._arun(
            ops.grant_request(binding_id, max_uses, ttl_seconds, reason, version)
        )

    def approve(self, grant_id: str) -> Grant:
        """Approve a grant. Needs a human session that is not the requester."""
        return self._run(ops.grant_decide(grant_id, True))

    async def aapprove(self, grant_id: str) -> Grant:
        return await self._arun(ops.grant_decide(grant_id, True))

    def deny(self, grant_id: str) -> Grant:
        return self._run(ops.grant_decide(grant_id, False))

    async def adeny(self, grant_id: str) -> Grant:
        return await self._arun(ops.grant_decide(grant_id, False))

    def revoke(self, grant_id: str, *, reason: str) -> Revocation:
        return self._run(ops.grant_revoke(grant_id, reason))

    async def arevoke(self, grant_id: str, *, reason: str) -> Revocation:
        return await self._arun(ops.grant_revoke(grant_id, reason))


class RuntimeIdentitiesResource(Namespace):
    """``client.vault.runtime_identities``: enrollment tokens for agent runtimes."""

    def list(self) -> IdentityList:
        return self._run(ops.identities_list())

    async def alist(self) -> IdentityList:
        return await self._arun(ops.identities_list())

    def issue(
        self,
        *,
        environment: str,
        agent_id: str,
        label: str,
        ttl_seconds: Optional[int] = None,
        declared_release: Optional[str] = None,
        declared_genome_digest: Optional[str] = None,
    ) -> IssuedIdentity:
        """Enroll an identity. The token is shown once: take it with ``issued.token.consume()``
        or pass it straight to a runtime client with ``issued.runtime_client(base_url)``."""
        return self._run(
            ops.identity_issue(
                environment,
                agent_id,
                label,
                ttl_seconds,
                declared_release,
                declared_genome_digest,
            )
        )

    async def aissue(
        self,
        *,
        environment: str,
        agent_id: str,
        label: str,
        ttl_seconds: Optional[int] = None,
        declared_release: Optional[str] = None,
        declared_genome_digest: Optional[str] = None,
    ) -> IssuedIdentity:
        return await self._arun(
            ops.identity_issue(
                environment,
                agent_id,
                label,
                ttl_seconds,
                declared_release,
                declared_genome_digest,
            )
        )

    def revoke(self, identity_id: str, *, reason: str) -> RuntimeIdentity:
        return self._run(ops.identity_revoke(identity_id, reason))

    async def arevoke(self, identity_id: str, *, reason: str) -> RuntimeIdentity:
        return await self._arun(ops.identity_revoke(identity_id, reason))


class ExecutionsResource(Namespace):
    """``client.vault.executions``: evidence of what ran, and settlement of unknown outcomes."""

    def list(self, *, state: Optional[str] = None, limit: Optional[int] = None) -> ExecutionList:
        return self._run(ops.executions_list(state, limit))

    async def alist(
        self, *, state: Optional[str] = None, limit: Optional[int] = None
    ) -> ExecutionList:
        return await self._arun(ops.executions_list(state, limit))

    def get(self, action_id: str) -> ExecutionStatus:
        """One execution by ``action_id``; reading never raises on a state."""
        return self._run(ops.execution_get(action_id))

    async def aget(self, action_id: str) -> ExecutionStatus:
        return await self._arun(ops.execution_get(action_id))

    def resolve(
        self, action_id: str, *, resolution: Literal["applied", "not_applied"], note: str
    ) -> ExecutionStatus:
        """Settle an ``outcome_unknown`` execution after verifying the destination.

        Records what the operator established; the action is never re-sent.
        """
        return self._run(ops.execution_resolve(action_id, resolution, note))

    async def aresolve(
        self, action_id: str, *, resolution: Literal["applied", "not_applied"], note: str
    ) -> ExecutionStatus:
        return await self._arun(ops.execution_resolve(action_id, resolution, note))


class ReceiptsResource(Namespace):
    """``client.vault.receipts``: append-only usage receipts, two phases per execution."""

    def list(self, *, action_id: Optional[str] = None, limit: Optional[int] = None) -> ReceiptList:
        return self._run(ops.receipts_list(action_id, limit))

    async def alist(
        self, *, action_id: Optional[str] = None, limit: Optional[int] = None
    ) -> ReceiptList:
        return await self._arun(ops.receipts_list(action_id, limit))


class RevocationsResource(Namespace):
    """``client.vault.revocations``: revocations with Agenomic and provider state."""

    def list(self) -> RevocationList:
        return self._run(ops.revocations_list())

    async def alist(self) -> RevocationList:
        return await self._arun(ops.revocations_list())

    def retry(self, revocation_id: str) -> Revocation:
        """Re-run the provider-side step of a revocation now."""
        return self._run(ops.revocation_retry(revocation_id))

    async def aretry(self, revocation_id: str) -> Revocation:
        return await self._arun(ops.revocation_retry(revocation_id))

    def lift(self, revocation_id: str, *, reason: str) -> Revocation:
        """Lift a kill switch on a workspace, agent, provider or binding (a human action)."""
        return self._run(ops.revocation_lift(revocation_id, reason))

    async def alift(self, revocation_id: str, *, reason: str) -> Revocation:
        return await self._arun(ops.revocation_lift(revocation_id, reason))


class VaultResource(Namespace):
    """``client.vault``: Agents Vault, an optional commercial module of Agenomic Cloud/Enterprise.

    Needs the Agents Vault add-on; ``status()`` stays readable when it is locked.

    Example:
        >>> from agenomic import Client
        >>> client = Client(api_key="key", base_url="https://cloud.example")
        >>> sorted(n for n in ("secrets", "bindings", "grants") if hasattr(client.vault, n))
        ['bindings', 'grants', 'secrets']
    """

    def __init__(self, client: Client) -> None:
        super().__init__(client)
        self.providers = ProvidersResource(client)
        self.secrets = SecretsResource(client)
        self.rotations = RotationsResource(client)
        self.bindings = BindingsResource(client)
        self.grants = GrantsResource(client)
        self.runtime_identities = RuntimeIdentitiesResource(client)
        self.executions = ExecutionsResource(client)
        self.receipts = ReceiptsResource(client)
        self.revocations = RevocationsResource(client)
        #: The agent plane: execute, executions, grants and delegations with a runtime token.
        self.runtime = RuntimeResource(client)

    def status(self) -> VaultStatus:
        """Installation and entitlement state as the server reports it (readable when locked)."""
        return self._run(ops.status())

    async def astatus(self) -> VaultStatus:
        return await self._arun(ops.status())

    def kill_switch(self, *, target_kind: KillTarget, target_id: str, reason: str) -> Revocation:
        """Engage a kill switch by binding, agent, session, workspace, provider, secret or grant."""
        return self._run(ops.kill_switch(target_kind, target_id, reason))

    async def akill_switch(
        self, *, target_kind: KillTarget, target_id: str, reason: str
    ) -> Revocation:
        return await self._arun(ops.kill_switch(target_kind, target_id, reason))
