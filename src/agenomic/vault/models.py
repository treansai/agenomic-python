"""Typed models of the Agents Vault wire contract.

Response models ignore fields they do not know, so a value the server should
never send cannot land in an SDK object. Request models forbid unknown fields:
a misspelt constraint in a binding is an error, never a silently dropped rule.
No model has a field that can hold a secret value; the only credential-like
field is the one-time runtime token, which is an :class:`IssuedToken`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agenomic.vault.sensitive import IssuedToken

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agenomic._client import Client


class _Response(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CapabilityState(_Response):
    """Entitlement evaluation reported by the server (the SDK never computes it)."""

    id: str = "agents_vault"
    enabled: bool = False
    reason: Optional[str] = None
    required_plan: Optional[str] = None


class VaultUsage(_Response):
    providers: int = 0
    secrets: int = 0
    bindings: int = 0
    runtime_identities: int = 0


class VaultStatus(_Response):
    """Installation and entitlement state; readable even when the add-on is locked.

    Example:
        >>> status = VaultStatus(installed=True, entitled=False,
        ...     capability=CapabilityState(enabled=False, reason="not_entitled", required_plan="cloud"))
        >>> (status.locked, status.upgrade_hint)
        (True, True)
    """

    installed: bool = False
    entitled: bool = False
    capability: CapabilityState = Field(default_factory=CapabilityState)
    permissions: list[str] = Field(default_factory=list)
    usage: VaultUsage = Field(default_factory=VaultUsage)
    limitations: list[str] = Field(default_factory=list)

    @property
    def locked(self) -> bool:
        """True when the server reports the add-on as not entitled."""
        return not self.entitled

    @property
    def upgrade_hint(self) -> bool:
        """True when the server's reason says an upgrade path exists."""
        return self.locked and self.capability.reason in ("not_entitled", "not_in_edition")


class Provider(_Response):
    id: str
    name: str = ""
    kind: str = ""
    mode: str = ""
    location: str = ""
    auth_method: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    health: str = "unknown"
    health_checked_at: Optional[datetime] = None
    state: str = "active"
    created_at: Optional[datetime] = None


class Secret(_Response):
    """Secret metadata. There is no value field: values are write-only."""

    id: str
    environment: str = ""
    name: str = ""
    secret_type: str = ""
    classification: str = ""
    owner_user_id: Optional[str] = None
    provider_id: Optional[str] = None
    provider_ref: Optional[str] = None
    state: str = ""
    current_version_id: Optional[str] = None
    revision: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SecretVersion(_Response):
    id: str
    provider_version: str = ""
    state: str = ""
    provenance: str = ""
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None


class SecretDetail(_Response):
    secret: Secret
    versions: list[SecretVersion] = Field(default_factory=list)


class Binding(_Response):
    id: str
    environment: str = ""
    name: str = ""
    agent_id: str = ""
    tool_name: str = ""
    state: str = ""
    active_version: Optional[int] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BindingVersion(_Response):
    version: int
    state: str = ""
    digest: str = ""
    content: dict[str, JsonValue] = Field(default_factory=dict)
    proposed_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class BindingDetail(_Response):
    binding: Binding
    versions: list[BindingVersion] = Field(default_factory=list)


class Grant(_Response):
    id: str
    binding_id: str = ""
    binding_version: int = 0
    environment: str = ""
    agent_id: str = ""
    state: str = ""
    max_uses: int = 0
    uses: int = 0
    expires_at: Optional[datetime] = None
    requested_by_user: Optional[str] = None
    requested_by_identity: Optional[str] = None
    requested_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    reason: str = ""
    parent_grant_id: Optional[str] = None
    depth: int = 0


class Revocation(_Response):
    """A revocation with the Agenomic and the provider state reported separately."""

    id: str
    target_kind: str = ""
    target_id: str = ""
    agenomic_state: str = ""
    provider_state: str = ""
    reason: str = ""
    requested_by: Optional[str] = None
    requested_at: Optional[datetime] = None
    provider_confirmed_at: Optional[datetime] = None
    attempts: int = 0
    next_attempt_at: Optional[datetime] = None
    last_error: Optional[str] = None
    lifted_at: Optional[datetime] = None
    lifted_by: Optional[str] = None
    lift_reason: Optional[str] = None


class RotationJob(_Response):
    id: str
    secret_id: str = ""
    new_version_id: Optional[str] = None
    old_version_id: Optional[str] = None
    retire_version_id: Optional[str] = None
    state: str = ""
    direction: str = ""
    overlap_seconds: int = 0
    retire_after: Optional[datetime] = None
    attempts: int = 0
    next_attempt_at: Optional[datetime] = None
    last_error: Optional[str] = None
    requested_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RuntimeIdentity(_Response):
    id: str
    environment: str = ""
    agent_id: str = ""
    label: str = ""
    declared_release: Optional[str] = None
    declared_genome_digest: Optional[str] = None
    assurance: str = ""
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class IssuedIdentity(_Response):
    """A freshly enrolled runtime identity with its one-time token.

    ``token`` renders as a constant mask; take the string once with
    ``token.consume()`` or hand it straight to a runtime client with
    :meth:`runtime_client`.

    Example:
        >>> issued = IssuedIdentity(identity=RuntimeIdentity(id="i-1"), token="vrt_example_token")
        >>> repr(issued.token)
        "IssuedToken('**********')"
        >>> client = issued.runtime_client("https://cloud.example")
        >>> client.is_cloud
        True
    """

    identity: RuntimeIdentity
    token: IssuedToken

    def runtime_client(self, base_url: str, *, timeout: float = 30.0) -> Client:
        """A client holding only this runtime token, for ``client.tools.execute``.

        The token is copied into the new client and ``issued.token.consume()`` stays
        available; build the client first, because after ``consume()`` the token is gone.
        """
        from agenomic._client import Client
        from agenomic.vault.sensitive import Sensitive, _unseal

        return Client(
            base_url=base_url, timeout=timeout, runtime_token=Sensitive(_unseal(self.token))
        )


class ExecutionSummary(_Response):
    action_id: str
    environment: str = ""
    agent_id: str = ""
    binding_id: str = ""
    binding_version: int = 0
    secret_version_id: Optional[str] = None
    grant_id: Optional[str] = None
    tool: str = ""
    state: str = ""
    run_id: Optional[str] = None
    attempts: int = 0
    request_digest: str = ""
    result_digest: Optional[str] = None
    error_class: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    ledger_run_id: str = ""
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Receipt(_Response):
    """A signed, append-only usage receipt (``authorized``, ``completed``, ``outcome_unknown``, ``refused``)."""

    id: str
    action_id: str = ""
    phase: str = ""
    ledger_run_id: str = ""
    ledger_event_hash: Optional[str] = None
    body: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class ExecuteResult(_Response):
    """The outcome of a successful execution: the business result and its receipt.

    ``result`` is what the server returned after its result filter; it never
    contains the credential. ``replayed`` is true when a replay fixture served it.

    Example:
        >>> out = ExecuteResult(action_id="a-1", result={"id": "c_1"}, receipt_id="r-1")
        >>> (out.status, out.receipt_id, out.result)
        ('succeeded', 'r-1', {'id': 'c_1'})
    """

    action_id: str
    state: str = "succeeded"
    result: JsonValue = None
    receipt_id: Optional[str] = None
    status_code: Optional[int] = None
    limitations: list[str] = Field(default_factory=list)
    replayed: bool = False

    @property
    def status(self) -> str:
        return self.state


class ExecutionStatus(_Response):
    """Any state of an execution as the server reports it; reading it never raises on a state.

    ``state`` is one of ``reserved``, ``authorized``, ``sent``, ``succeeded``,
    ``failed``, ``refused`` or ``outcome_unknown``.

    Example:
        >>> ExecutionStatus(status="finished", action_id="a-1", state="outcome_unknown").terminal
        True
    """

    status: str = "finished"
    action_id: str = ""
    state: Optional[str] = None
    receipt_id: Optional[str] = None
    result: JsonValue = None
    status_code: Optional[int] = None
    error_class: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)
    approval_id: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    replayed: bool = False

    @property
    def terminal(self) -> bool:
        return self.state in ("succeeded", "failed", "refused", "outcome_unknown")


class Destination(_Request):
    """Fixed HTTPS destination of a binding; the agent cannot change it."""

    scheme: Literal["https"] = "https"
    host: str
    port: int = 443
    allow_private: Optional[bool] = None


class BearerAuth(_Request):
    kind: Literal["bearer"] = "bearer"


class HeaderAuth(_Request):
    kind: Literal["header"] = "header"
    name: str


class BasicAuth(_Request):
    kind: Literal["basic"] = "basic"
    username: str


class QueryAuth(_Request):
    kind: Literal["query"] = "query"
    name: str


AuthPlacement = Annotated[
    Union[BearerAuth, HeaderAuth, BasicAuth, QueryAuth], Field(discriminator="kind")
]


class PathParam(_Request):
    name: str
    pattern: Optional[str] = None
    allowed: Optional[list[str]] = None


class NoBody(_Request):
    mode: Literal["none"] = "none"


class ArgumentsBody(_Request):
    mode: Literal["arguments"] = "arguments"


class FieldsBody(_Request):
    mode: Literal["fields"] = "fields"
    names: list[str]


BodyRule = Annotated[Union[NoBody, ArgumentsBody, FieldsBody], Field(discriminator="mode")]


class RequestTemplate(_Request):
    """The request the executor builds; only bounded placeholders take agent input."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    path_params: Optional[list[PathParam]] = None
    query: Optional[dict[str, str]] = None
    static_query: Optional[dict[str, str]] = None
    body: BodyRule = Field(default_factory=NoBody)
    static_headers: Optional[dict[str, str]] = None
    idempotency_header: Optional[str] = None


class ResponseRule(_Request):
    max_bytes: Optional[int] = None
    content_types: Optional[list[str]] = None
    fields: Optional[list[str]] = None


class UsageRules(_Request):
    read_only: Optional[bool] = None
    irreversible: Optional[bool] = None
    allowed_hours_utc: Optional[tuple[int, int]] = None
    constraints: Optional[list[dict[str, JsonValue]]] = None
    sensitive_mandate: Optional[str] = None


class BindingContent(_Request):
    """What a binding allows: secret, destination, auth placement, request template, effect.

    Example:
        >>> content = BindingContent(
        ...     secret_id="s-1", upstream_identity="crm-service", tool_contract_ref="crm.get_customer@1",
        ...     destination=Destination(host="crm.example.test"), auth=BearerAuth(),
        ...     request=RequestTemplate(method="GET", path="/customers/{id}",
        ...         path_params=[PathParam(name="id", pattern="^c_[0-9]+$")]),
        ...     effect="read")
        >>> content.model_dump(mode="json", exclude_none=True)["request"]["body"]
        {'mode': 'none'}
    """

    secret_id: str
    upstream_identity: str
    tool_contract_ref: str
    destination: Destination
    auth: AuthPlacement
    request: RequestTemplate
    effect: Literal["read", "write"]
    response: Optional[ResponseRule] = None
    usage_rules: Optional[UsageRules] = None


class TokenAuth(_Request):
    """Provider auth through a token held in an environment variable of the executor."""

    method: Literal["token"] = "token"
    env_var: str


class AppRoleAuth(_Request):
    method: Literal["app_role"] = "app_role"
    role_id_env: str
    secret_id_env: str
    mount: str


class KubernetesAuth(_Request):
    method: Literal["kubernetes"] = "kubernetes"
    role: str
    mount: str
    jwt_path: str


ProviderAuth = Annotated[
    Union[TokenAuth, AppRoleAuth, KubernetesAuth], Field(discriminator="method")
]


class ProviderDescriptor(_Request):
    """How the executor reaches a vault backend; names environment variables, never values."""

    kind: Literal[
        "openbao", "hashicorp_vault", "aws_secrets_manager", "azure_key_vault", "gcp_secret_manager"
    ]
    address: str
    mount: str
    auth: ProviderAuth
    namespace: Optional[str] = None
    dynamic_mount: Optional[str] = None
