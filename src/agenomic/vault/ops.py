"""Builders of every vault operation: a route, its body and the parser of its reply.

Pure functions, no I/O. The sync and async namespaces run the same :class:`Op`,
so a route is described once. Every secret value travels as a
:class:`Sensitive` inside a body flagged ``secret_bearing``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Optional, Union

from pydantic import BaseModel, JsonValue, SecretStr

from agenomic.vault.execution import parse_status
from agenomic.vault.models import (
    Binding,
    BindingContent,
    BindingDetail,
    BindingVersion,
    ExecutionStatus,
    ExecutionSummary,
    Grant,
    IssuedIdentity,
    Provider,
    ProviderDescriptor,
    Receipt,
    Revocation,
    RotationJob,
    RuntimeIdentity,
    Secret,
    SecretDetail,
    VaultStatus,
)
from agenomic.vault.sensitive import Sensitive, _as_sensitive
from agenomic.vault.transport import Op, get, list_of, model_of, post, present, query_of, segment

PROVIDERS = "/v1/vault/providers"
SECRETS = "/v1/vault/secrets"
ROTATIONS = "/v1/vault/rotations"
BINDINGS = "/v1/vault/bindings"
GRANTS = "/v1/vault/grants"
IDENTITIES = "/v1/vault/runtime-identities"
EXECUTIONS = "/v1/vault/executions"
RUNTIME = "/v1/vault/runtime"

Content = Union[BindingContent, Mapping[str, JsonValue]]
Descriptor = Union[ProviderDescriptor, Mapping[str, JsonValue]]
SecretValue = Union[Sensitive, SecretStr]


def _json(value: Union[BaseModel, Mapping[str, JsonValue]]) -> dict[str, object]:
    if isinstance(value, BaseModel):
        return dict(value.model_dump(mode="json", exclude_none=True))
    return dict(value)


def _reason(reason: str) -> dict[str, object]:
    return {"reason": reason}


def status() -> Op[VaultStatus]:
    return Op(get("/v1/vault/status"), model_of(VaultStatus))


def providers_list() -> Op[list[Provider]]:
    return Op(get(PROVIDERS), list_of(Provider))


def provider_get(provider_id: str) -> Op[Provider]:
    return Op(get(f"{PROVIDERS}/{segment(provider_id)}"), model_of(Provider))


def provider_create(name: str, mode: str, descriptor: Descriptor) -> Op[Provider]:
    body = {"name": name, "mode": mode, "descriptor": _json(descriptor)}
    return Op(post(PROVIDERS, body), model_of(Provider))


def provider_health(provider_id: str) -> Op[Provider]:
    return Op(post(f"{PROVIDERS}/{segment(provider_id)}/health"), model_of(Provider))


def provider_state(provider_id: str, state: str, reason: str) -> Op[Provider]:
    body = {"state": state, "reason": reason}
    return Op(post(f"{PROVIDERS}/{segment(provider_id)}/state", body), model_of(Provider))


def secrets_list(
    environment: Optional[str], provider_id: Optional[str], state: Optional[str]
) -> Op[list[Secret]]:
    query = query_of(environment=environment, provider_id=provider_id, state=state)
    return Op(get(SECRETS, query=query), list_of(Secret))


def secret_get(secret_id: str) -> Op[SecretDetail]:
    return Op(get(f"{SECRETS}/{segment(secret_id)}"), model_of(SecretDetail))


def secret_create(
    environment: str,
    name: str,
    secret_type: str,
    provider_id: str,
    value: SecretValue,
    classification: Optional[str],
    provider_ref: Optional[str],
) -> Op[SecretDetail]:
    body: dict[str, object] = {
        "environment": environment,
        "name": name,
        "secret_type": secret_type,
        "provider_id": provider_id,
        "value": _as_sensitive(value),
    }
    body.update(present(classification=classification, provider_ref=provider_ref))
    return Op(post(SECRETS, body, secret=True), model_of(SecretDetail))


def secret_reference(
    environment: str,
    name: str,
    secret_type: str,
    provider_id: str,
    provider_ref: str,
    provider_version: str,
    classification: Optional[str],
) -> Op[SecretDetail]:
    body: dict[str, object] = {
        "environment": environment,
        "name": name,
        "secret_type": secret_type,
        "provider_id": provider_id,
        "provider_ref": provider_ref,
        "provider_version": provider_version,
    }
    body.update(present(classification=classification))
    return Op(post(f"{SECRETS}/references", body), model_of(SecretDetail))


def secret_add_version(secret_id: str, value: SecretValue) -> Op[SecretDetail]:
    body = {"value": _as_sensitive(value)}
    path = f"{SECRETS}/{segment(secret_id)}/versions"
    return Op(post(path, body, secret=True), model_of(SecretDetail))


def secret_revoke(secret_id: str, reason: str) -> Op[Revocation]:
    path = f"{SECRETS}/{segment(secret_id)}/revoke"
    return Op(post(path, _reason(reason)), model_of(Revocation))


def rotation_start(
    secret_id: str, value: SecretValue, overlap_seconds: Optional[int]
) -> Op[RotationJob]:
    body: dict[str, object] = {"value": _as_sensitive(value)}
    if overlap_seconds is not None:
        body["overlap_seconds"] = int(overlap_seconds)
    path = f"{SECRETS}/{segment(secret_id)}/rotations"
    return Op(post(path, body, secret=True), model_of(RotationJob))


def rotations_list(secret_id: Optional[str], state: Optional[str]) -> Op[list[RotationJob]]:
    query = query_of(secret_id=secret_id, state=state)
    return Op(get(ROTATIONS, query=query), list_of(RotationJob))


def rotation_get(rotation_id: str) -> Op[RotationJob]:
    return Op(get(f"{ROTATIONS}/{segment(rotation_id)}"), model_of(RotationJob))


def rotation_activate(rotation_id: str) -> Op[RotationJob]:
    return Op(post(f"{ROTATIONS}/{segment(rotation_id)}/activate"), model_of(RotationJob))


def rotation_rollback(rotation_id: str, reason: str) -> Op[RotationJob]:
    path = f"{ROTATIONS}/{segment(rotation_id)}/rollback"
    return Op(post(path, _reason(reason)), model_of(RotationJob))


def bindings_list(
    environment: Optional[str], agent_id: Optional[str], state: Optional[str]
) -> Op[list[Binding]]:
    query = query_of(environment=environment, agent_id=agent_id, state=state)
    return Op(get(BINDINGS, query=query), list_of(Binding))


def binding_get(binding_id: str) -> Op[BindingDetail]:
    return Op(get(f"{BINDINGS}/{segment(binding_id)}"), model_of(BindingDetail))


def binding_create(
    environment: str, name: str, agent_id: str, tool_name: str, content: Content
) -> Op[BindingDetail]:
    body = {
        "environment": environment,
        "name": name,
        "agent_id": agent_id,
        "tool_name": tool_name,
        "content": _json(content),
    }
    return Op(post(BINDINGS, body), model_of(BindingDetail))


def binding_propose(binding_id: str, content: Content) -> Op[BindingVersion]:
    path = f"{BINDINGS}/{segment(binding_id)}/versions"
    return Op(post(path, {"content": _json(content)}), model_of(BindingVersion))


def binding_submit(binding_id: str, version: int) -> Op[BindingVersion]:
    path = f"{BINDINGS}/{segment(binding_id)}/versions/{int(version)}/submit"
    return Op(post(path), model_of(BindingVersion))


def binding_decide(binding_id: str, version: int, approve: bool) -> Op[BindingVersion]:
    path = f"{BINDINGS}/{segment(binding_id)}/versions/{int(version)}/decide"
    return Op(post(path, {"approve": approve}), model_of(BindingVersion))


def binding_activate(binding_id: str, version: int) -> Op[BindingDetail]:
    path = f"{BINDINGS}/{segment(binding_id)}/versions/{int(version)}/activate"
    return Op(post(path), model_of(BindingDetail))


def binding_revoke(binding_id: str, reason: str) -> Op[Revocation]:
    path = f"{BINDINGS}/{segment(binding_id)}/revoke"
    return Op(post(path, _reason(reason)), model_of(Revocation))


def grants_list(
    binding_id: Optional[str], agent_id: Optional[str], state: Optional[str]
) -> Op[list[Grant]]:
    query = query_of(binding_id=binding_id, agent_id=agent_id, state=state)
    return Op(get(GRANTS, query=query), list_of(Grant))


def _grant_body(
    binding_id: str, max_uses: int, ttl_seconds: int, reason: str, version: Optional[int]
) -> dict[str, object]:
    body: dict[str, object] = {
        "binding_id": binding_id,
        "max_uses": int(max_uses),
        "ttl_seconds": int(ttl_seconds),
        "reason": reason,
    }
    if version is not None:
        body["version"] = int(version)
    return body


def grant_request(
    binding_id: str, max_uses: int, ttl_seconds: int, reason: str, version: Optional[int]
) -> Op[Grant]:
    body = _grant_body(binding_id, max_uses, ttl_seconds, reason, version)
    return Op(post(GRANTS, body), model_of(Grant))


def grant_decide(grant_id: str, approve: bool) -> Op[Grant]:
    path = f"{GRANTS}/{segment(grant_id)}/decide"
    return Op(post(path, {"approve": approve}), model_of(Grant))


def grant_revoke(grant_id: str, reason: str) -> Op[Revocation]:
    path = f"{GRANTS}/{segment(grant_id)}/revoke"
    return Op(post(path, _reason(reason)), model_of(Revocation))


def identities_list() -> Op[list[RuntimeIdentity]]:
    return Op(get(IDENTITIES), list_of(RuntimeIdentity))


def identity_issue(
    environment: str,
    agent_id: str,
    label: str,
    ttl_seconds: Optional[int],
    declared_release: Optional[str],
    declared_genome_digest: Optional[str],
) -> Op[IssuedIdentity]:
    body: dict[str, object] = {"environment": environment, "agent_id": agent_id, "label": label}
    body.update(
        present(declared_release=declared_release, declared_genome_digest=declared_genome_digest)
    )
    if ttl_seconds is not None:
        body["ttl_seconds"] = int(ttl_seconds)
    return Op(post(IDENTITIES, body), model_of(IssuedIdentity))


def identity_revoke(identity_id: str, reason: str) -> Op[RuntimeIdentity]:
    path = f"{IDENTITIES}/{segment(identity_id)}/revoke"
    return Op(post(path, _reason(reason)), model_of(RuntimeIdentity))


def executions_list(state: Optional[str], limit: Optional[int]) -> Op[list[ExecutionSummary]]:
    return Op(get(EXECUTIONS, query=query_of(state=state, limit=limit)), list_of(ExecutionSummary))


def execution_get(action_id: str) -> Op[ExecutionStatus]:
    path = f"{EXECUTIONS}/{segment(action_id)}"
    return Op(get(path), lambda reply: parse_status(reply, action_id))


def execution_resolve(
    action_id: str, resolution: Literal["applied", "not_applied"], note: str
) -> Op[ExecutionStatus]:
    path = f"{EXECUTIONS}/{segment(action_id)}/resolve"
    body = {"resolution": resolution, "note": note}
    return Op(post(path, body), lambda reply: parse_status(reply, action_id))


def receipts_list(action_id: Optional[str], limit: Optional[int]) -> Op[list[Receipt]]:
    query = query_of(action_id=action_id, limit=limit)
    return Op(get("/v1/vault/receipts", query=query), list_of(Receipt))


def revocations_list() -> Op[list[Revocation]]:
    return Op(get("/v1/vault/revocations"), list_of(Revocation))


def revocation_retry(revocation_id: str) -> Op[Revocation]:
    path = f"/v1/vault/revocations/{segment(revocation_id)}/retry"
    return Op(post(path), model_of(Revocation))


def revocation_lift(revocation_id: str, reason: str) -> Op[Revocation]:
    path = f"/v1/vault/revocations/{segment(revocation_id)}/lift"
    return Op(post(path, _reason(reason)), model_of(Revocation))


def kill_switch(target_kind: str, target_id: str, reason: str) -> Op[Revocation]:
    body = {"target_kind": target_kind, "target_id": target_id, "reason": reason}
    return Op(post("/v1/vault/kill-switch", body), model_of(Revocation))


def runtime_execution_get(action_id: str) -> Op[ExecutionStatus]:
    path = f"{RUNTIME}/executions/{segment(action_id)}"
    return Op(get(path, plane="runtime"), lambda reply: parse_status(reply, action_id))


def runtime_grants_list(binding_id: Optional[str], state: Optional[str]) -> Op[list[Grant]]:
    query = query_of(binding_id=binding_id, state=state)
    return Op(get(f"{RUNTIME}/grants", plane="runtime", query=query), list_of(Grant))


def runtime_grant_request(
    binding_id: str, max_uses: int, ttl_seconds: int, reason: str, version: Optional[int]
) -> Op[Grant]:
    body = _grant_body(binding_id, max_uses, ttl_seconds, reason, version)
    return Op(post(f"{RUNTIME}/grants", body, plane="runtime"), model_of(Grant))


def runtime_grant_delegate(
    grant_id: str, delegate_agent_id: str, max_uses: int, ttl_seconds: int, reason: str
) -> Op[Grant]:
    body: dict[str, object] = {
        "delegate_agent_id": delegate_agent_id,
        "max_uses": int(max_uses),
        "ttl_seconds": int(ttl_seconds),
        "reason": reason,
    }
    path = f"{RUNTIME}/grants/{segment(grant_id)}/delegations"
    return Op(post(path, body, plane="runtime"), model_of(Grant))
