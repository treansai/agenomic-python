from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, TypeVar, Union, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from agenomic._transport import ApiResponse, aapi_request, api_request, segment
from agenomic._version import __version__
from agenomic.exceptions import ApiError
from agenomic.prompts.bundle import BundleTrust, PromptBundle
from agenomic.prompts.cache import PromptCache
from agenomic.prompts.digest import canonical_json_v1, prompt_digest
from agenomic.prompts.discovery import PYTHON_GRAMMAR, SCANNER_NAME
from agenomic.prompts.errors import (
    PromptIntegrityError,
    PromptRefError,
    binding_error,
    integrity_error,
)
from agenomic.prompts.importer import (
    DISCOVERY_SCHEMA,
    ISSUE_MESSAGES,
    ISSUE_SEVERITY,
    Source,
    build_import_request,
    check_prompts_file,
    default_decisions,
    load_prompts_file,
    new_idempotency_key,
    verify_plan,
)
from agenomic.prompts.models import (
    ExecutionBinding,
    ManagedPromptVersion,
    PromptVersionRecord,
    ResolvedFrom,
)
from agenomic.prompts.refs import (
    PromptAliasRef,
    PromptReference,
    PromptUri,
    PromptVersionRef,
    parse_execution_ref,
    parse_prompt_ref,
)
from agenomic.prompts.render import PromptIssue, RenderedMessage
from agenomic.prompts.secrets import SECRET_PATTERN_SET

if TYPE_CHECKING:
    from agenomic._client import Client
    from agenomic.integrations.langchain_prompts import LangChainImport
    from agenomic.prompts.local import LocalPromptEngine

__all__ = [
    "BindingsResource",
    "Channel",
    "ChannelEvent",
    "ChannelMovePreview",
    "ChannelsResource",
    "Draft",
    "ImportPlan",
    "Page",
    "PinnedRefs",
    "PromptAlias",
    "PromptAliasesResource",
    "PromptDraftsResource",
    "PromptSummary",
    "PromptVersionSummary",
    "PromptVersionsResource",
    "PromptsResource",
    "RenderResult",
]

_R = TypeVar("_R")
_T = TypeVar("_T")
_V = TypeVar("_V", bound=BaseModel)
_MANAGED_KEY = "__agenomic_prompt_set"
_LANGCHAIN_CONFIG = "langchain_core.runnables.config"
_RUNTIME_LABEL = "runtime_registration"
_USAGES = frozenset({"system", "instructions", "user", "chat", "tool_description", "other"})
_SLOT_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", re.ASCII)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
_OBSERVATION_KEYS = frozenset(
    {
        "slot_path",
        "node_path",
        "prompt_ref",
        "content_digest",
        "rendered_hash",
        "overlay",
        "alias",
        "alias_generation",
        "unmanaged",
        "role_layout",
        "count",
        "first_at",
        "last_at",
    }
)
_OVERLAY_KEYS = frozenset({"digest", "position"})
_USAGE_BATCH = 500


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    body: Optional[Mapping[str, Any]] = None
    params: Optional[Mapping[str, str]] = None
    if_match: Optional[int] = None
    retry: bool = False


Step = Union[Call, Callable[[], Any]]
Flow = Generator[Step, Any, _R]


def _execute(client: Client, step: Step) -> Any:
    if isinstance(step, Call):
        return api_request(
            client,
            step.method,
            step.path,
            step.body,
            if_match=step.if_match,
            retry=step.retry,
            params=step.params,
        )
    return step()


async def _aexecute(client: Client, step: Step) -> Any:
    if isinstance(step, Call):
        return await aapi_request(
            client,
            step.method,
            step.path,
            step.body,
            if_match=step.if_match,
            retry=step.retry,
            params=step.params,
        )
    return await asyncio.to_thread(step)


def run_flow(client: Client, flow: Flow[_R]) -> _R:
    value: Any = None
    error: Optional[Exception] = None
    while True:
        try:
            step = flow.throw(error) if error is not None else flow.send(value)
        except StopIteration as stop:
            return cast(_R, stop.value)
        value, error = None, None
        try:
            value = _execute(client, step)
        except Exception as raised:
            error = raised


async def arun_flow(client: Client, flow: Flow[_R]) -> _R:
    value: Any = None
    error: Optional[Exception] = None
    while True:
        try:
            step = flow.throw(error) if error is not None else flow.send(value)
        except StopIteration as stop:
            return cast(_R, stop.value)
        value, error = None, None
        try:
            value = await _aexecute(client, step)
        except Exception as raised:
            error = raised


class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class PromptSummary(_View):
    prompt_id: str
    kind: Literal["text", "chat", "fragment"]
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: list[str] = []
    status: str = "active"
    latest_version: Optional[int] = None
    metadata_revision: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PromptVersionSummary(_View):
    prompt_id: str
    version: int
    content_digest: str
    parent_version: Optional[int] = None
    change_message: Optional[str] = None
    created_at: Optional[str] = None


class Draft(_View):
    prompt_id: str
    revision: int
    base_version: Optional[int] = None
    origin: str
    content: dict[str, Any]
    validation: dict[str, Any]
    updated_at: Optional[str] = None


class PromptAlias(_View):
    prompt_id: str
    alias: str
    version: int
    content_digest: str
    generation: int
    updated_at: Optional[str] = None


class Channel(_View):
    agent_id: str
    name: str
    release_id: Optional[str] = None
    release: Optional[dict[str, Any]] = None
    generation: int
    protected: bool
    materialized: bool


class ChannelEvent(_View):
    generation: int
    action: str
    from_release_id: Optional[str] = None
    to_release_id: Optional[str] = None
    reason: Optional[str] = None
    created_at: Optional[str] = None


class ChannelMovePreview(_View):
    agent_id: str
    action: str
    actions: dict[str, Any] = {}
    move_url: Optional[str] = None


@dataclass(frozen=True)
class Page(Generic[_T]):
    items: list[_T]
    next_cursor: Optional[str]


@dataclass(frozen=True)
class RenderResult:
    kind: Literal["text", "chat"]
    ref: PromptVersionRef
    content_digest: str
    rendered_hash: str
    text: Optional[str] = None
    messages: Optional[list[Union[RenderedMessage, object]]] = None


@dataclass(frozen=True)
class ImportPlan:
    plan: dict[str, Any]
    import_id: Optional[str] = None
    status: Optional[str] = None
    replayed: bool = False
    report_digest: Optional[str] = None
    expires_at: Optional[str] = None
    slots: Optional[dict[str, Any]] = None

    @property
    def plan_digest(self) -> str:
        return cast(str, self.plan["plan_digest"])

    @property
    def items(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.plan["items"])

    def decisions(self) -> list[dict[str, Any]]:
        return default_decisions(self.plan)


class PinnedRefs(Mapping[str, ManagedPromptVersion]):
    __slots__ = ("_versions",)

    def __init__(self, versions: Mapping[str, ManagedPromptVersion]) -> None:
        self._versions = dict(versions)

    def __getitem__(self, ref: str) -> ManagedPromptVersion:
        return self._versions[ref]

    def __iter__(self) -> Iterator[str]:
        return iter(self._versions)

    def __len__(self) -> int:
        return len(self._versions)


def _invalid(response: ApiResponse, what: str) -> ApiError:
    return ApiError("invalid_response", response.status, f"the response carries no valid {what}")


def _cloud_required(operation: str) -> ApiError:
    return ApiError(
        "cloud_required",
        0,
        f"{operation} needs Agenomic Cloud; the local prompt engine does not simulate it",
    )


def _view(model: type[_V], value: Any, response: Optional[ApiResponse], what: str) -> _V:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        if response is None:
            raise ApiError("invalid_response", 0, f"invalid {what}") from error
        raise _invalid(response, what) from error


def _member(response: ApiResponse, key: str) -> Any:
    if key not in response.body:
        raise _invalid(response, key)
    return response.body[key]


def _counter(response: ApiResponse, value: int, what: str) -> None:
    if response.etag is not None and response.etag != value:
        raise _invalid(response, f"{what} matching its ETag")


def _items(response: ApiResponse, key: str) -> list[Any]:
    value = _member(response, key)
    if not isinstance(value, list):
        raise _invalid(response, key)
    return value


def _cursor(response: ApiResponse) -> Optional[str]:
    value = response.body.get("next_cursor")
    if value is not None and not isinstance(value, str):
        raise _invalid(response, "next_cursor")
    return value


def _selector(channel: Optional[str], release_id: Optional[str]) -> dict[str, str]:
    if (channel is None) == (release_id is None):
        raise ValueError("name exactly one of channel and release_id")
    if channel is not None:
        return {"channel": channel}
    return {"release_id": cast(str, release_id)}


def _in_managed_run() -> bool:
    module = sys.modules.get(_LANGCHAIN_CONFIG)
    variable = getattr(module, "var_child_runnable_config", None)
    config = variable.get() if variable is not None else None
    if not isinstance(config, Mapping):
        return False
    configurable = config.get("configurable")
    return isinstance(configurable, Mapping) and configurable.get(_MANAGED_KEY) is not None


def _refuse_alias_in_managed_run() -> None:
    if _in_managed_run():
        raise PromptRefError(
            "alias_in_managed_run",
            0,
            "a managed run reads its release manifest only; aliases are never resolved inside it",
        )


def _parse(ref: Union[str, PromptReference]) -> PromptReference:
    return parse_execution_ref(ref) if isinstance(ref, str) else ref


def _engine(client: Client) -> Optional[LocalPromptEngine]:
    return client._prompt_engine


def whoami_flow(client: Client) -> Flow[dict[str, Any]]:
    if _engine(client) is not None:
        raise _cloud_required("whoami")
    known = client._cached_whoami()
    if known is not None:
        return known
    response = yield Call("GET", "/v1/whoami", retry=True)
    return client._remember_whoami(response.body)


def workspace_flow(client: Client) -> Flow[str]:
    known = client._known_workspace()
    if known is not None:
        return known
    identity = yield from whoami_flow(client)
    return cast(str, identity["org_id"])


def _cached_version(cache: PromptCache, workspace_id: str, ref: PromptVersionRef) -> Any:
    try:
        return cache.get_version(workspace_id, ref.prompt_id, ref.version)
    except PromptIntegrityError as error:
        if error.code != "cache_conflict":
            raise
        return None


def _store_version(cache: PromptCache, workspace_id: str, version: ManagedPromptVersion) -> None:
    try:
        cache.put_version(workspace_id, version)
    except PromptIntegrityError as error:
        if error.code != "cache_conflict":
            raise


def _record(item: Any, workspace_id: str, response: ApiResponse) -> PromptVersionRecord:
    if not isinstance(item, Mapping):
        raise _invalid(response, "prompt version")
    record = _view(
        PromptVersionRecord, {**item, "workspace_id": workspace_id}, response, "prompt version"
    )
    uri = item.get("canonical_uri")
    if uri is None:
        return record
    try:
        parsed = parse_prompt_ref(uri) if isinstance(uri, str) else None
    except PromptRefError:
        parsed = None
    if not isinstance(parsed, PromptUri):
        raise _invalid(response, "canonical_uri")
    if parsed.workspace_id != workspace_id:
        raise PromptRefError(
            "workspace_mismatch",
            0,
            "the registry answered for another workspace than the configured workspace_id",
        )
    if (parsed.prompt_id, parsed.version) != (record.prompt_id, record.version):
        raise _invalid(response, "canonical_uri")
    return record


def _closure(
    response: ApiResponse, workspace_id: str, ref: PromptVersionRef
) -> ManagedPromptVersion:
    root = _record(_member(response, "version"), workspace_id, response)
    if (root.prompt_id, root.version) != (ref.prompt_id, ref.version):
        raise _invalid(response, f"version {ref}")
    fragments = response.body.get("fragments", [])
    if not isinstance(fragments, list):
        raise _invalid(response, "fragments")
    records = {root.ref: root}
    for item in fragments:
        record = _record(item, workspace_id, response)
        records[record.ref] = record
    return ManagedPromptVersion.from_record(
        root,
        workspace_id=workspace_id,
        lookup=lambda prompt_id, number: records.get(f"{prompt_id}:{number}"),
    )


def version_flow(
    client: Client, workspace_id: str, ref: PromptVersionRef
) -> Flow[ManagedPromptVersion]:
    cache = client.prompt_cache
    cached = yield partial(_cached_version, cache, workspace_id, ref)
    if isinstance(cached, ManagedPromptVersion):
        return cached
    response = yield Call(
        "GET",
        f"/v1/prompts/{segment(ref.prompt_id)}/versions/{ref.version}",
        params={"include": "fragments"},
        retry=True,
    )
    version = _closure(response, workspace_id, ref)
    yield partial(_store_version, cache, workspace_id, version)
    return version


def _check_digest(version: ManagedPromptVersion, expected: str) -> None:
    if version.content_digest != expected:
        raise integrity_error(
            "prompt_digest_mismatch",
            f"{version.ref} does not carry the digest the registry resolved",
            ref=str(version.ref),
            expected=expected,
            actual=version.content_digest,
        )


def _resolution(response: ApiResponse, ref: PromptAliasRef) -> tuple[PromptVersionRef, int, str]:
    body = response.body
    alias = body.get("alias")
    version = body.get("version")
    digest = body.get("content_digest")
    if (
        body.get("prompt_id") != ref.prompt_id
        or not isinstance(alias, Mapping)
        or alias.get("name") != ref.alias
        or isinstance(alias.get("generation"), bool)
        or not isinstance(alias.get("generation"), int)
        or isinstance(version, bool)
        or not isinstance(version, int)
        or not isinstance(digest, str)
    ):
        raise _invalid(response, f"resolution of {ref}")
    return PromptVersionRef(ref.prompt_id, version), alias["generation"], digest


def alias_flow(
    client: Client, workspace_id: str, ref: PromptAliasRef
) -> Flow[ManagedPromptVersion]:
    response = yield Call("POST", "/v1/prompts/resolve", {"ref": str(ref)}, retry=True)
    pinned, generation, digest = _resolution(response, ref)
    version = yield from version_flow(client, workspace_id, pinned)
    _check_digest(version, digest)
    return version.model_copy(update={"resolved_from": ResolvedFrom(ref.alias, generation)})


def get_flow(client: Client, ref: Union[str, PromptReference]) -> Flow[ManagedPromptVersion]:
    parsed = _parse(ref)
    if isinstance(parsed, PromptAliasRef):
        _refuse_alias_in_managed_run()
    engine = _engine(client)
    if engine is not None:
        return engine.get(parsed)
    workspace_id = yield from workspace_flow(client)
    if isinstance(parsed, PromptAliasRef):
        return (yield from alias_flow(client, workspace_id, parsed))
    if isinstance(parsed, PromptUri):
        parsed = parsed.to_version_ref(workspace_id)
    return (yield from version_flow(client, workspace_id, parsed))


def _binding(document: Any, workspace_id: str, agent_id: str) -> ExecutionBinding:
    binding = _view(ExecutionBinding, document, None, "execution binding")
    if binding.workspace_id != workspace_id or binding.agent_id != agent_id:
        raise binding_error(
            "binding_mismatch",
            "the binding belongs to another workspace or agent",
            binding_id=binding.binding_id,
        )
    return binding


def binding_bundle(binding: ExecutionBinding, artifacts: Any) -> PromptBundle:
    if not isinstance(artifacts, Mapping):
        raise ApiError("invalid_response", 0, "the binding carries no artifacts")
    bundle = PromptBundle.from_online_response(
        artifacts,
        expected_workspace_id=binding.workspace_id,
        expected_agent_id=binding.agent_id,
        expected_manifest_digest=binding.prompt_manifest_digest,
    )
    if bundle.release_id != binding.release_id:
        raise binding_error(
            "binding_mismatch",
            "the artifacts belong to another release than the binding",
            binding_id=binding.binding_id,
        )
    digests = bundle.child_manifest_digests
    if digests.keys() != binding.children.keys():
        raise integrity_error(
            "manifest_digest_mismatch",
            "the artifacts and the binding pin different children",
            expected=sorted(binding.children),
            actual=sorted(digests),
        )
    for child_id, child in binding.children.items():
        if digests.get(child_id) != child.get("prompt_manifest_digest"):
            raise integrity_error(
                "manifest_digest_mismatch",
                "a child manifest differs from the binding pin",
                child_agent_id=child_id,
                expected=child.get("prompt_manifest_digest"),
                actual=digests.get(child_id),
            )
    return bundle


def _runtime_client(overrides: Optional[Mapping[str, Optional[str]]]) -> dict[str, Optional[str]]:
    runtime: dict[str, Optional[str]] = {
        "sdk": "agenomic-python",
        "sdk_version": __version__,
        "adapter": None,
        "adapter_version": None,
    }
    for key, value in (overrides or {}).items():
        if key not in runtime:
            raise ValueError(f"unknown runtime_client member {key}")
        runtime[key] = value
    return runtime


def create_binding_flow(
    client: Client,
    agent_id: str,
    *,
    thread_key: str,
    scope: Literal["thread", "execution"],
    channel: Optional[str] = None,
    release_id: Optional[str] = None,
    child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    expect_manifest_digest: Optional[str] = None,
    runtime_client: Optional[Mapping[str, Optional[str]]] = None,
) -> Flow[tuple[ExecutionBinding, PromptBundle, bool]]:
    selector = _selector(channel, release_id)
    runtime = _runtime_client(runtime_client)
    engine = _engine(client)
    if engine is not None:
        if child_selectors:
            raise _cloud_required("child_selectors")
        document, artifacts, created = engine.create_binding(
            agent_id,
            thread_key=thread_key,
            scope=scope,
            channel=channel,
            release_id=release_id,
            expect_manifest_digest=expect_manifest_digest,
            runtime_client=runtime,
        )
        workspace_id = engine.workspace_id
    else:
        workspace_id = yield from workspace_flow(client)
        body: dict[str, Any] = {
            "thread_key": thread_key,
            "scope": scope,
            "selector": selector,
            "runtime_client": runtime,
            "include": ["artifacts"],
        }
        if child_selectors:
            body["child_selectors"] = {key: dict(value) for key, value in child_selectors.items()}
        if expect_manifest_digest is not None:
            body["expect"] = {"prompt_manifest_digest": expect_manifest_digest}
        response = yield Call("POST", f"/v1/agents/{segment(agent_id)}/bindings", body, retry=True)
        document = _member(response, "binding")
        artifacts = response.body.get("artifacts")
        created = _member(response, "created")
        if not isinstance(created, bool):
            raise _invalid(response, "created flag")
    binding = _binding(document, workspace_id, agent_id)
    if binding.thread_key != thread_key or binding.scope != scope:
        raise binding_error(
            "binding_mismatch",
            "the binding answers another thread key or scope",
            binding_id=binding.binding_id,
        )
    return binding, binding_bundle(binding, artifacts), created


def get_binding_flow(
    client: Client, agent_id: str, binding_id: str
) -> Flow[tuple[ExecutionBinding, PromptBundle]]:
    engine = _engine(client)
    if engine is not None:
        document, artifacts = engine.get_binding(agent_id, binding_id)
        workspace_id = engine.workspace_id
    else:
        workspace_id = yield from workspace_flow(client)
        response = yield Call(
            "GET",
            f"/v1/agents/{segment(agent_id)}/bindings/{segment(binding_id)}",
            params={"include": "artifacts"},
            retry=True,
        )
        document = _member(response, "binding")
        artifacts = response.body.get("artifacts")
    binding = _binding(document, workspace_id, agent_id)
    if binding.binding_id != binding_id:
        raise binding_error("binding_mismatch", "the registry answered another binding")
    return binding, binding_bundle(binding, artifacts)


def import_report_flow(
    client: Client,
    report: Mapping[str, Any],
    agent_id: Optional[str],
    options: Optional[Mapping[str, Any]],
) -> Flow[ImportPlan]:
    if _engine(client) is not None:
        raise _cloud_required("prompts.import_report")
    body = build_import_request(report, agent_id=agent_id, options=options)
    source = {"kind": "discovery_report", "digest": prompt_digest(body["report"])}
    response = yield Call("POST", "/v1/prompts/imports", body, retry=True)
    record = _member(response, "import")
    if not isinstance(record, Mapping):
        raise _invalid(response, "import")
    plan = verify_plan(record.get("plan"))
    import_id = record.get("import_id", plan.get("plan_id"))
    if (
        not isinstance(import_id, str)
        or plan.get("plan_id") != import_id
        or plan.get("agent_id") != agent_id
        or plan.get("source") != source
    ):
        raise _invalid(response, "import plan for the request")
    return ImportPlan(
        plan=plan,
        import_id=import_id,
        status=record.get("status"),
        replayed=response.body.get("replayed") is True,
        report_digest=record.get("report_digest"),
        expires_at=record.get("expires_at"),
    )


def apply_import_flow(
    client: Client,
    import_id: str,
    *,
    plan_digest: str,
    items: Sequence[Mapping[str, Any]],
    mode: str,
    declare_slots: bool,
    expected_slots_revision: Optional[int],
    idempotency_key: Optional[str],
    agent_id: Optional[str],
) -> Flow[dict[str, Any]]:
    if _engine(client) is not None:
        raise _cloud_required("prompts.apply_import")
    if declare_slots and expected_slots_revision is None:
        raise ValueError("declare_slots needs expected_slots_revision")
    body: dict[str, Any] = {
        "idempotency_key": idempotency_key or new_idempotency_key(),
        "plan_digest": plan_digest,
        "mode": mode,
        "declare_slots": declare_slots,
        "items": [dict(item) for item in items],
    }
    if agent_id is not None:
        body["agent_id"] = agent_id
    response: ApiResponse = yield Call(
        "POST",
        f"/v1/prompts/imports/{segment(import_id)}/apply",
        body,
        if_match=expected_slots_revision if declare_slots else None,
        retry=True,
    )
    if response.body.get("import_id") != import_id or not isinstance(
        response.body.get("results"), list
    ):
        raise _invalid(response, f"result of import {import_id}")
    return response.body


def _prompts_file(document: Union[Mapping[str, Any], Source]) -> dict[str, Any]:
    if isinstance(document, Mapping):
        return check_prompts_file(dict(document))
    return load_prompts_file(document)


def plan_declarations_flow(
    client: Client, document: Union[Mapping[str, Any], Source]
) -> Flow[ImportPlan]:
    if _engine(client) is not None:
        raise _cloud_required("prompts.plan_declarations")
    sent = _prompts_file(document)
    source = {"kind": "prompts_file", "digest": prompt_digest(sent)}
    response = yield Call("POST", "/v1/prompts/declarations/plan", {"document": sent}, retry=True)
    plan = verify_plan(_member(response, "plan"))
    slots = response.body.get("slots")
    if plan.get("source") != source or (slots is not None and not isinstance(slots, dict)):
        raise _invalid(response, "plan of the prompts file")
    return ImportPlan(plan=plan, slots=slots)


def apply_declarations_flow(
    client: Client,
    document: Union[Mapping[str, Any], Source],
    *,
    plan_digest: str,
    idempotency_key: Optional[str],
    expected_slots_revision: Optional[int],
) -> Flow[dict[str, Any]]:
    if _engine(client) is not None:
        raise _cloud_required("prompts.apply_declarations")
    sent = _prompts_file(document)
    declares_slots = isinstance(sent.get("slots"), list) and isinstance(sent.get("agent_id"), str)
    if declares_slots and expected_slots_revision is None:
        raise ValueError("a prompts file that declares slots needs expected_slots_revision")
    body = {
        "idempotency_key": idempotency_key or new_idempotency_key("declarations-apply"),
        "document": sent,
        "plan_digest": plan_digest,
    }
    response: ApiResponse = yield Call(
        "POST",
        "/v1/prompts/declarations/apply",
        body,
        if_match=expected_slots_revision if declares_slots else None,
        retry=True,
    )
    if response.body.get("plan_digest") != plan_digest or not isinstance(
        response.body.get("results"), list
    ):
        raise _invalid(response, "result of the prompts file")
    return response.body


def _runtime_prompt_id(slot_path: str, taken: set[str]) -> str:
    base = re.sub(r"_+", "_", "prm_" + slot_path.replace(".", "_"))[:64].rstrip("_-")
    candidate = base
    counter = 2
    while candidate in taken:
        suffix = f"_{counter}"
        candidate = base[: 64 - len(suffix)].rstrip("_-") + suffix
        counter += 1
    taken.add(candidate)
    return candidate


def _runtime_issue(item: PromptIssue, supported: bool) -> dict[str, Any]:
    code = item.syntax if item.code == "syntax_error" and item.syntax else item.code
    severity = ISSUE_SEVERITY.get(item.code, "warning" if supported else "error")
    if item.code == "syntax_error":
        message = f"template syntax error: {code}"
    else:
        message = ISSUE_MESSAGES.get(code, f"the prompt cannot be imported: {code}")
    return {"code": code, "severity": severity, "line": None, "column": None, "message": message}


def _runtime_candidate(
    slot_path: str, converted: LangChainImport, taken: set[str]
) -> dict[str, Any]:
    construct = (
        "langchain.chat_prompt_template"
        if converted.prompt_kind == "chat"
        else "langchain.prompt_template"
    )
    identity = {"path": slot_path, "line": 1, "column": 1, "construct": construct}
    digest = hashlib.sha256(canonical_json_v1(identity).encode("utf-8")).hexdigest()
    usage = slot_path.rsplit(".", 1)[1]
    supported = converted.status == "supported"
    return {
        "candidate_id": "cand_" + digest[:16],
        "status": converted.status,
        "construct": construct,
        "source": {
            "path": slot_path,
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 1,
            "symbol": None,
            "enclosing_function": None,
        },
        "proposal": {
            "prompt_id": _runtime_prompt_id(slot_path, taken),
            "prompt_kind": converted.prompt_kind,
            "slot_path": slot_path,
            "node_path": None,
            "usage": usage if usage in _USAGES else "other",
        },
        "content": converted.content,
        "content_digest": converted.content_digest,
        "issues": [_runtime_issue(item, supported) for item in converted.issues],
        "secret_findings": [
            {"pattern": found.pattern, "line": 1, "column": 1, "length": found.length}
            for found in converted.secret_findings
        ],
    }


def _runtime_report(slots: Mapping[str, Any]) -> dict[str, Any]:
    if not slots:
        raise ValueError("register_runtime needs at least one slot")
    for slot_path in slots:
        if (
            not isinstance(slot_path, str)
            or len(slot_path) > 128
            or not _SLOT_PATH.fullmatch(slot_path)
        ):
            raise ValueError(f"invalid slot path {slot_path!r}")
    from agenomic.integrations.langchain_prompts import from_langchain

    taken: set[str] = set()
    candidates = [
        _runtime_candidate(slot_path, from_langchain(slots[slot_path]), taken)
        for slot_path in sorted(slots)
    ]
    return {
        "schema": DISCOVERY_SCHEMA,
        "scanner": {
            "name": SCANNER_NAME,
            "version": __version__[:64] or "0",
            "python_grammar": PYTHON_GRAMMAR,
            "secret_patterns": SECRET_PATTERN_SET,
        },
        "root": {"label": _RUNTIME_LABEL, "vcs": None},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limits": {"max_files": 4000, "max_file_bytes": 512 * 1024},
        "files": [],
        "candidates": candidates,
    }


def register_runtime_flow(
    client: Client, agent_id: str, slots: Mapping[str, Any]
) -> Flow[ImportPlan]:
    if _engine(client) is not None:
        raise _cloud_required("prompts.register_runtime")
    report = _runtime_report(slots)
    return (yield from import_report_flow(client, report, agent_id, None))


def _observation(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"observation {index} must be a mapping")
    unknown = sorted(set(item) - _OBSERVATION_KEYS)
    if unknown:
        raise ValueError(
            f"observation {index} carries {unknown[0]}; usage reports refs and hashes only"
        )
    rendered = item.get("rendered_hash")
    if not isinstance(rendered, str) or not _SHA256.fullmatch(rendered):
        raise ValueError(
            f"observation {index} needs the sha256 rendered_hash, never the tracking input_hash"
        )
    observation = dict(item)
    overlay = item.get("overlay")
    if overlay is not None:
        if not isinstance(overlay, Mapping) or set(overlay) - _OVERLAY_KEYS:
            raise ValueError(f"observation {index} overlay carries only digest and position")
        observation["overlay"] = dict(overlay)
    return observation


def report_usage_flow(
    client: Client, agent_id: str, binding_id: str, observations: Sequence[Mapping[str, Any]]
) -> Flow[None]:
    if _engine(client) is not None:
        raise _cloud_required("bindings.report_usage")
    batch = [_observation(item, index) for index, item in enumerate(observations)]
    path = f"/v1/agents/{segment(agent_id)}/bindings/{segment(binding_id)}/usage"
    for start in range(0, len(batch), _USAGE_BATCH):
        yield Call("POST", path, {"observations": batch[start : start + _USAGE_BATCH]}, retry=True)


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PromptVersionsResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _list(
        self, prompt_id: str, cursor: Optional[str], limit: int
    ) -> Flow[Page[PromptVersionSummary]]:
        if _engine(self._client) is not None:
            raise _cloud_required("versions.list")
        params = {"limit": str(limit)}
        if cursor is not None:
            params["cursor"] = cursor
        response = yield Call(
            "GET", f"/v1/prompts/{segment(prompt_id)}/versions", params=params, retry=True
        )
        items = [
            _view(PromptVersionSummary, item, response, "version summary")
            for item in _items(response, "versions")
        ]
        return Page(items, _cursor(response))

    def list(
        self, prompt_id: str, *, cursor: Optional[str] = None, limit: int = 50
    ) -> Page[PromptVersionSummary]:
        return run_flow(self._client, self._list(prompt_id, cursor, limit))

    async def alist(
        self, prompt_id: str, *, cursor: Optional[str] = None, limit: int = 50
    ) -> Page[PromptVersionSummary]:
        return await arun_flow(self._client, self._list(prompt_id, cursor, limit))

    def _get(self, prompt_id: str, version: int) -> Flow[ManagedPromptVersion]:
        ref = parse_execution_ref(f"{prompt_id}:{version}")
        return (yield from get_flow(self._client, ref))

    def get(self, prompt_id: str, version: int) -> ManagedPromptVersion:
        return run_flow(self._client, self._get(prompt_id, version))

    async def aget(self, prompt_id: str, version: int) -> ManagedPromptVersion:
        return await arun_flow(self._client, self._get(prompt_id, version))


class PromptDraftsResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _draft(self, response: ApiResponse) -> Draft:
        draft = _view(Draft, _member(response, "draft"), response, "draft")
        _counter(response, draft.revision, "draft revision")
        return draft

    def _get(self, prompt_id: str) -> Flow[Draft]:
        engine = _engine(self._client)
        if engine is not None:
            return _view(Draft, engine.get_draft(prompt_id), None, "draft")
        response = yield Call("GET", f"/v1/prompts/{segment(prompt_id)}/draft", retry=True)
        return self._draft(response)

    def get(self, prompt_id: str) -> Draft:
        return run_flow(self._client, self._get(prompt_id))

    async def aget(self, prompt_id: str) -> Draft:
        return await arun_flow(self._client, self._get(prompt_id))

    def _save(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        base_version: Optional[int],
        expected_revision: int,
    ) -> Flow[Draft]:
        engine = _engine(self._client)
        if engine is not None:
            saved = engine.save_draft(
                prompt_id, content, base_version=base_version, expected_revision=expected_revision
            )
            return _view(Draft, saved, None, "draft")
        response = yield Call(
            "PUT",
            f"/v1/prompts/{segment(prompt_id)}/draft",
            {"base_version": base_version, "content": dict(content)},
            if_match=expected_revision,
        )
        return self._draft(response)

    def save(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        base_version: Optional[int],
        expected_revision: int,
    ) -> Draft:
        return run_flow(
            self._client, self._save(prompt_id, content, base_version, expected_revision)
        )

    async def asave(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        base_version: Optional[int],
        expected_revision: int,
    ) -> Draft:
        return await arun_flow(
            self._client, self._save(prompt_id, content, base_version, expected_revision)
        )


class PromptAliasesResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _alias(self, response: ApiResponse) -> PromptAlias:
        alias = _view(PromptAlias, _member(response, "alias"), response, "alias")
        _counter(response, alias.generation, "alias generation")
        return alias

    def _get(self, prompt_id: str, alias: str) -> Flow[PromptAlias]:
        engine = _engine(self._client)
        if engine is not None:
            return _view(PromptAlias, engine.get_alias(prompt_id, alias), None, "alias")
        response = yield Call(
            "GET", f"/v1/prompts/{segment(prompt_id)}/aliases/{segment(alias)}", retry=True
        )
        return self._alias(response)

    def get(self, prompt_id: str, alias: str) -> PromptAlias:
        return run_flow(self._client, self._get(prompt_id, alias))

    async def aget(self, prompt_id: str, alias: str) -> PromptAlias:
        return await arun_flow(self._client, self._get(prompt_id, alias))

    def _move(
        self, prompt_id: str, alias: str, version: int, expected_generation: int
    ) -> Flow[PromptAlias]:
        engine = _engine(self._client)
        if engine is not None:
            moved = engine.move_alias(
                prompt_id, alias, version=version, expected_generation=expected_generation
            )
            return _view(PromptAlias, moved, None, "alias")
        response = yield Call(
            "PUT",
            f"/v1/prompts/{segment(prompt_id)}/aliases/{segment(alias)}",
            {"version": version},
            if_match=expected_generation,
        )
        return self._alias(response)

    def move(
        self, prompt_id: str, alias: str, *, version: int, expected_generation: int
    ) -> PromptAlias:
        return run_flow(self._client, self._move(prompt_id, alias, version, expected_generation))

    async def amove(
        self, prompt_id: str, alias: str, *, version: int, expected_generation: int
    ) -> PromptAlias:
        return await arun_flow(
            self._client, self._move(prompt_id, alias, version, expected_generation)
        )


class PromptsResource:
    def __init__(self, client: Client) -> None:
        self._client = client
        self.versions = PromptVersionsResource(client)
        self.drafts = PromptDraftsResource(client)
        self.aliases = PromptAliasesResource(client)

    @property
    def local(self) -> LocalPromptEngine:
        engine = _engine(self._client)
        if engine is None:
            raise ApiError(
                "cloud_required",
                0,
                "client.prompts.local exists only without base_url; this client talks to the registry",
            )
        return engine

    def get(self, ref: Union[str, PromptReference]) -> ManagedPromptVersion:
        return run_flow(self._client, get_flow(self._client, ref))

    async def aget(self, ref: Union[str, PromptReference]) -> ManagedPromptVersion:
        return await arun_flow(self._client, get_flow(self._client, ref))

    def resolve(self, ref: Union[str, PromptAliasRef]) -> ManagedPromptVersion:
        return run_flow(self._client, get_flow(self._client, ref))

    async def aresolve(self, ref: Union[str, PromptAliasRef]) -> ManagedPromptVersion:
        return await arun_flow(self._client, get_flow(self._client, ref))

    def _pin(self, refs: Sequence[str]) -> Flow[PinnedRefs]:
        if isinstance(refs, str):
            raise TypeError("pin takes a sequence of references")
        versions: dict[str, ManagedPromptVersion] = {}
        for ref in refs:
            if ref not in versions:
                versions[ref] = yield from get_flow(self._client, ref)
        return PinnedRefs(versions)

    def pin(self, refs: Sequence[str]) -> PinnedRefs:
        return run_flow(self._client, self._pin(refs))

    async def apin(self, refs: Sequence[str]) -> PinnedRefs:
        return await arun_flow(self._client, self._pin(refs))

    def _list(
        self,
        query: Optional[str],
        tags: Sequence[str],
        kind: Optional[str],
        status: str,
        cursor: Optional[str],
        limit: int,
    ) -> Flow[Page[PromptSummary]]:
        if _engine(self._client) is not None:
            raise _cloud_required("prompts.list")
        params = {"status": status, "limit": str(limit)}
        for key, value in (("q", query), ("kind", kind), ("cursor", cursor)):
            if value is not None:
                params[key] = value
        if tags:
            params["tags"] = ",".join(tags)
        response = yield Call("GET", "/v1/prompts", params=params, retry=True)
        items = [
            _view(PromptSummary, item, response, "prompt") for item in _items(response, "prompts")
        ]
        return Page(items, _cursor(response))

    def list(
        self,
        *,
        query: Optional[str] = None,
        tags: Sequence[str] = (),
        kind: Optional[str] = None,
        status: Literal["active", "archived", "all"] = "active",
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Page[PromptSummary]:
        return run_flow(self._client, self._list(query, tags, kind, status, cursor, limit))

    async def alist(
        self,
        *,
        query: Optional[str] = None,
        tags: Sequence[str] = (),
        kind: Optional[str] = None,
        status: Literal["active", "archived", "all"] = "active",
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Page[PromptSummary]:
        return await arun_flow(self._client, self._list(query, tags, kind, status, cursor, limit))

    def _render(
        self, ref: Union[str, PromptReference], variables: Optional[Mapping[str, Any]]
    ) -> Flow[RenderResult]:
        version = yield from get_flow(self._client, ref)
        rendered = version.render(variables)
        return RenderResult(
            kind=rendered.kind,
            ref=version.ref,
            content_digest=version.content_digest,
            rendered_hash=rendered.rendered_hash,
            text=rendered.text,
            messages=rendered.messages,
        )

    def render(
        self, ref: Union[str, PromptReference], variables: Optional[Mapping[str, Any]] = None
    ) -> RenderResult:
        return run_flow(self._client, self._render(ref, variables))

    async def arender(
        self, ref: Union[str, PromptReference], variables: Optional[Mapping[str, Any]] = None
    ) -> RenderResult:
        return await arun_flow(self._client, self._render(ref, variables))

    def _resolve_agent(
        self, agent_id: str, channel: Optional[str], release_id: Optional[str]
    ) -> Flow[PromptBundle]:
        selector = _selector(channel, release_id)
        engine = _engine(self._client)
        if engine is not None:
            document = engine.resolve(agent_id, channel=channel, release_id=release_id)
            workspace_id = engine.workspace_id
            expected = document.get("prompt_manifest_digest")
        else:
            workspace_id = yield from workspace_flow(self._client)
            response = yield Call(
                "GET", f"/v1/agents/{segment(agent_id)}/resolve", params=selector, retry=True
            )
            if _member(response, "agent_id") != agent_id:
                raise _invalid(response, f"resolution of agent {agent_id}")
            document = _member(response, "artifacts")
            expected = _member(response, "prompt_manifest_digest")
        if not isinstance(document, Mapping) or not isinstance(expected, str):
            raise ApiError("invalid_response", 0, "the agent resolution carries no artifacts")
        return PromptBundle.from_online_response(
            document,
            expected_workspace_id=workspace_id,
            expected_agent_id=agent_id,
            expected_manifest_digest=expected,
        )

    def resolve_agent(
        self, agent_id: str, *, channel: Optional[str] = None, release_id: Optional[str] = None
    ) -> PromptBundle:
        return run_flow(self._client, self._resolve_agent(agent_id, channel, release_id))

    async def aresolve_agent(
        self, agent_id: str, *, channel: Optional[str] = None, release_id: Optional[str] = None
    ) -> PromptBundle:
        return await arun_flow(self._client, self._resolve_agent(agent_id, channel, release_id))

    def _export_bundle(
        self,
        agent_id: str,
        channel: Optional[str],
        release_id: Optional[str],
        expires_in_days: Optional[int],
        path: Optional[os.PathLike[str]],
        trust: Optional[BundleTrust],
    ) -> Flow[PromptBundle]:
        params = _selector(channel, release_id)
        if _engine(self._client) is not None:
            raise _cloud_required("prompts.export_bundle (use client.prompts.local.export_bundle)")
        if expires_in_days is not None:
            params["expires_in_days"] = str(expires_in_days)
        workspace_id = yield from workspace_flow(self._client)
        response = yield Call(
            "GET", f"/v1/agents/{segment(agent_id)}/prompt-bundle", params=params, retry=True
        )
        document = response.body
        if trust is None:
            issuer = document.get("issuer")
            key_id = issuer.get("key_id") if isinstance(issuer, Mapping) else None
            if not isinstance(key_id, str) or key_id == "":
                raise integrity_error(
                    "bundle_signature_invalid", "the exported bundle names no issuer key"
                )
            key = yield Call("GET", f"/v1/signing-keys/{segment(key_id)}", retry=True)
            pem = key.body.get("public_key_pem")
            if key.body.get("key_id") != key_id or not isinstance(pem, str):
                raise _invalid(key, f"signing key {key_id}")
            trust = BundleTrust.from_pems({key_id: pem})
        bundle = PromptBundle.load(
            document,
            expected_workspace_id=workspace_id,
            expected_agent_id=agent_id,
            trust=trust,
            allow_ungoverned_bundle=True,
        )
        if path is not None:
            yield partial(_write_json, Path(path), document)
        return bundle

    def export_bundle(
        self,
        agent_id: str,
        *,
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        path: Optional[os.PathLike[str]] = None,
        trust: Optional[BundleTrust] = None,
    ) -> PromptBundle:
        return run_flow(
            self._client,
            self._export_bundle(agent_id, channel, release_id, expires_in_days, path, trust),
        )

    async def aexport_bundle(
        self,
        agent_id: str,
        *,
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        path: Optional[os.PathLike[str]] = None,
        trust: Optional[BundleTrust] = None,
    ) -> PromptBundle:
        return await arun_flow(
            self._client,
            self._export_bundle(agent_id, channel, release_id, expires_in_days, path, trust),
        )

    def _create(
        self,
        prompt_id: str,
        name: str,
        kind: Literal["text", "chat", "fragment"],
        description: Optional[str],
        owner: Optional[str],
        tags: Sequence[str],
    ) -> Flow[PromptSummary]:
        engine = _engine(self._client)
        if engine is not None:
            created = engine.create_prompt(
                prompt_id, kind=kind, name=name, description=description, owner=owner, tags=tags
            )
            return _view(PromptSummary, created, None, "prompt")
        body: dict[str, Any] = {
            "prompt_id": prompt_id,
            "kind": kind,
            "name": name,
            "tags": list(tags),
        }
        if description is not None:
            body["description"] = description
        if owner is not None:
            body["owner"] = owner
        response = yield Call("POST", "/v1/prompts", body)
        prompt = _view(PromptSummary, _member(response, "prompt"), response, "prompt")
        _counter(response, prompt.metadata_revision, "metadata revision")
        return prompt

    def create(
        self,
        prompt_id: str,
        *,
        name: str,
        kind: Literal["text", "chat", "fragment"],
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> PromptSummary:
        return run_flow(self._client, self._create(prompt_id, name, kind, description, owner, tags))

    async def acreate(
        self,
        prompt_id: str,
        *,
        name: str,
        kind: Literal["text", "chat", "fragment"],
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> PromptSummary:
        return await arun_flow(
            self._client, self._create(prompt_id, name, kind, description, owner, tags)
        )

    def _publish(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        parent_version: Optional[int],
        change_message: str,
        variable_descriptions: Optional[Mapping[str, str]],
    ) -> Flow[ManagedPromptVersion]:
        engine = _engine(self._client)
        if engine is not None:
            return engine.publish(
                prompt_id,
                content,
                parent_version=parent_version,
                change_message=change_message,
                variable_descriptions=variable_descriptions,
            )
        workspace_id = yield from workspace_flow(self._client)
        body: dict[str, Any] = {
            "content": dict(content),
            "parent_version": parent_version,
            "change_message": change_message,
        }
        if variable_descriptions is not None:
            body["variable_descriptions"] = dict(variable_descriptions)
        response = yield Call(
            "POST", f"/v1/prompts/{segment(prompt_id)}/versions", body, retry=True
        )
        record = _record(_member(response, "version"), workspace_id, response)
        if record.prompt_id != prompt_id:
            raise _invalid(response, f"version of {prompt_id}")
        if record.content.get("fragments"):
            ref = PromptVersionRef(record.prompt_id, record.version)
            version = yield from version_flow(self._client, workspace_id, ref)
            _check_digest(version, record.content_digest)
            return version
        version = ManagedPromptVersion.from_record(
            record, workspace_id=workspace_id, lookup=lambda prompt_id, number: None
        )
        yield partial(_store_version, self._client.prompt_cache, workspace_id, version)
        return version

    def publish(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        parent_version: Optional[int],
        change_message: str,
        variable_descriptions: Optional[Mapping[str, str]] = None,
    ) -> ManagedPromptVersion:
        return run_flow(
            self._client,
            self._publish(
                prompt_id, content, parent_version, change_message, variable_descriptions
            ),
        )

    async def apublish(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        parent_version: Optional[int],
        change_message: str,
        variable_descriptions: Optional[Mapping[str, str]] = None,
    ) -> ManagedPromptVersion:
        return await arun_flow(
            self._client,
            self._publish(
                prompt_id, content, parent_version, change_message, variable_descriptions
            ),
        )

    def import_report(
        self,
        report: Mapping[str, Any],
        *,
        agent_id: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ImportPlan:
        return run_flow(self._client, import_report_flow(self._client, report, agent_id, options))

    async def aimport_report(
        self,
        report: Mapping[str, Any],
        *,
        agent_id: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ImportPlan:
        return await arun_flow(
            self._client, import_report_flow(self._client, report, agent_id, options)
        )

    def apply_import(
        self,
        import_id: str,
        *,
        plan_digest: str,
        items: Sequence[Mapping[str, Any]],
        mode: Literal["publish", "draft"] = "publish",
        declare_slots: bool = False,
        expected_slots_revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return run_flow(
            self._client,
            apply_import_flow(
                self._client,
                import_id,
                plan_digest=plan_digest,
                items=items,
                mode=mode,
                declare_slots=declare_slots,
                expected_slots_revision=expected_slots_revision,
                idempotency_key=idempotency_key,
                agent_id=agent_id,
            ),
        )

    async def aapply_import(
        self,
        import_id: str,
        *,
        plan_digest: str,
        items: Sequence[Mapping[str, Any]],
        mode: Literal["publish", "draft"] = "publish",
        declare_slots: bool = False,
        expected_slots_revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return await arun_flow(
            self._client,
            apply_import_flow(
                self._client,
                import_id,
                plan_digest=plan_digest,
                items=items,
                mode=mode,
                declare_slots=declare_slots,
                expected_slots_revision=expected_slots_revision,
                idempotency_key=idempotency_key,
                agent_id=agent_id,
            ),
        )

    def plan_declarations(self, document: Union[Mapping[str, Any], Source]) -> ImportPlan:
        return run_flow(self._client, plan_declarations_flow(self._client, document))

    async def aplan_declarations(self, document: Union[Mapping[str, Any], Source]) -> ImportPlan:
        return await arun_flow(self._client, plan_declarations_flow(self._client, document))

    def apply_declarations(
        self,
        document: Union[Mapping[str, Any], Source],
        *,
        plan_digest: str,
        idempotency_key: Optional[str] = None,
        expected_slots_revision: Optional[int] = None,
    ) -> dict[str, Any]:
        return run_flow(
            self._client,
            apply_declarations_flow(
                self._client,
                document,
                plan_digest=plan_digest,
                idempotency_key=idempotency_key,
                expected_slots_revision=expected_slots_revision,
            ),
        )

    async def aapply_declarations(
        self,
        document: Union[Mapping[str, Any], Source],
        *,
        plan_digest: str,
        idempotency_key: Optional[str] = None,
        expected_slots_revision: Optional[int] = None,
    ) -> dict[str, Any]:
        return await arun_flow(
            self._client,
            apply_declarations_flow(
                self._client,
                document,
                plan_digest=plan_digest,
                idempotency_key=idempotency_key,
                expected_slots_revision=expected_slots_revision,
            ),
        )

    def register_runtime(self, agent_id: str, slots: Mapping[str, Any]) -> ImportPlan:
        return run_flow(self._client, register_runtime_flow(self._client, agent_id, slots))

    async def aregister_runtime(self, agent_id: str, slots: Mapping[str, Any]) -> ImportPlan:
        return await arun_flow(self._client, register_runtime_flow(self._client, agent_id, slots))


class BindingsResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        agent_id: str,
        *,
        thread_key: str,
        scope: Literal["thread", "execution"],
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
        expect_manifest_digest: Optional[str] = None,
        runtime_client: Optional[Mapping[str, Optional[str]]] = None,
    ) -> tuple[ExecutionBinding, PromptBundle, bool]:
        return run_flow(
            self._client,
            create_binding_flow(
                self._client,
                agent_id,
                thread_key=thread_key,
                scope=scope,
                channel=channel,
                release_id=release_id,
                child_selectors=child_selectors,
                expect_manifest_digest=expect_manifest_digest,
                runtime_client=runtime_client,
            ),
        )

    async def acreate(
        self,
        agent_id: str,
        *,
        thread_key: str,
        scope: Literal["thread", "execution"],
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
        expect_manifest_digest: Optional[str] = None,
        runtime_client: Optional[Mapping[str, Optional[str]]] = None,
    ) -> tuple[ExecutionBinding, PromptBundle, bool]:
        return await arun_flow(
            self._client,
            create_binding_flow(
                self._client,
                agent_id,
                thread_key=thread_key,
                scope=scope,
                channel=channel,
                release_id=release_id,
                child_selectors=child_selectors,
                expect_manifest_digest=expect_manifest_digest,
                runtime_client=runtime_client,
            ),
        )

    def get(self, agent_id: str, binding_id: str) -> tuple[ExecutionBinding, PromptBundle]:
        return run_flow(self._client, get_binding_flow(self._client, agent_id, binding_id))

    async def aget(self, agent_id: str, binding_id: str) -> tuple[ExecutionBinding, PromptBundle]:
        return await arun_flow(self._client, get_binding_flow(self._client, agent_id, binding_id))

    def _counterfactual(
        self, agent_id: str, parent_binding_id: str, thread_key: str, release_id: str
    ) -> Flow[ExecutionBinding]:
        if _engine(self._client) is not None:
            raise _cloud_required("bindings.counterfactual")
        workspace_id = yield from workspace_flow(self._client)
        response = yield Call(
            "POST",
            f"/v1/agents/{segment(agent_id)}/bindings/{segment(parent_binding_id)}/children",
            {"thread_key": thread_key, "selector": {"release_id": release_id}},
            retry=True,
        )
        binding = _binding(_member(response, "binding"), workspace_id, agent_id)
        if (
            binding.parent_binding_id != parent_binding_id
            or binding.thread_key != thread_key
            or binding.release_id != release_id
        ):
            raise binding_error(
                "binding_mismatch",
                "the counterfactual binding differs from the request",
                binding_id=binding.binding_id,
            )
        return binding

    def counterfactual(
        self, agent_id: str, parent_binding_id: str, *, thread_key: str, release_id: str
    ) -> ExecutionBinding:
        return run_flow(
            self._client, self._counterfactual(agent_id, parent_binding_id, thread_key, release_id)
        )

    async def acounterfactual(
        self, agent_id: str, parent_binding_id: str, *, thread_key: str, release_id: str
    ) -> ExecutionBinding:
        return await arun_flow(
            self._client, self._counterfactual(agent_id, parent_binding_id, thread_key, release_id)
        )

    def report_usage(
        self, agent_id: str, binding_id: str, observations: Sequence[Mapping[str, Any]]
    ) -> None:
        run_flow(self._client, report_usage_flow(self._client, agent_id, binding_id, observations))

    async def areport_usage(
        self, agent_id: str, binding_id: str, observations: Sequence[Mapping[str, Any]]
    ) -> None:
        await arun_flow(
            self._client, report_usage_flow(self._client, agent_id, binding_id, observations)
        )


class ChannelsResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _list(self, agent_id: str) -> Flow[builtins.list[Channel]]:
        if _engine(self._client) is not None:
            raise _cloud_required("channels.list")
        response = yield Call("GET", f"/v1/agents/{segment(agent_id)}/channels", retry=True)
        return [_view(Channel, item, response, "channel") for item in _items(response, "channels")]

    def list(self, agent_id: str) -> builtins.list[Channel]:
        return run_flow(self._client, self._list(agent_id))

    async def alist(self, agent_id: str) -> builtins.list[Channel]:
        return await arun_flow(self._client, self._list(agent_id))

    def _get(self, agent_id: str, name: str) -> Flow[Channel]:
        engine = _engine(self._client)
        if engine is not None:
            state = engine.get_channel(agent_id, name)
            state.pop("history")
            return _view(Channel, state, None, "channel")
        response = yield Call(
            "GET", f"/v1/agents/{segment(agent_id)}/channels/{segment(name)}", retry=True
        )
        channel = _view(Channel, _member(response, "channel"), response, "channel")
        _counter(response, channel.generation, "channel generation")
        return channel

    def get(self, agent_id: str, name: str) -> Channel:
        return run_flow(self._client, self._get(agent_id, name))

    async def aget(self, agent_id: str, name: str) -> Channel:
        return await arun_flow(self._client, self._get(agent_id, name))

    def _history(self, agent_id: str, name: str, after: int) -> Flow[builtins.list[ChannelEvent]]:
        engine = _engine(self._client)
        if engine is not None:
            return [
                _view(ChannelEvent, event, None, "channel event")
                for event in engine.get_channel(agent_id, name)["history"]
                if event["generation"] > after
            ]
        events: builtins.list[ChannelEvent] = []
        cursor = after
        while True:
            response = yield Call(
                "GET",
                f"/v1/agents/{segment(agent_id)}/channels/{segment(name)}/history",
                params={"after": str(cursor)},
                retry=True,
            )
            events.extend(
                _view(ChannelEvent, item, response, "channel event")
                for item in _items(response, "events")
            )
            following = response.body.get("next_after")
            if following is None:
                return events
            if isinstance(following, bool) or not isinstance(following, int) or following <= cursor:
                raise _invalid(response, "next_after")
            cursor = following

    def history(self, agent_id: str, name: str, *, after: int = 0) -> builtins.list[ChannelEvent]:
        return run_flow(self._client, self._history(agent_id, name, after))

    async def ahistory(
        self, agent_id: str, name: str, *, after: int = 0
    ) -> builtins.list[ChannelEvent]:
        return await arun_flow(self._client, self._history(agent_id, name, after))

    def _move_preview(
        self, agent_id: str, name: str, release_id: Optional[str], action: str
    ) -> Flow[ChannelMovePreview]:
        if action not in ("promote", "rollback"):
            raise ValueError("action must be promote or rollback")
        if action == "promote" and release_id is None:
            raise ValueError("a promote preview needs release_id")
        if _engine(self._client) is not None:
            raise _cloud_required("channels.move_preview")
        params = {"action": action}
        if release_id is not None:
            params["release_id" if action == "promote" else "to_release_id"] = release_id
        response = yield Call(
            "GET",
            f"/v1/agents/{segment(agent_id)}/channels/{segment(name)}/move-preview",
            params=params,
            retry=True,
        )
        return _view(ChannelMovePreview, response.body, response, "move preview")

    def move_preview(
        self,
        agent_id: str,
        name: str,
        *,
        release_id: Optional[str] = None,
        action: Literal["promote", "rollback"] = "promote",
    ) -> ChannelMovePreview:
        return run_flow(self._client, self._move_preview(agent_id, name, release_id, action))

    async def amove_preview(
        self,
        agent_id: str,
        name: str,
        *,
        release_id: Optional[str] = None,
        action: Literal["promote", "rollback"] = "promote",
    ) -> ChannelMovePreview:
        return await arun_flow(self._client, self._move_preview(agent_id, name, release_id, action))
