from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, cast

from agenomic.exceptions import ApiError
from agenomic.prompts.bundle import BUNDLE_SCHEMA, PromptBundle
from agenomic.prompts.cache import PromptCache
from agenomic.prompts.errors import (
    PromptIntegrityError,
    RegistryUnavailableError,
    binding_error,
)
from agenomic.prompts.models import ExecutionBinding, ResolvedClosure
from agenomic.prompts.resources import (
    Flow,
    arun_flow,
    binding_bundle,
    create_binding_flow,
    get_binding_flow,
    run_flow,
    whoami_flow,
    workspace_flow,
)

if TYPE_CHECKING:
    from agenomic._client import Client
    from agenomic.prompts.local import LocalPromptEngine

__all__ = [
    "BindingAuthority",
    "CloudBindingAuthority",
    "LocalBindingAuthority",
    "bundle_from_closure",
    "counters",
    "refuse_privileged_credential",
]

logger = logging.getLogger("agenomic.prompts")

Scope = Literal["thread", "execution"]
BindingResult = tuple[ExecutionBinding, PromptBundle, bool]
OUTAGE_CACHED_BINDING = "registry_outage_cached_binding_total"
_EVICTING_STATUSES = frozenset({401, 403, 404})
_COUNTERS_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {OUTAGE_CACHED_BINDING: 0}


def counters() -> dict[str, int]:
    with _COUNTERS_LOCK:
        return dict(_COUNTERS)


def _increment(name: str) -> None:
    with _COUNTERS_LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + 1


class BindingAuthority(Protocol):
    def create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult: ...

    async def acreate_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult: ...

    def get(self, agent_id: str, binding_id: str) -> ExecutionBinding: ...

    async def aget(self, agent_id: str, binding_id: str) -> ExecutionBinding: ...

    def resolution(self, binding: ExecutionBinding) -> PromptBundle: ...

    async def aresolution(self, binding: ExecutionBinding) -> PromptBundle: ...


def _target(selector: Mapping[str, str]) -> tuple[Optional[str], Optional[str]]:
    if set(selector) not in ({"channel"}, {"release_id"}):
        raise ValueError("a selector names exactly one of channel and release_id")
    return selector.get("channel"), selector.get("release_id")


def _same_target(
    binding: ExecutionBinding, scope: Scope, channel: Optional[str], release_id: Optional[str]
) -> bool:
    resolved = binding.resolved_from
    if binding.scope != scope:
        return False
    if channel is not None:
        return bool(resolved.get("channel") == channel)
    return bool(resolved.get("release_id") == release_id)


def bundle_from_closure(binding: ExecutionBinding, closure: ResolvedClosure) -> PromptBundle:
    resolved = binding.resolved_from
    source: dict[str, Any] = (
        {"channel": resolved["channel"], "channel_generation": resolved.get("generation")}
        if "channel" in resolved
        else dict(resolved)
    )
    document = {
        "schema": BUNDLE_SCHEMA,
        "workspace_id": binding.workspace_id,
        "agent_id": binding.agent_id,
        "source": source,
        "release": {
            "release_id": binding.release_id,
            "release_name": binding.release_name,
            "genome_version": binding.genome_version,
            "bundle_id": binding.runtime.get("bundle_id"),
            "bundle_hash": binding.runtime.get("bundle_hash"),
        },
        "prompt_manifest_digest": closure.prompt_manifest_digest,
        "manifest": closure.manifest,
        "children": closure.children,
        "prompts": closure.prompts,
        "prompt_bundle_digest": closure.prompt_bundle_digest,
        "expires_at": None,
    }
    return binding_bundle(binding, document)


def refuse_privileged_credential(identity: Mapping[str, Any], allow: bool) -> None:
    scopes = identity.get("api_key_scopes")
    privileged = (
        not isinstance(scopes, list) or not scopes or bool({"write", "admin"} & set(scopes))
    )
    if privileged and not allow:
        raise binding_error(
            "privileged_credential",
            "this credential can publish or administer; execute with a read key, or pass "
            "allow_privileged_credential=True",
            api_key_scopes=scopes,
        )


def _remember(
    cache: PromptCache, workspace_id: str, binding: ExecutionBinding, bundle: PromptBundle
) -> None:
    cache.put_binding(workspace_id, binding)
    cache.put_closure(workspace_id, bundle.closure())


def _cached_pin(
    cache: PromptCache, workspace_id: str, agent_id: str, thread_key: str
) -> Optional[tuple[ExecutionBinding, ResolvedClosure]]:
    binding = cache.get_binding(workspace_id, agent_id, thread_key)
    if binding is None:
        return None
    closure = cache.get_closure(workspace_id, binding.prompt_manifest_digest)
    return None if closure is None else (binding, closure)


class CloudBindingAuthority:
    def __init__(
        self,
        client: Client,
        *,
        cache: Optional[PromptCache] = None,
        runtime_client: Optional[Mapping[str, Optional[str]]] = None,
        allow_privileged_credential: Optional[bool] = None,
    ) -> None:
        if not client.is_cloud:
            raise ValueError("CloudBindingAuthority needs a client with base_url")
        self._client = client
        self._cache = cache if cache is not None else client.prompt_cache
        self._runtime_client = dict(runtime_client or {})
        self._allow_privileged = allow_privileged_credential

    def _credential(self) -> Flow[None]:
        if self._allow_privileged is None:
            return
        identity = yield from whoami_flow(self._client)
        refuse_privileged_credential(identity, self._allow_privileged)

    def _confirm(self, agent_id: str, thread_key: str) -> Flow[None]:
        workspace_id = yield from workspace_flow(self._client)
        try:
            yield from self._credential()
        except RegistryUnavailableError:
            return
        except ApiError as error:
            if error.status in _EVICTING_STATUSES:
                yield partial(self._cache.evict_binding, workspace_id, agent_id, thread_key)
            raise

    def confirm_credential(self, agent_id: str, thread_key: str) -> None:
        run_flow(self._client, self._confirm(agent_id, thread_key))

    async def aconfirm_credential(self, agent_id: str, thread_key: str) -> None:
        await arun_flow(self._client, self._confirm(agent_id, thread_key))

    def _create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]],
    ) -> Flow[BindingResult]:
        channel, release_id = _target(selector)
        workspace_id = yield from workspace_flow(self._client)
        try:
            yield from self._credential()
            binding, bundle, created = yield from create_binding_flow(
                self._client,
                agent_id,
                thread_key=thread_key,
                scope=scope,
                channel=channel,
                release_id=release_id,
                child_selectors=child_selectors,
                runtime_client=self._runtime_client,
            )
        except RegistryUnavailableError as outage:
            return (
                yield from self._fallback(
                    workspace_id, agent_id, thread_key, scope, channel, release_id, outage
                )
            )
        except ApiError as error:
            if error.status in _EVICTING_STATUSES:
                yield partial(self._cache.evict_binding, workspace_id, agent_id, thread_key)
            raise
        yield partial(_remember, self._cache, workspace_id, binding, bundle)
        return binding, bundle, created

    def _fallback(
        self,
        workspace_id: str,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        channel: Optional[str],
        release_id: Optional[str],
        outage: RegistryUnavailableError,
    ) -> Flow[BindingResult]:
        try:
            pinned = yield partial(_cached_pin, self._cache, workspace_id, agent_id, thread_key)
            if pinned is None or not _same_target(pinned[0], scope, channel, release_id):
                raise outage
            binding = cast(ExecutionBinding, pinned[0])
            bundle = bundle_from_closure(binding, pinned[1])
        except PromptIntegrityError as error:
            raise outage from error
        logger.warning(
            "registry unavailable (%s); thread of agent %s continues on its cached binding %s",
            outage.details.get("cause"),
            agent_id,
            binding.binding_id,
        )
        _increment(OUTAGE_CACHED_BINDING)
        return binding, bundle, False

    def create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        return run_flow(
            self._client,
            self._create_or_get(agent_id, thread_key, scope, selector, child_selectors),
        )

    async def acreate_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        return await arun_flow(
            self._client,
            self._create_or_get(agent_id, thread_key, scope, selector, child_selectors),
        )

    def _get(self, agent_id: str, binding_id: str) -> Flow[ExecutionBinding]:
        workspace_id = yield from workspace_flow(self._client)
        yield from self._credential()
        binding, bundle = yield from get_binding_flow(self._client, agent_id, binding_id)
        yield partial(self._cache.put_closure, workspace_id, bundle.closure())
        return binding

    def get(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        return run_flow(self._client, self._get(agent_id, binding_id))

    async def aget(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        return await arun_flow(self._client, self._get(agent_id, binding_id))

    def _resolution(self, binding: ExecutionBinding) -> Flow[PromptBundle]:
        workspace_id = yield from workspace_flow(self._client)
        try:
            closure = yield partial(
                self._cache.get_closure, workspace_id, binding.prompt_manifest_digest
            )
        except PromptIntegrityError as error:
            if error.code != "cache_conflict":
                raise
            closure = None
        if isinstance(closure, ResolvedClosure):
            return bundle_from_closure(binding, closure)
        try:
            yield from self._credential()
            fetched, bundle = yield from get_binding_flow(
                self._client, binding.agent_id, binding.binding_id
            )
        except ApiError as error:
            if error.status in _EVICTING_STATUSES:
                yield partial(
                    self._cache.evict_binding, workspace_id, binding.agent_id, binding.thread_key
                )
            raise
        if fetched.prompt_manifest_digest != binding.prompt_manifest_digest:
            raise binding_error(
                "binding_mismatch",
                "the registry answered another manifest for this binding",
                binding_id=binding.binding_id,
            )
        yield partial(self._cache.put_closure, workspace_id, bundle.closure())
        return bundle

    def resolution(self, binding: ExecutionBinding) -> PromptBundle:
        return run_flow(self._client, self._resolution(binding))

    async def aresolution(self, binding: ExecutionBinding) -> PromptBundle:
        return await arun_flow(self._client, self._resolution(binding))


class LocalBindingAuthority:
    def __init__(self, engine: LocalPromptEngine) -> None:
        self._engine = engine

    def _artifacts(
        self, document: Mapping[str, Any], artifacts: Mapping[str, Any], agent_id: str
    ) -> tuple[ExecutionBinding, PromptBundle]:
        binding = ExecutionBinding.model_validate(document)
        if binding.workspace_id != self._engine.workspace_id or binding.agent_id != agent_id:
            raise binding_error("binding_mismatch", "the binding belongs to another agent")
        return binding, binding_bundle(binding, artifacts)

    def create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        channel, release_id = _target(selector)
        if child_selectors:
            raise ApiError(
                "cloud_required",
                0,
                "child_selectors need Agenomic Cloud; the local prompt engine does not simulate them",
            )
        document, artifacts, created = self._engine.create_binding(
            agent_id, thread_key=thread_key, scope=scope, channel=channel, release_id=release_id
        )
        binding, bundle = self._artifacts(document, artifacts, agent_id)
        return binding, bundle, created

    async def acreate_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        return self.create_or_get(agent_id, thread_key, scope, selector, child_selectors)

    def get(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        document, artifacts = self._engine.get_binding(agent_id, binding_id)
        return self._artifacts(document, artifacts, agent_id)[0]

    async def aget(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        return self.get(agent_id, binding_id)

    def resolution(self, binding: ExecutionBinding) -> PromptBundle:
        document, artifacts = self._engine.get_binding(binding.agent_id, binding.binding_id)
        fetched, bundle = self._artifacts(document, artifacts, binding.agent_id)
        if fetched.prompt_manifest_digest != binding.prompt_manifest_digest:
            raise binding_error(
                "binding_mismatch",
                "the engine holds another manifest for this binding",
                binding_id=binding.binding_id,
            )
        return bundle

    async def aresolution(self, binding: ExecutionBinding) -> PromptBundle:
        return self.resolution(binding)
