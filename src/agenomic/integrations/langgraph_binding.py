from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import threading
import uuid
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Optional, Union, cast

import ulid
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.config import get_config_list, var_child_runnable_config
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import Command
from typing_extensions import Self, override

from agenomic._version import __version__
from agenomic.exceptions import ApiError
from agenomic.integrations.langchain_prompts import to_langchain_messages
from agenomic.prompts.authority import (
    BindingAuthority,
    CloudBindingAuthority,
    LocalBindingAuthority,
    bundle_from_closure,
    refuse_privileged_credential,
)
from agenomic.prompts.bundle import BundleTrust, PromptBundle
from agenomic.prompts.cache import PromptCache
from agenomic.prompts.digest import prompt_digest
from agenomic.prompts.errors import (
    PromptIntegrityError,
    PromptRefError,
    RegistryUnavailableError,
    api_error,
    binding_error,
    integrity_error,
    render_error,
)
from agenomic.prompts.models import ExecutionBinding, ManagedPromptVersion, Placeholder
from agenomic.prompts.pinned import PinnedPromptSet, execution_key, thread_key
from agenomic.prompts.refs import is_uuid
from agenomic.prompts.render import RenderedPrompt, render_validated

if TYPE_CHECKING:
    from agenomic._client import Client

__all__ = [
    "INFLIGHT",
    "RESERVED_KEYS",
    "TESTED_LANGCHAIN_CORE",
    "TESTED_LANGGRAPH",
    "AgenomicUntestedVersionWarning",
    "AgentFactory",
    "LocalBindingStore",
    "ManagedGraph",
    "OfflineBundleAuthority",
    "PinnedPrompts",
    "PreissuedBindingAuthority",
    "bind_langgraph",
    "counters",
    "managed_prompt",
    "prompts_for",
    "scope_config",
]

SET_KEY = "__agenomic_prompt_set"
SCOPE_KEY = "agenomic_agent_scope"
EXECUTION_KEY = "agenomic_execution_key"
PIN_SCALARS = (
    "agenomic_binding_id",
    "agenomic_prompt_manifest_digest",
    "agenomic_agent_id",
    "agenomic_release_id",
    "agenomic_genome_version",
)
EXPERIMENT_SCALARS = ("agenomic_experiment_id", "agenomic_experiment_arm_key")
SLOT_KEYS = (
    "agenomic_prompt_slots",
    "agenomic_prompt_refs",
    "agenomic_prompt_content_digests",
    "agenomic_rendered_hash",
)
RESERVED_KEYS = frozenset((*PIN_SCALARS, *EXPERIMENT_SCALARS, *SLOT_KEYS, SCOPE_KEY, SET_KEY))
TESTED_LANGGRAPH = frozenset({"1.2.11", "1.0.10"})
TESTED_LANGCHAIN_CORE = frozenset({"1.6.3"})
INFLIGHT = "langgraph_binding_inflight"
LOCAL_BINDING_SCHEMA = "agenomic.local_execution_binding/v1"
_NODE_PREFIX = re.compile(r"[A-Za-z0-9_-]+(?:\|[A-Za-z0-9_-]+)*", re.ASCII)
_CHECKPOINT_MODES = frozenset({"checkpoints", "debug"})
_MAX_PINNED_SETS = 1024
_FACTORY_SAVER_HINT = (
    "the agent factory has built no graph yet, so it cannot read the checkpoint that names "
    "this execution's binding; pass AgentFactory(build, checkpointer=saver)"
)

Scope = Literal["thread", "execution"]
BindingResult = tuple[ExecutionBinding, PromptBundle, bool]
GraphTarget = Union["PregelProtocol[Any, Any, Any, Any]", "AgentFactory"]
_Effect = tuple[Any, ...]

_GAUGE_LOCK = threading.Lock()
_GAUGE: dict[str, int] = {INFLIGHT: 0}
_WARNED: dict[str, bool] = {"done": False}


class AgenomicUntestedVersionWarning(UserWarning):
    pass


def counters() -> dict[str, int]:
    with _GAUGE_LOCK:
        return dict(_GAUGE)


def _gauge(delta: int) -> None:
    with _GAUGE_LOCK:
        _GAUGE[INFLIGHT] += delta


@contextlib.contextmanager
def _inflight() -> Iterator[None]:
    _gauge(1)
    try:
        yield
    finally:
        _gauge(-1)


def _installed(name: str) -> Optional[str]:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def _warn_untested_versions() -> None:
    with _GAUGE_LOCK:
        if _WARNED["done"]:
            return
        _WARNED["done"] = True
    langgraph_version = _installed("langgraph")
    core_version = _installed("langchain-core")
    if langgraph_version in TESTED_LANGGRAPH and core_version in TESTED_LANGCHAIN_CORE:
        return
    warnings.warn(
        f"langgraph {langgraph_version} with langchain-core {core_version} is not a tested "
        f"point of the Agenomic LangGraph adapter (tested: langgraph "
        f"{', '.join(sorted(TESTED_LANGGRAPH))}, langchain-core "
        f"{', '.join(sorted(TESTED_LANGCHAIN_CORE))})",
        AgenomicUntestedVersionWarning,
        stacklevel=3,
    )


def _runtime_client() -> dict[str, Optional[str]]:
    return {
        "sdk": "agenomic-python",
        "sdk_version": __version__,
        "adapter": "langgraph",
        "adapter_version": _installed("langgraph"),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _saver(graph: Any) -> Optional[BaseCheckpointSaver[Any]]:
    value = getattr(graph, "checkpointer", None)
    return value if isinstance(value, BaseCheckpointSaver) else None


def _corrupt(path: Path, reason: str) -> ApiError:
    return binding_error(
        "binding_store_corrupt",
        f"the local binding store file {path.name} is unusable",
        path=str(path),
        reason=reason,
    )


def _encode(binding: ExecutionBinding) -> bytes:
    body = {"schema": LOCAL_BINDING_SCHEMA, "binding": binding.to_document()}
    document = {**body, "record_digest": prompt_digest(body)}
    return json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _thread_hex(thread_key_value: str) -> str:
    return hashlib.sha256(thread_key_value.encode("utf-8")).hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


class LocalBindingStore:
    def __init__(self, directory: Optional[os.PathLike[str]] = None) -> None:
        self._root = Path(directory) if directory is not None else None
        self._lock = threading.Lock()
        self._memory: dict[tuple[str, str, str], ExecutionBinding] = {}

    @classmethod
    def in_memory(cls) -> LocalBindingStore:
        return cls()

    @property
    def directory(self) -> Optional[Path]:
        return self._root

    def _agent_dir(self, workspace_id: str, agent_id: str) -> Path:
        if not is_uuid(workspace_id) or not is_uuid(agent_id):
            raise ValueError("binding store keys need lowercase uuids")
        assert self._root is not None
        return self._root / workspace_id / agent_id

    def _path(self, workspace_id: str, agent_id: str, thread_key_value: str) -> Path:
        return self._agent_dir(workspace_id, agent_id) / f"{_thread_hex(thread_key_value)}.json"

    def _parse(self, path: Path, raw: bytes) -> ExecutionBinding:
        try:
            document = json.loads(raw.decode("utf-8"))
            if (
                not isinstance(document, dict)
                or set(document) != {"schema", "binding", "record_digest"}
                or document["schema"] != LOCAL_BINDING_SCHEMA
            ):
                raise _corrupt(path, "unexpected_shape")
            body = {"schema": document["schema"], "binding": document["binding"]}
            if prompt_digest(body) != document["record_digest"]:
                raise _corrupt(path, "digest_mismatch")
            binding = ExecutionBinding.model_validate(document["binding"])
        except ApiError:
            raise
        except ValueError as error:
            raise _corrupt(path, "unparseable") from error
        if path.stem != _thread_hex(binding.thread_key):
            raise _corrupt(path, "key_mismatch")
        return binding

    def _read(self, path: Path) -> Optional[ExecutionBinding]:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        return self._parse(path, raw)

    def get(
        self, workspace_id: str, agent_id: str, thread_key_value: str
    ) -> Optional[ExecutionBinding]:
        if self._root is None:
            with self._lock:
                return self._memory.get((workspace_id, agent_id, thread_key_value))
        path = self._path(workspace_id, agent_id, thread_key_value)
        binding = self._read(path)
        if binding is not None and (
            binding.workspace_id,
            binding.agent_id,
            binding.thread_key,
        ) != (workspace_id, agent_id, thread_key_value):
            raise _corrupt(path, "key_mismatch")
        return binding

    def _publish(self, path: Path, payload: bytes) -> bool:
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".tmp-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        handle = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        _fsync_directory(directory)
        return True

    def put(self, binding: ExecutionBinding) -> tuple[ExecutionBinding, bool]:
        key = (binding.workspace_id, binding.agent_id, binding.thread_key)
        if self._root is None:
            with self._lock:
                existing = self._memory.get(key)
                if existing is not None:
                    return existing, False
                self._memory[key] = binding
                return binding, True
        path = self._path(*key)
        if self._publish(path, _encode(binding)):
            return binding, True
        winner = self.get(*key)
        if winner is None:
            raise _corrupt(path, "missing_after_conflict")
        return winner, False

    def find(self, workspace_id: str, agent_id: str, binding_id: str) -> Optional[ExecutionBinding]:
        if self._root is None:
            with self._lock:
                for binding in self._memory.values():
                    if (binding.workspace_id, binding.agent_id, binding.binding_id) == (
                        workspace_id,
                        agent_id,
                        binding_id,
                    ):
                        return binding
            return None
        directory = self._agent_dir(workspace_id, agent_id)
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json")):
            stored = self._read(path)
            if stored is not None and stored.binding_id == binding_id:
                return stored
        return None


def _selector_compatible(
    binding: ExecutionBinding, scope: str, selector: Mapping[str, str]
) -> bool:
    resolved = binding.resolved_from
    return binding.scope == scope and all(
        resolved.get(key) == value for key, value in selector.items()
    )


def _conflict(binding: ExecutionBinding) -> ApiError:
    return api_error(
        "execution_binding_conflict",
        409,
        "the thread is already bound to another selector, scope or manifest",
        {
            "binding_id": binding.binding_id,
            "release_id": binding.release_id,
            "resolved_from": dict(binding.resolved_from),
            "prompt_manifest_digest": binding.prompt_manifest_digest,
        },
    )


class OfflineBundleAuthority:
    def __init__(
        self,
        bundle: PromptBundle,
        retained: Sequence[PromptBundle],
        store: LocalBindingStore,
        *,
        runtime_client: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        self._bundle = bundle
        self._bundles = (bundle, *retained)
        self._store = store
        self._runtime_client = dict(runtime_client or _runtime_client())

    def _bundle_for(self, binding: ExecutionBinding) -> PromptBundle:
        for candidate in self._bundles:
            if (
                candidate.release_id == binding.release_id
                and candidate.prompt_manifest_digest == binding.prompt_manifest_digest
            ):
                return candidate
        raise binding_error(
            "binding_mismatch",
            "the thread is pinned to a manifest absent from the bundle and the retained bundles",
            binding_id=binding.binding_id,
            prompt_manifest_digest=binding.prompt_manifest_digest,
        )

    def _new_binding(
        self, agent_id: str, thread_key_value: str, scope: Scope, selector: Mapping[str, str]
    ) -> ExecutionBinding:
        bundle = self._bundle
        release = bundle.release
        resolved: dict[str, Any] = (
            {"channel": selector["channel"], "generation": bundle.source.get("channel_generation")}
            if "channel" in selector
            else {"release_id": bundle.release_id}
        )
        children = {
            child_id: {
                "release_id": child.get("release_id"),
                "genome_version": child.get("genome_version"),
                "prompt_manifest_digest": child.get("prompt_manifest_digest"),
                "source": "manifest",
                "channel": None,
                "generation": None,
            }
            for child_id, child in bundle.document["children"].items()
        }
        return ExecutionBinding.model_validate(
            {
                "schema": "agenomic.execution_binding/v1",
                "binding_id": "bnd_" + ulid.new().str.lower(),
                "workspace_id": bundle.workspace_id,
                "agent_id": agent_id,
                "thread_key": thread_key_value,
                "scope": scope,
                "release_id": bundle.release_id,
                "release_name": str(release.get("release_name") or ""),
                "genome_version": release.get("genome_version"),
                "prompt_manifest_digest": bundle.prompt_manifest_digest,
                "runtime": {
                    "bundle_id": release.get("bundle_id"),
                    "bundle_hash": release.get("bundle_hash"),
                },
                "resolved_from": resolved,
                "children": children,
                "parent_binding_id": None,
                "experiment": None,
                "runtime_client": dict(self._runtime_client),
                "created_at": _utc_now(),
                "created_by": {"user_id": None, "api_key_id": None},
            }
        )

    def create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        workspace_id = self._bundle.workspace_id
        existing = self._store.get(workspace_id, agent_id, thread_key)
        created = False
        if existing is None:
            existing, created = self._store.put(
                self._new_binding(agent_id, thread_key, scope, selector)
            )
        if not _selector_compatible(existing, scope, selector):
            raise _conflict(existing)
        return existing, self._bundle_for(existing), created

    async def acreate_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        if self._store.directory is None:
            return self.create_or_get(agent_id, thread_key, scope, selector, child_selectors)
        return await asyncio.to_thread(
            self.create_or_get, agent_id, thread_key, scope, selector, child_selectors
        )

    def get(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        found = self._store.find(self._bundle.workspace_id, agent_id, binding_id)
        if found is None:
            raise api_error(
                "execution_binding_not_found", 404, f"binding {binding_id} not found", None
            )
        return found

    async def aget(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        if self._store.directory is None:
            return self.get(agent_id, binding_id)
        return await asyncio.to_thread(self.get, agent_id, binding_id)

    def resolution(self, binding: ExecutionBinding) -> PromptBundle:
        return self._bundle_for(binding)

    async def aresolution(self, binding: ExecutionBinding) -> PromptBundle:
        return self._bundle_for(binding)


def _check_resolution(binding: ExecutionBinding, resolution: PromptBundle) -> None:
    if (resolution.workspace_id, resolution.agent_id) != (binding.workspace_id, binding.agent_id):
        raise integrity_error(
            "bundle_scope_mismatch", "the resolution belongs to another workspace or agent"
        )
    if resolution.release_id != binding.release_id:
        raise binding_error(
            "binding_mismatch",
            "the resolution belongs to another release than the binding",
            binding_id=binding.binding_id,
        )
    if resolution.prompt_manifest_digest != binding.prompt_manifest_digest:
        raise integrity_error(
            "manifest_digest_mismatch",
            "the resolution pins another manifest than the binding",
            expected=binding.prompt_manifest_digest,
            actual=resolution.prompt_manifest_digest,
        )
    digests = resolution.child_manifest_digests
    for child_id, child in binding.children.items():
        if digests.get(child_id) != child.get("prompt_manifest_digest"):
            raise integrity_error(
                "manifest_digest_mismatch",
                "a child manifest differs from the binding pin",
                child_agent_id=child_id,
                expected=child.get("prompt_manifest_digest"),
                actual=digests.get(child_id),
            )


class PreissuedBindingAuthority:
    def __init__(self, binding: ExecutionBinding, resolution: PromptBundle) -> None:
        _check_resolution(binding, resolution)
        self._binding = binding
        self._resolution = resolution

    def _mismatch(self) -> ApiError:
        return binding_error(
            "binding_target_mismatch",
            "the call names another thread or agent than the pre-issued binding",
            binding_id=self._binding.binding_id,
        )

    def create_or_get(
        self,
        agent_id: str,
        thread_key: str,
        scope: Scope,
        selector: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> BindingResult:
        binding = self._binding
        if (agent_id, thread_key, scope) != (binding.agent_id, binding.thread_key, binding.scope):
            raise self._mismatch()
        return binding, self._resolution, False

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
        if (agent_id, binding_id) != (self._binding.agent_id, self._binding.binding_id):
            raise self._mismatch()
        return self._binding

    async def aget(self, agent_id: str, binding_id: str) -> ExecutionBinding:
        return self.get(agent_id, binding_id)

    def resolution(self, binding: ExecutionBinding) -> PromptBundle:
        if binding.binding_id != self._binding.binding_id:
            raise binding_error("binding_mismatch", "the resolution belongs to another binding")
        return self._resolution

    async def aresolution(self, binding: ExecutionBinding) -> PromptBundle:
        return self.resolution(binding)


def _node_path(namespace: Any) -> str:
    if not isinstance(namespace, str) or not namespace:
        return ""
    return "|".join(segment.split(":", 1)[0] for segment in namespace.split("|"))


def _pin_scalars_present(configurable: Mapping[str, Any]) -> bool:
    return any(configurable.get(key) is not None for key in PIN_SCALARS[:2])


def _consistent(value: Any, configurable: Mapping[str, Any]) -> bool:
    return (
        isinstance(value, PinnedPromptSet)
        and configurable.get("agenomic_binding_id") == value.binding_id
        and configurable.get("agenomic_prompt_manifest_digest") == value.prompt_manifest_digest
    )


def _pinned_from(config: Optional[Mapping[str, Any]]) -> tuple[PinnedPromptSet, Mapping[str, Any]]:
    configurable = (config or {}).get("configurable") or {}
    value = configurable.get(SET_KEY)
    if value is None:
        if _pin_scalars_present(configurable):
            raise binding_error(
                "prompt_set_unavailable",
                "the config carries a binding pin but no pinned prompt set; the set never "
                "crosses a serialization or process boundary, pass the node config itself",
                binding_id=configurable.get("agenomic_binding_id"),
            )
        raise binding_error(
            "binding_missing",
            "no pinned prompt set in the config; invoke the graph through bind_langgraph and "
            "pass the node config explicitly",
        )
    if not _consistent(value, configurable):
        raise binding_error(
            "binding_mismatch", "the pinned prompt set disagrees with the binding pin"
        )
    return cast(PinnedPromptSet, value), configurable


def _scoped_agent(
    pinned: PinnedPromptSet, configurable: Mapping[str, Any], agent_id: Optional[str]
) -> str:
    scoped = agent_id or configurable.get(SCOPE_KEY)
    if not isinstance(scoped, str):
        scoped = pinned.agent_for_node(_node_path(configurable.get("checkpoint_ns")))
    if not pinned.is_pinned(scoped):
        raise binding_error(
            "child_agent_not_pinned",
            "the agent is not pinned by this binding",
            child_agent_id=scoped,
        )
    return scoped


class PinnedPrompts:
    __slots__ = ("_agent_id", "_config", "_pinned", "_rendered")

    def __init__(
        self, pinned: PinnedPromptSet, agent_id: str, config: Optional[Mapping[str, Any]] = None
    ) -> None:
        self._pinned = pinned
        self._agent_id = agent_id
        self._config: Mapping[str, Any] = config or {}
        self._rendered: dict[str, str] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def workspace_id(self) -> str:
        return self._pinned.workspace_id

    @property
    def binding_id(self) -> str:
        return self._pinned.binding_id

    @property
    def prompt_manifest_digest(self) -> str:
        return self._pinned.manifest_digest_for(self._agent_id)

    @property
    def release_id(self) -> Optional[str]:
        return self._pinned.release_id_for(self._agent_id)

    @property
    def genome_version(self) -> Optional[str]:
        return self._pinned.genome_version_for(self._agent_id)

    @property
    def pinned_set(self) -> PinnedPromptSet:
        return self._pinned

    def version(self, slot_path: str) -> ManagedPromptVersion:
        return self._pinned.version(slot_path, agent_id=self._agent_id)

    def _render(
        self,
        slot_path: str,
        variables: Optional[Mapping[str, Any]],
        *,
        history: Optional[Sequence[Any]] = None,
        expect_kind: Optional[Literal["text", "chat"]] = None,
    ) -> RenderedPrompt:
        result = render_validated(
            self.version(slot_path)._validated(),
            variables,
            history=history,
            expect_kind=expect_kind,
        )
        self._rendered[slot_path] = result.rendered_hash
        return result

    def render_text(self, slot_path: str, variables: Optional[Mapping[str, Any]] = None) -> str:
        return cast(str, self._render(slot_path, variables, expect_kind="text").text)

    def render_messages(
        self, slot_path: str, variables: Optional[Mapping[str, Any]] = None
    ) -> list[BaseMessage]:
        result = self._render(slot_path, variables, expect_kind="chat")
        return to_langchain_messages(result.messages or [])

    def compose(
        self,
        slot_path: str,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        history: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        result = self._render(slot_path, variables, history=list(history))
        return to_langchain_messages(result.messages or [])

    def config_for(self, *slot_paths: str) -> RunnableConfig:
        if not slot_paths:
            raise ValueError("config_for needs at least one slot path")
        versions = [self.version(slot_path) for slot_path in slot_paths]
        metadata = dict(self._config.get("metadata") or {})
        metadata["agenomic_prompt_slots"] = ",".join(slot_paths)
        metadata["agenomic_prompt_refs"] = ",".join(str(version.ref) for version in versions)
        metadata["agenomic_prompt_content_digests"] = ",".join(
            version.content_digest for version in versions
        )
        rendered = [self._rendered[slot] for slot in slot_paths if slot in self._rendered]
        if len(rendered) == 1:
            metadata["agenomic_rendered_hash"] = rendered[0]
        else:
            metadata.pop("agenomic_rendered_hash", None)
        return cast(RunnableConfig, {**self._config, "metadata": metadata})


def prompts_for(config: RunnableConfig, *, agent_id: Optional[str] = None) -> PinnedPrompts:
    pinned, configurable = _pinned_from(config)
    return PinnedPrompts(pinned, _scoped_agent(pinned, configurable, agent_id), config)


def scope_config(
    config: RunnableConfig, agent_id: str, *, thread_id: Optional[str] = None
) -> RunnableConfig:
    pinned, configurable = _pinned_from(config)
    _scoped_agent(pinned, configurable, agent_id)
    if thread_id is None:
        return cast(
            RunnableConfig,
            {**config, "configurable": {**configurable, SCOPE_KEY: agent_id}},
        )
    source = cast(Mapping[str, Any], config)
    fresh: dict[str, Any] = {
        key: source[key] for key in ("callbacks", "tags", "recursion_limit") if key in source
    }
    metadata = source.get("metadata") or {}
    fresh["metadata"] = {
        key: metadata[key] for key in (*PIN_SCALARS, *EXPERIMENT_SCALARS) if key in metadata
    }
    scoped = {key: configurable[key] for key in PIN_SCALARS if key in configurable}
    fresh["configurable"] = {
        "thread_id": thread_id,
        **scoped,
        SCOPE_KEY: agent_id,
        SET_KEY: pinned,
    }
    return cast(RunnableConfig, fresh)


def managed_prompt(
    slot_path: str,
    *,
    variables: Union[
        Mapping[str, Any], Callable[[Mapping[str, Any]], Mapping[str, Any]], None
    ] = None,
    history_key: str = "messages",
    agent_id: Optional[str] = None,
) -> Runnable[Mapping[str, Any], list[BaseMessage]]:
    def render(state: Mapping[str, Any], config: RunnableConfig) -> list[BaseMessage]:
        prompts = prompts_for(config, agent_id=agent_id)
        version = prompts.version(slot_path)
        values = dict(variables(state) if callable(variables) else (variables or {}))
        history = list(state.get(history_key) or [])
        if version.content.kind == "text":
            return [SystemMessage(prompts.render_text(slot_path, values)), *history]
        placeholders = [
            entry.placeholder for entry in version.content.body if isinstance(entry, Placeholder)
        ]
        if not placeholders:
            return prompts.compose(slot_path, values, history=history)
        if history_key not in placeholders:
            raise render_error({"code": "history_conflict", "variable": history_key})
        return prompts.render_messages(slot_path, {**values, history_key: history})

    return RunnableLambda(render, name=f"agenomic_prompt:{slot_path}")


FactoryInput = Union[PinnedPromptSet, PinnedPrompts]


def _factory_key(prompts: FactoryInput) -> tuple[str, str, str, str]:
    if isinstance(prompts, PinnedPrompts):
        return (
            prompts.workspace_id,
            prompts.agent_id,
            prompts.prompt_manifest_digest,
            prompts.genome_version or "",
        )
    return (
        prompts.workspace_id,
        prompts.agent_id,
        prompts.prompt_manifest_digest,
        prompts.genome_version or "",
    )


def _topology_mismatch() -> ApiError:
    return binding_error(
        "factory_topology_mismatch",
        "the agent factory built a graph with other nodes or another checkpointer, so "
        "threads could not move between its graphs",
    )


class AgentFactory:
    def __init__(
        self,
        build: Callable[[FactoryInput], PregelProtocol[Any, Any, Any, Any]],
        *,
        checkpointer: Optional[BaseCheckpointSaver[Any]] = None,
        max_entries: int = 32,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if checkpointer is not None and not isinstance(checkpointer, BaseCheckpointSaver):
            raise TypeError(
                "checkpointer must be the checkpointer of every graph the factory builds"
            )
        self._build = build
        self._checkpointer = checkpointer
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._graphs: OrderedDict[tuple[str, str, str, str], Any] = OrderedDict()
        self._building: dict[tuple[str, str, str, str], threading.Lock] = {}
        self._reference: Optional[Any] = None

    @property
    def checkpointer(self) -> Optional[BaseCheckpointSaver[Any]]:
        return self._checkpointer

    @property
    def reference(self) -> Optional[PregelProtocol[Any, Any, Any, Any]]:
        with self._lock:
            return cast(Optional[PregelProtocol[Any, Any, Any, Any]], self._reference)

    def _recall(self, key: tuple[str, str, str, str]) -> Optional[Any]:
        with self._lock:
            graph = self._graphs.get(key)
            if graph is not None:
                self._graphs.move_to_end(key)
            return graph

    def _check_topology(self, graph: Any) -> None:
        if self._checkpointer is not None and _saver(graph) is not self._checkpointer:
            raise _topology_mismatch()
        reference = self._reference
        if reference is None:
            return
        nodes = getattr(graph, "nodes", None)
        expected = getattr(reference, "nodes", None)
        same_nodes = (
            set(nodes) == set(expected) if nodes is not None and expected is not None else True
        )
        if not same_nodes or _saver(graph) is not _saver(reference):
            raise _topology_mismatch()

    def get(self, prompts: FactoryInput) -> PregelProtocol[Any, Any, Any, Any]:
        key = _factory_key(prompts)
        found = self._recall(key)
        if found is not None:
            return cast(PregelProtocol[Any, Any, Any, Any], found)
        with self._lock:
            building = self._building.setdefault(key, threading.Lock())
        with building:
            found = self._recall(key)
            if found is not None:
                return cast(PregelProtocol[Any, Any, Any, Any], found)
            graph = self._build(prompts)
            with self._lock:
                self._check_topology(graph)
                if self._reference is None:
                    self._reference = graph
                self._graphs[key] = graph
                while len(self._graphs) > self._max_entries:
                    self._graphs.popitem(last=False)
                self._building.pop(key, None)
            return graph


def _refuse_reserved(config: Optional[Mapping[str, Any]], where: str) -> None:
    if not config:
        return
    for member in ("configurable", "metadata"):
        values = config.get(member) or {}
        for key in sorted(RESERVED_KEYS & set(values)):
            raise binding_error(
                "agenomic_reserved_key",
                f"{key} is set by the managed graph and cannot come from {where}",
                key=key,
            )


def _inherited_set() -> bool:
    parent = var_child_runnable_config.get()
    if not isinstance(parent, Mapping):
        return False
    configurable = parent.get("configurable")
    return isinstance(configurable, Mapping) and isinstance(
        configurable.get(SET_KEY), PinnedPromptSet
    )


def _clean_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        configurable = value.get("configurable")
        if isinstance(configurable, Mapping) and SET_KEY in configurable:
            return {
                **value,
                "configurable": {k: v for k, v in configurable.items() if k != SET_KEY},
            }
    return value


def _strip_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    changed: dict[str, Any] = {}
    for key in ("config", "parent_config"):
        if key in data:
            clean = _clean_config(data[key])
            if clean is not data[key]:
                changed[key] = clean
    payload = data.get("payload")
    if isinstance(payload, dict):
        clean_payload = _strip_data(payload)
        if clean_payload is not payload:
            changed["payload"] = clean_payload
    return {**data, **changed} if changed else data


def _identity(chunk: Any) -> Any:
    return chunk


def _stream_filter(kwargs: Mapping[str, Any], graph: Any) -> Callable[[Any], Any]:
    mode = kwargs.get("stream_mode")
    if mode is None:
        mode = getattr(graph, "stream_mode", None) or "values"
    modes = {mode} if isinstance(mode, str) else set(cast(Sequence[str], mode))
    if not modes & _CHECKPOINT_MODES:
        return _identity
    if kwargs.get("version") == "v2":

        def strip_part(chunk: Any) -> Any:
            if isinstance(chunk, dict) and chunk.get("type") in _CHECKPOINT_MODES:
                return {**chunk, "data": _strip_data(chunk.get("data"))}
            return chunk

        return strip_part
    listed = not isinstance(mode, str)

    def strip(chunk: Any) -> Any:
        if isinstance(chunk, tuple) and chunk:
            if listed and (len(chunk) < 2 or chunk[-2] not in _CHECKPOINT_MODES):
                return chunk
            return (*chunk[:-1], _strip_data(chunk[-1]))
        return _strip_data(chunk)

    return strip


def _strip_result(result: Any, kwargs: Mapping[str, Any], graph: Any) -> Any:
    if "stream_mode" not in kwargs or not isinstance(result, list):
        return result
    strip = _stream_filter(kwargs, graph)
    return result if strip is _identity else [strip(chunk) for chunk in result]


def _is_resume(input: Any) -> bool:
    return input is None or (isinstance(input, Command) and input.resume is not None)


@dataclass(frozen=True)
class _Admitted:
    config: RunnableConfig
    graph: Any


class _Binder:
    def __init__(
        self,
        *,
        authority: BindingAuthority,
        agent_id: str,
        workspace_id: str,
        pin_scope: Scope,
        channel: Optional[str],
        release_id: Optional[str],
        preissued_binding_id: Optional[str],
        children: Mapping[str, str],
        child_selectors: Optional[Mapping[str, Mapping[str, str]]],
        cache: PromptCache,
        revalidate: Literal["per_invocation", "never"],
    ) -> None:
        self.authority = authority
        self.agent_id = agent_id
        self.workspace_id = workspace_id
        self.pin_scope = pin_scope
        self.channel = channel
        self.release_id = release_id
        self.preissued_binding_id = preissued_binding_id
        self.children = dict(children)
        self.child_selectors = (
            {key: dict(value) for key, value in child_selectors.items()}
            if child_selectors
            else None
        )
        self.cache = cache
        self.revalidate = revalidate
        self.remember = revalidate == "never" and not isinstance(authority, CloudBindingAuthority)
        self._lock = threading.Lock()
        self._sets: OrderedDict[tuple[str, str, str], PinnedPromptSet] = OrderedDict()

    @property
    def selector(self) -> dict[str, str]:
        if self.channel is not None:
            return {"channel": self.channel}
        if self.release_id is not None:
            return {"release_id": self.release_id}
        return {}

    def key(self, kind: Literal["thread", "exec"], identifier: Any) -> str:
        text = identifier if isinstance(identifier, str) else str(identifier)
        if self.preissued_binding_id is not None:
            return text
        if kind == "thread":
            return thread_key(self.workspace_id, text)
        return execution_key(self.workspace_id, text)

    def check(self, binding: ExecutionBinding, bundle: PromptBundle) -> None:
        if (binding.workspace_id, binding.agent_id) != (self.workspace_id, self.agent_id):
            raise binding_error(
                "binding_mismatch",
                "the binding belongs to another workspace or agent",
                binding_id=binding.binding_id,
            )
        if (
            bundle.release_id != binding.release_id
            or bundle.prompt_manifest_digest != binding.prompt_manifest_digest
        ):
            raise binding_error(
                "binding_mismatch",
                "the pinned artifacts belong to another release or manifest than the binding",
                binding_id=binding.binding_id,
            )
        resolved = dict(binding.resolved_from)
        if self.preissued_binding_id is not None:
            matches = binding.binding_id == self.preissued_binding_id
        elif self.channel is not None:
            matches = resolved.get("channel") == self.channel
        else:
            matches = resolved == {"release_id": self.release_id}
        if not matches or binding.scope != self.pin_scope:
            raise binding_error(
                "binding_target_mismatch",
                "the binding was created for another target than this managed graph",
                binding_id=binding.binding_id,
                resolved_from=resolved,
            )

    def cached(self, thread_key_value: str) -> Optional[tuple[ExecutionBinding, PromptBundle]]:
        try:
            binding = self.cache.get_binding(self.workspace_id, self.agent_id, thread_key_value)
            if binding is None:
                return None
            closure = self.cache.get_closure(self.workspace_id, binding.prompt_manifest_digest)
        except PromptIntegrityError as error:
            if error.code != "cache_conflict":
                raise
            return None
        if closure is None:
            return None
        return binding, bundle_from_closure(binding, closure)

    async def acached(
        self, thread_key_value: str
    ) -> Optional[tuple[ExecutionBinding, PromptBundle]]:
        try:
            binding = await self.cache.aget_binding(
                self.workspace_id, self.agent_id, thread_key_value
            )
            if binding is None:
                return None
            closure = await self.cache.aget_closure(
                self.workspace_id, binding.prompt_manifest_digest
            )
        except PromptIntegrityError as error:
            if error.code != "cache_conflict":
                raise
            return None
        if closure is None:
            return None
        return binding, bundle_from_closure(binding, closure)

    def confirm(self, thread_key_value: str) -> None:
        if isinstance(self.authority, CloudBindingAuthority):
            self.authority.confirm_credential(self.agent_id, thread_key_value)

    async def aconfirm(self, thread_key_value: str) -> None:
        if isinstance(self.authority, CloudBindingAuthority):
            await self.authority.aconfirm_credential(self.agent_id, thread_key_value)

    def store(self, binding: ExecutionBinding, bundle: PromptBundle) -> None:
        self.cache.put_binding(self.workspace_id, binding)
        self.cache.put_closure(self.workspace_id, bundle.closure())

    async def astore(self, binding: ExecutionBinding, bundle: PromptBundle) -> None:
        await self.cache.aput_binding(self.workspace_id, binding)
        await self.cache.aput_closure(self.workspace_id, bundle.closure())

    def pinned_set(self, binding: ExecutionBinding, bundle: PromptBundle) -> PinnedPromptSet:
        key = (binding.binding_id, binding.prompt_manifest_digest, bundle.prompt_bundle_digest)
        with self._lock:
            found = self._sets.get(key)
            if found is not None:
                self._sets.move_to_end(key)
                return found
        pinned = bundle.pinned_set(binding_id=binding.binding_id, node_children=self.children)
        with self._lock:
            self._sets[key] = pinned
            while len(self._sets) > _MAX_PINNED_SETS:
                self._sets.popitem(last=False)
        return pinned


def _inject(
    merged: dict[str, Any],
    configurable: dict[str, Any],
    metadata: dict[str, Any],
    binding: ExecutionBinding,
    pinned: PinnedPromptSet,
    agent_id: str,
) -> RunnableConfig:
    scalars: dict[str, str] = {
        "agenomic_binding_id": binding.binding_id,
        "agenomic_prompt_manifest_digest": binding.prompt_manifest_digest,
        "agenomic_agent_id": agent_id,
        "agenomic_release_id": binding.release_id,
    }
    if binding.genome_version:
        scalars["agenomic_genome_version"] = binding.genome_version
    configurable.update(scalars)
    configurable[SET_KEY] = pinned
    metadata.update(scalars)
    experiment = binding.experiment
    if isinstance(experiment, Mapping):
        for source, target in zip(("experiment_id", "arm_key"), EXPERIMENT_SCALARS, strict=True):
            value = experiment.get(source)
            if isinstance(value, str):
                metadata[target] = value
    merged["configurable"] = configurable
    merged["metadata"] = metadata
    return cast(RunnableConfig, merged)


def _prompt_set_unavailable(message: str) -> ApiError:
    return binding_error("prompt_set_unavailable", message)


class ManagedGraph(PregelProtocol[Any, Any, Any, Any]):
    def __init__(
        self,
        binder: _Binder,
        *,
        inner: Optional[Any] = None,
        factory: Optional[AgentFactory] = None,
        overrides: tuple[tuple[Optional[RunnableConfig], dict[str, Any]], ...] = (),
    ) -> None:
        self._binder = binder
        self._inner = inner
        self._factory = factory
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:
            factory = self.__dict__.get("_factory")
            reference = factory.reference if isinstance(factory, AgentFactory) else None
            if reference is None:
                raise AttributeError(
                    f"{name}: prompt_set_unavailable, the agent factory has built no graph yet"
                )
            inner = self._apply(reference)
        return getattr(inner, name)

    @property
    def agent_id(self) -> str:
        return self._binder.agent_id

    @property
    def workspace_id(self) -> str:
        return self._binder.workspace_id

    @property
    def pin_scope(self) -> Scope:
        return self._binder.pin_scope

    def _apply(self, graph: Any) -> Any:
        for config, kwargs in self._overrides:
            graph = graph.with_config(config, **kwargs)
        return graph

    def _graph_for(self, pinned: PinnedPromptSet, agent_id: str) -> Any:
        if self._factory is None:
            return self._inner
        target: FactoryInput = (
            pinned if agent_id == pinned.agent_id else PinnedPrompts(pinned, agent_id)
        )
        return self._apply(self._factory.get(target))

    def _bound_configurable(self) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        sources = (
            [getattr(self._inner, "config", None)]
            if self._factory is None
            else [{**(config or {}), **kwargs} for config, kwargs in self._overrides]
        )
        for source in sources:
            if isinstance(source, Mapping):
                bound.update(source.get("configurable") or {})
        return bound

    def _any_graph(self) -> Any:
        if self._factory is None:
            return self._inner
        reference = self._factory.reference
        if reference is None:
            raise _prompt_set_unavailable(
                "the agent factory has built no graph yet; invoke the managed graph first"
            )
        return self._apply(reference)

    def _checkpoint_saver(self) -> Optional[BaseCheckpointSaver[Any]]:
        if self._factory is None:
            return _saver(self._inner)
        if self._factory.checkpointer is not None:
            return self._factory.checkpointer
        if self._factory.reference is None:
            raise _prompt_set_unavailable(_FACTORY_SAVER_HINT)
        return _saver(self._any_graph())

    def _admission(
        self, input: Any, config: Optional[RunnableConfig], *, state_update: bool
    ) -> Generator[_Effect, Any, _Admitted]:
        binder = self._binder
        merged: dict[str, Any] = dict(config or {})
        configurable: dict[str, Any] = dict(merged.get("configurable") or {})
        metadata: dict[str, Any] = dict(merged.get("metadata") or {})
        parent = configurable.get(SET_KEY)
        if _consistent(parent, configurable):
            pinned = cast(PinnedPromptSet, parent)
            if pinned.workspace_id != binder.workspace_id:
                raise binding_error(
                    "binding_mismatch", "the enclosing managed run belongs to another workspace"
                )
            if not pinned.is_pinned(binder.agent_id):
                raise binding_error(
                    "child_agent_not_pinned",
                    "the agent is not pinned by the enclosing binding",
                    child_agent_id=binder.agent_id,
                )
            configurable[SCOPE_KEY] = binder.agent_id
            merged["configurable"] = configurable
            return _Admitted(cast(RunnableConfig, merged), self._graph_for(pinned, binder.agent_id))
        if parent is None and _inherited_set():
            raise binding_error(
                "nested_bind_unsupported",
                "a managed graph was invoked inside a managed run without the run config; pass "
                "the node config, or scope_config(config, agent_id, thread_id=...)",
                agent_id=binder.agent_id,
            )
        _refuse_reserved(merged, "the caller")
        located = {**self._bound_configurable(), **configurable}
        thread_id = located.get("thread_id")
        if thread_id is not None:
            configurable["thread_id"] = thread_id
        binding: Optional[ExecutionBinding] = None
        bundle: Optional[PromptBundle] = None
        created = False
        key = ""
        if binder.pin_scope == "thread":
            if thread_id is None:
                raise binding_error(
                    "thread_id_required", "thread scope needs configurable.thread_id"
                )
            key = binder.key("thread", thread_id)
        elif thread_id is not None and (state_update or _is_resume(input)):
            saver = self._checkpoint_saver()
            location: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
            if located.get("checkpoint_id") is not None:
                location["checkpoint_id"] = located["checkpoint_id"]
            found = None if saver is None else (yield ("checkpoint", saver, location))
            binding_id = found.get("agenomic_binding_id") if isinstance(found, Mapping) else None
            if not isinstance(binding_id, str):
                raise binding_error(
                    "execution_binding_unrecoverable",
                    "no binding id in the latest checkpoint of this thread",
                )
            binding = cast(ExecutionBinding, (yield ("get", binding_id)))
            if binding.agent_id != binder.agent_id or binding.scope != "execution":
                raise binding_error(
                    "binding_mismatch",
                    "the checkpoint points to a binding of another agent or scope",
                    binding_id=binding_id,
                )
            bundle = cast(PromptBundle, (yield ("resolution", binding)))
        else:
            identifier = located.get(EXECUTION_KEY)
            if identifier is None or identifier == "":
                raise binding_error(
                    "execution_key_required",
                    "execution scope needs configurable.agenomic_execution_key, the same value "
                    "for every retry of one logical request",
                )
            key = binder.key("exec", identifier)
        if binding is None and binder.revalidate == "never":
            found_pin = yield ("cached", key)
            if found_pin is not None:
                binding, bundle = found_pin
        if binding is None:
            binding, bundle, created = yield ("create", key)
            if binder.remember:
                yield ("store", binding, bundle)
        assert binding is not None
        assert bundle is not None
        binder.check(binding, bundle)
        pinned = binder.pinned_set(binding, bundle)
        graph = self._graph_for(pinned, binder.agent_id)
        if created and binder.pin_scope == "thread":
            saver = _saver(graph)
            latest = (
                None
                if saver is None
                else (yield ("checkpoint", saver, {"thread_id": thread_id, "checkpoint_ns": ""}))
            )
            stamp = (
                latest.get("agenomic_prompt_manifest_digest")
                if isinstance(latest, Mapping)
                else None
            )
            if stamp is not None and stamp != binding.prompt_manifest_digest:
                raise binding_error(
                    "binding_checkpoint_mismatch",
                    "the thread's latest checkpoint was written under another prompt manifest",
                    binding_id=binding.binding_id,
                    checkpoint_digest=stamp,
                )
        return _Admitted(
            _inject(merged, configurable, metadata, binding, pinned, binder.agent_id), graph
        )

    def _effect(self, effect: _Effect) -> Any:
        binder = self._binder
        kind = effect[0]
        if kind == "create":
            return binder.authority.create_or_get(
                binder.agent_id,
                effect[1],
                binder.pin_scope,
                binder.selector,
                binder.child_selectors,
            )
        if kind == "get":
            return binder.authority.get(binder.agent_id, effect[1])
        if kind == "resolution":
            return binder.authority.resolution(effect[1])
        if kind == "checkpoint":
            found = effect[1].get_tuple({"configurable": effect[2]})
            return None if found is None else found.metadata
        if kind == "cached":
            found = binder.cached(effect[1])
            if found is not None:
                binder.confirm(effect[1])
            return found
        binder.store(effect[1], effect[2])
        return None

    async def _aeffect(self, effect: _Effect) -> Any:
        binder = self._binder
        kind = effect[0]
        if kind == "create":
            return await binder.authority.acreate_or_get(
                binder.agent_id,
                effect[1],
                binder.pin_scope,
                binder.selector,
                binder.child_selectors,
            )
        if kind == "get":
            return await binder.authority.aget(binder.agent_id, effect[1])
        if kind == "resolution":
            return await binder.authority.aresolution(effect[1])
        if kind == "checkpoint":
            found = await effect[1].aget_tuple({"configurable": effect[2]})
            return None if found is None else found.metadata
        if kind == "cached":
            found = await binder.acached(effect[1])
            if found is not None:
                await binder.aconfirm(effect[1])
            return found
        await binder.astore(effect[1], effect[2])
        return None

    def _admit(
        self, input: Any, config: Optional[RunnableConfig], *, state_update: bool = False
    ) -> _Admitted:
        flow = self._admission(input, config, state_update=state_update)
        try:
            effect = next(flow)
            while True:
                effect = flow.send(self._effect(effect))
        except StopIteration as stop:
            return cast(_Admitted, stop.value)

    async def _aadmit(
        self, input: Any, config: Optional[RunnableConfig], *, state_update: bool = False
    ) -> _Admitted:
        flow = self._admission(input, config, state_update=state_update)
        try:
            effect = next(flow)
            while True:
                effect = flow.send(await self._aeffect(effect))
        except StopIteration as stop:
            return cast(_Admitted, stop.value)

    @override
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        admitted = self._admit(input, config)
        with _inflight():
            result = admitted.graph.invoke(input, admitted.config, **kwargs)
        return _strip_result(result, kwargs, admitted.graph)

    @override
    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        admitted = await self._aadmit(input, config)
        with _inflight():
            result = await admitted.graph.ainvoke(input, admitted.config, **kwargs)
        return _strip_result(result, kwargs, admitted.graph)

    @override
    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Iterator[Any]:
        admitted = self._admit(input, config)
        strip = _stream_filter(kwargs, admitted.graph)
        iterator = admitted.graph.stream(input, admitted.config, **kwargs)
        _gauge(1)
        try:
            for chunk in iterator:
                yield strip(chunk)
        finally:
            _gauge(-1)
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    @override
    async def astream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        admitted = await self._aadmit(input, config)
        strip = _stream_filter(kwargs, admitted.graph)
        iterator = admitted.graph.astream(input, admitted.config, **kwargs)
        _gauge(1)
        try:
            async for chunk in iterator:
                yield strip(chunk)
        finally:
            _gauge(-1)
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                await aclose()

    @override
    def stream_events(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        admitted = self._admit(input, config)
        return admitted.graph.stream_events(input, admitted.config, **kwargs)

    @override
    def astream_events(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        **kwargs: Any,
    ) -> Any:
        if version == "v3":
            return self._astream_events_v3(input, config, kwargs)
        return self._astream_events(input, config, version, kwargs)

    async def _astream_events_v3(
        self, input: Any, config: Optional[RunnableConfig], kwargs: dict[str, Any]
    ) -> Any:
        admitted = await self._aadmit(input, config)
        return await admitted.graph.astream_events(input, admitted.config, version="v3", **kwargs)

    async def _astream_events(
        self,
        input: Any,
        config: Optional[RunnableConfig],
        version: str,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        admitted = await self._aadmit(input, config)
        iterator = admitted.graph.astream_events(input, admitted.config, version=version, **kwargs)
        _gauge(1)
        try:
            async for event in iterator:
                yield event
        finally:
            _gauge(-1)
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                await aclose()

    @override
    def update_state(
        self,
        config: RunnableConfig,
        values: Any,
        as_node: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> RunnableConfig:
        admitted = self._admit(None, config, state_update=True)
        extra = {} if task_id is None else {"task_id": task_id}
        return cast(
            RunnableConfig, admitted.graph.update_state(admitted.config, values, as_node, **extra)
        )

    @override
    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: Any,
        as_node: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> RunnableConfig:
        admitted = await self._aadmit(None, config, state_update=True)
        extra = {} if task_id is None else {"task_id": task_id}
        return cast(
            RunnableConfig,
            await admitted.graph.aupdate_state(admitted.config, values, as_node, **extra),
        )

    @override
    def bulk_update_state(self, config: RunnableConfig, updates: Sequence[Any]) -> RunnableConfig:
        admitted = self._admit(None, config, state_update=True)
        return cast(RunnableConfig, admitted.graph.bulk_update_state(admitted.config, updates))

    @override
    async def abulk_update_state(
        self, config: RunnableConfig, updates: Sequence[Any]
    ) -> RunnableConfig:
        admitted = await self._aadmit(None, config, state_update=True)
        return cast(
            RunnableConfig, await admitted.graph.abulk_update_state(admitted.config, updates)
        )

    def _state_graph(self, config: RunnableConfig) -> Any:
        if self._factory is None or self._factory.reference is not None:
            return self._any_graph()
        if self._binder.pin_scope != "thread" and self._factory.checkpointer is None:
            raise _prompt_set_unavailable(_FACTORY_SAVER_HINT)
        return self._admit(None, config, state_update=True).graph

    async def _astate_graph(self, config: RunnableConfig) -> Any:
        if self._factory is None or self._factory.reference is not None:
            return self._any_graph()
        if self._binder.pin_scope != "thread" and self._factory.checkpointer is None:
            raise _prompt_set_unavailable(_FACTORY_SAVER_HINT)
        return (await self._aadmit(None, config, state_update=True)).graph

    @override
    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> Any:
        return self._state_graph(config).get_state(config, subgraphs=subgraphs)

    @override
    async def aget_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> Any:
        graph = await self._astate_graph(config)
        return await graph.aget_state(config, subgraphs=subgraphs)

    @override
    def get_state_history(self, config: RunnableConfig, **kwargs: Any) -> Iterator[Any]:
        return cast(Iterator[Any], self._state_graph(config).get_state_history(config, **kwargs))

    @override
    async def aget_state_history(self, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Any]:
        graph = await self._astate_graph(config)
        async for snapshot in graph.aget_state_history(config, **kwargs):
            yield snapshot

    @override
    def get_graph(self, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        return self._any_graph().get_graph(config, **kwargs)

    @override
    async def aget_graph(self, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        return await self._any_graph().aget_graph(config, **kwargs)

    @property
    @override
    def InputType(self) -> Any:
        return self._any_graph().InputType

    @property
    @override
    def OutputType(self) -> Any:
        return self._any_graph().OutputType

    @override
    def get_input_schema(self, config: Optional[RunnableConfig] = None) -> Any:
        return self._any_graph().get_input_schema(config)

    @override
    def get_output_schema(self, config: Optional[RunnableConfig] = None) -> Any:
        return self._any_graph().get_output_schema(config)

    @property
    @override
    def config_specs(self) -> Any:
        return self._any_graph().config_specs

    @override
    def config_schema(self, *, include: Optional[Sequence[str]] = None) -> Any:
        return self._any_graph().config_schema(include=include)

    @override
    def with_config(self, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Self:
        _refuse_reserved(config, "with_config")
        _refuse_reserved(kwargs, "with_config")
        if self._factory is not None:
            return type(self)(
                self._binder,
                factory=self._factory,
                overrides=(*self._overrides, (config, dict(kwargs))),
            )
        return type(self)(self._binder, inner=self._any_graph().with_config(config, **kwargs))

    def copy(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("use with_config: a copy of the inner graph would drop the binding")

    @override
    def with_retry(self, **kwargs: Any) -> Runnable[Any, Any]:
        retry = super().with_retry(**kwargs)
        if self._binder.pin_scope != "execution":
            return retry
        return _ExecutionKeyed(retry)


def _keyed(config: Optional[RunnableConfig]) -> RunnableConfig:
    merged: dict[str, Any] = dict(config or {})
    configurable = dict(merged.get("configurable") or {})
    if configurable.get(EXECUTION_KEY) is None:
        configurable[EXECUTION_KEY] = str(uuid.uuid4())
    merged["configurable"] = configurable
    return cast(RunnableConfig, merged)


class _ExecutionKeyed(Runnable[Any, Any]):
    def __init__(self, bound: Runnable[Any, Any]) -> None:
        self.bound = bound

    @override
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        return self.bound.invoke(input, _keyed(config), **kwargs)

    @override
    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        return await self.bound.ainvoke(input, _keyed(config), **kwargs)

    @override
    def batch(
        self,
        inputs: list[Any],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Any]:
        configs = [_keyed(item) for item in get_config_list(config, len(inputs))]
        return self.bound.batch(inputs, configs, return_exceptions=return_exceptions, **kwargs)

    @override
    async def abatch(
        self,
        inputs: list[Any],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Any]:
        configs = [_keyed(item) for item in get_config_list(config, len(inputs))]
        return await self.bound.abatch(
            inputs, configs, return_exceptions=return_exceptions, **kwargs
        )


def _load_offline(
    bundle: Union[PromptBundle, str, os.PathLike[str], Mapping[str, Any]],
    *,
    workspace_id: str,
    agent_id: str,
    trust: Optional[BundleTrust],
    expected_bundle_digest: Optional[str],
    expected_manifest_digest: Optional[str],
    allow_ungoverned_bundle: bool,
) -> PromptBundle:
    if isinstance(bundle, PromptBundle):
        if (bundle.workspace_id, bundle.agent_id) != (workspace_id, agent_id):
            raise integrity_error(
                "bundle_scope_mismatch", "the bundle belongs to another workspace or agent"
            )
        if (
            expected_manifest_digest is not None
            and bundle.prompt_manifest_digest != expected_manifest_digest
        ):
            raise integrity_error(
                "manifest_digest_mismatch",
                "the manifest digest differs from the expected digest",
                expected=expected_manifest_digest,
                actual=bundle.prompt_manifest_digest,
            )
        return bundle
    if trust is None and expected_bundle_digest is None:
        raise ValueError("offline mode needs trust or expected_bundle_digest to load a bundle")
    return PromptBundle.load(
        bundle,
        expected_workspace_id=workspace_id,
        expected_agent_id=agent_id,
        trust=trust,
        expected_bundle_digest=expected_bundle_digest,
        expected_manifest_digest=expected_manifest_digest,
        allow_ungoverned_bundle=allow_ungoverned_bundle,
    )


def _load_retained(
    retained: Sequence[Union[PromptBundle, tuple[Union[str, os.PathLike[str]], str]]],
    *,
    workspace_id: str,
    agent_id: str,
    trust: Optional[BundleTrust],
    allow_ungoverned_bundle: bool,
) -> list[PromptBundle]:
    loaded: list[PromptBundle] = []
    for entry in retained:
        if isinstance(entry, PromptBundle):
            loaded.append(
                _load_offline(
                    entry,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    trust=None,
                    expected_bundle_digest=None,
                    expected_manifest_digest=None,
                    allow_ungoverned_bundle=allow_ungoverned_bundle,
                )
            )
        elif isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
            loaded.append(
                PromptBundle.load(
                    entry[0],
                    expected_workspace_id=workspace_id,
                    expected_agent_id=agent_id,
                    trust=trust,
                    expected_bundle_digest=entry[1],
                    allow_ungoverned_bundle=allow_ungoverned_bundle,
                )
            )
        else:
            raise ValueError(
                "retained_bundles takes loaded PromptBundle objects or "
                "(path, expected_bundle_digest) pairs"
            )
    return loaded


def _check_children(children: Optional[Mapping[str, str]]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for prefix, child in (children or {}).items():
        if not isinstance(prefix, str) or _NODE_PREFIX.fullmatch(prefix) is None:
            raise ValueError(f"children key {prefix!r} is not a node path prefix")
        if not isinstance(child, str) or not is_uuid(child):
            raise ValueError("children values are lowercase child agent uuids")
        checked[prefix] = child
    return checked


def _check_child_selectors(
    child_selectors: Optional[Mapping[str, Mapping[str, str]]],
) -> Optional[dict[str, dict[str, str]]]:
    if not child_selectors:
        return None
    checked: dict[str, dict[str, str]] = {}
    for child, selector in child_selectors.items():
        if not is_uuid(child) or set(selector) not in ({"channel"}, {"release_id"}):
            raise ValueError(
                "child_selectors maps a child agent uuid to {'channel': name} or "
                "{'release_id': uuid}"
            )
        checked[child] = dict(selector)
    return checked


def _check_privileged(client: Client, allow: bool) -> None:
    try:
        identity = client.whoami()
    except RegistryUnavailableError as outage:
        if client.workspace_id is not None:
            return
        raise RegistryUnavailableError(
            "registry_unavailable",
            outage.status,
            "the registry is unavailable and the client has no workspace_id; set workspace_id "
            "(AGENOMIC_WORKSPACE_ID) to bind during an outage and resume cached threads",
            outage.details,
        ) from outage
    refuse_privileged_credential(identity, allow)


def _online_authority(
    client: Client,
    *,
    workspace_id: Optional[str],
    allow_privileged_credential: bool,
    child_selectors: Optional[Mapping[str, Mapping[str, str]]],
    cache: PromptCache,
) -> tuple[BindingAuthority, str]:
    if client.is_cloud:
        _check_privileged(client, allow_privileged_credential)
        known = client.workspace_id
        authority: BindingAuthority = CloudBindingAuthority(
            client,
            cache=cache,
            runtime_client=_runtime_client(),
            allow_privileged_credential=allow_privileged_credential,
        )
    else:
        if child_selectors:
            raise ApiError(
                "cloud_required",
                0,
                "child_selectors need Agenomic Cloud; the local prompt engine does not simulate "
                "them",
            )
        engine = client.prompts.local
        known = engine.workspace_id
        authority = LocalBindingAuthority(engine)
    if known is None:
        raise ApiError("invalid_response", 0, "the workspace of the client is unknown")
    if workspace_id is not None and workspace_id != known:
        raise PromptRefError(
            "workspace_mismatch",
            0,
            "workspace_id differs from the workspace of the client",
        )
    return authority, known


def bind_langgraph(
    graph: GraphTarget,
    *,
    client: Optional[Client] = None,
    agent_id: str,
    channel: Optional[str] = None,
    release_id: Optional[str] = None,
    pin_scope: Scope = "thread",
    bundle: Union[PromptBundle, str, os.PathLike[str], Mapping[str, Any], None] = None,
    retained_bundles: Sequence[Union[PromptBundle, tuple[Union[str, os.PathLike[str]], str]]] = (),
    offline: bool = False,
    trust: Optional[BundleTrust] = None,
    expected_bundle_digest: Optional[str] = None,
    expected_manifest_digest: Optional[str] = None,
    allow_ungoverned_bundle: bool = False,
    allow_privileged_credential: bool = False,
    workspace_id: Optional[str] = None,
    binding_store: Optional[LocalBindingStore] = None,
    children: Optional[Mapping[str, str]] = None,
    child_selectors: Optional[Mapping[str, Mapping[str, str]]] = None,
    cache: Optional[PromptCache] = None,
    revalidate: Literal["per_invocation", "never"] = "per_invocation",
    binding: Union[ExecutionBinding, Mapping[str, Any], None] = None,
    resolution: Optional[PromptBundle] = None,
) -> ManagedGraph:
    if isinstance(graph, ManagedGraph):
        raise ValueError("the graph is already bound; bind only the root graph")
    if not isinstance(graph, (PregelProtocol, AgentFactory)):
        raise TypeError("bind_langgraph takes a compiled LangGraph graph or an AgentFactory")
    if not isinstance(agent_id, str) or not is_uuid(agent_id):
        raise ValueError("agent_id must be the lowercase agents.id uuid")
    if pin_scope not in ("thread", "execution"):
        raise ValueError("pin_scope must be thread or execution")
    if revalidate not in ("per_invocation", "never"):
        raise ValueError("revalidate must be per_invocation or never")
    node_children = _check_children(children)
    selectors = _check_child_selectors(child_selectors)
    if isinstance(graph, PregelProtocol):
        _refuse_reserved(getattr(graph, "config", None), "the inner graph config")
    _warn_untested_versions()
    offline_only = (
        bundle is not None
        or bool(retained_bundles)
        or trust is not None
        or expected_bundle_digest is not None
        or expected_manifest_digest is not None
        or allow_ungoverned_bundle
        or binding_store is not None
    )
    preissued_id: Optional[str] = None
    authority: BindingAuthority
    if (binding is None) != (resolution is None):
        raise ValueError("binding and resolution are given together")
    if binding is not None and resolution is not None:
        if offline or offline_only or client is not None or channel or release_id:
            raise ValueError("a pre-issued binding takes no client, target or bundle")
        issued = (
            binding
            if isinstance(binding, ExecutionBinding)
            else ExecutionBinding.model_validate(binding)
        )
        if issued.agent_id != agent_id:
            raise binding_error(
                "binding_target_mismatch",
                "the pre-issued binding belongs to another agent",
                binding_id=issued.binding_id,
            )
        if issued.scope != pin_scope:
            raise ValueError("pin_scope must equal the scope of the pre-issued binding")
        authority = PreissuedBindingAuthority(issued, resolution)
        resolved_workspace = issued.workspace_id
        preissued_id = issued.binding_id
        store_cache = cache if cache is not None else PromptCache()
    else:
        if (channel is None) == (release_id is None):
            raise ValueError("name exactly one of channel and release_id")
        if offline:
            if client is not None:
                raise ValueError("offline mode takes no client")
            if bundle is None or workspace_id is None or not is_uuid(workspace_id):
                raise ValueError("offline mode needs bundle and the workspace_id uuid")
            if selectors:
                raise ValueError("child_selectors need an online binding")
            loaded = _load_offline(
                bundle,
                workspace_id=workspace_id,
                agent_id=agent_id,
                trust=trust,
                expected_bundle_digest=expected_bundle_digest,
                expected_manifest_digest=expected_manifest_digest,
                allow_ungoverned_bundle=allow_ungoverned_bundle,
            )
            if (channel is not None and loaded.source.get("channel") != channel) or (
                release_id is not None and loaded.release_id != release_id
            ):
                raise binding_error(
                    "binding_target_mismatch",
                    "the bundle was exported for another channel or release",
                    source=dict(loaded.source),
                    release_id=loaded.release_id,
                )
            retained = _load_retained(
                retained_bundles,
                workspace_id=workspace_id,
                agent_id=agent_id,
                trust=trust,
                allow_ungoverned_bundle=allow_ungoverned_bundle,
            )
            if binding_store is None:
                if (
                    isinstance(graph, AgentFactory)
                    or _saver(graph) is not None
                    or (getattr(graph, "checkpointer", None) is True)
                ):
                    raise ValueError(
                        "a graph with a checkpointer needs LocalBindingStore(directory), so "
                        "offline pins survive a restart"
                    )
                binding_store = LocalBindingStore.in_memory()
            authority = OfflineBundleAuthority(loaded, retained, binding_store)
            resolved_workspace = workspace_id
            store_cache = cache if cache is not None else PromptCache()
        else:
            if offline_only:
                raise ValueError(
                    "bundle, retained_bundles, trust, expected digests, "
                    "allow_ungoverned_bundle and binding_store apply to offline mode only"
                )
            if client is None:
                raise ValueError("online mode needs a client; pass offline=True with a bundle")
            store_cache = cache if cache is not None else client.prompt_cache
            authority, resolved_workspace = _online_authority(
                client,
                workspace_id=workspace_id,
                allow_privileged_credential=allow_privileged_credential,
                child_selectors=selectors,
                cache=store_cache,
            )
    binder = _Binder(
        authority=authority,
        agent_id=agent_id,
        workspace_id=resolved_workspace,
        pin_scope=pin_scope,
        channel=channel,
        release_id=release_id,
        preissued_binding_id=preissued_id,
        children=node_children,
        child_selectors=selectors,
        cache=store_cache,
        revalidate=revalidate,
    )
    if isinstance(graph, AgentFactory):
        return ManagedGraph(binder, factory=graph)
    return ManagedGraph(binder, inner=graph)
