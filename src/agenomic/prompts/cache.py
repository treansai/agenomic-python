from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

from pydantic import ValidationError

from agenomic.prompts.digest import artifact_set_digest, content_digest, manifest_digest
from agenomic.prompts.errors import PromptIntegrityError, integrity_error
from agenomic.prompts.models import (
    ExecutionBinding,
    ManagedPromptVersion,
    PromptVersionRecord,
    ResolvedClosure,
)
from agenomic.prompts.refs import MAX_VERSION, is_prompt_id, is_uuid

logger = logging.getLogger("agenomic.prompts.cache")

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})", re.ASCII)
_VERSION_FILE = "agenomic.prompt_cache_version/v1"
_T = TypeVar("_T")


def _uuid(value: str) -> str:
    if not isinstance(value, str) or not is_uuid(value):
        raise ValueError("cache keys need a lowercase uuid")
    return value


def _prompt_id(value: str) -> str:
    if not isinstance(value, str) or not is_prompt_id(value):
        raise ValueError("cache keys need a valid prompt id")
    return value


def _version(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_VERSION:
        raise ValueError("cache keys need a positive prompt version")
    return value


def _hex(digest: str) -> str:
    match = _DIGEST.fullmatch(digest) if isinstance(digest, str) else None
    if match is None:
        raise ValueError("cache keys need a sha256 digest")
    return match.group(1)


def _thread_hex(thread_key: str) -> str:
    if not isinstance(thread_key, str) or thread_key == "":
        raise ValueError("cache keys need a thread key")
    return hashlib.sha256(thread_key.encode("utf-8")).hexdigest()


def _conflict(message: str, **details: Any) -> PromptIntegrityError:
    logger.warning("prompt cache conflict: %s", message)
    return integrity_error("cache_conflict", message, **details)


class PromptCache:
    def __init__(
        self,
        directory: Optional[os.PathLike[str]] = None,
        *,
        max_memory_entries: int = 4096,
        max_bindings: int = 100_000,
    ) -> None:
        self._root = Path(directory) / "v1" if directory is not None else None
        self._max_entries = max_memory_entries
        self._max_bindings = max_bindings
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._index: dict[tuple[str, str, int], str] = {}
        self._bindings: OrderedDict[tuple[str, str, str], ExecutionBinding] = OrderedDict()

    def _remember(self, key: tuple[Any, ...], value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                evicted, _ = self._entries.popitem(last=False)
                if evicted[0] == "v":
                    self._index.pop((evicted[1], evicted[2], evicted[3]), None)

    def _recall(self, key: tuple[Any, ...]) -> Any:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def _write(self, path: Path, payload: Any) -> None:
        assert self._root is not None
        directories = [self._root]
        for part in path.parent.relative_to(self._root).parts:
            directories.append(directories[-1] / part)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                os.chmod(directory, 0o700)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        os.replace(temporary, path)

    def _read(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise _conflict(f"unreadable cache file {path.name}") from error

    def _version_dir(self, workspace_id: str, prompt_id: str, version: int) -> Path:
        assert self._root is not None
        return self._root / workspace_id / "prompts" / prompt_id / str(version)

    def get_version(
        self, workspace_id: str, prompt_id: str, version: int
    ) -> Optional[ManagedPromptVersion]:
        key = (_uuid(workspace_id), _prompt_id(prompt_id), _version(version))
        with self._lock:
            digest = self._index.get(key)
        if digest is not None:
            found = self._recall(("v", *key, digest))
            if isinstance(found, ManagedPromptVersion):
                return found
        if self._root is None:
            return None
        directory = self._version_dir(*key)
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not files:
            return None
        if len(files) > 1:
            raise _conflict(f"two digests cached for {prompt_id}:{version}")
        loaded = self._load_version(workspace_id, prompt_id, version, files[0])
        self._index_version(workspace_id, loaded)
        return loaded

    def _load_version(
        self, workspace_id: str, prompt_id: str, version: int, path: Path
    ) -> ManagedPromptVersion:
        payload = self._read(path)
        try:
            if (
                payload.get("schema") != _VERSION_FILE
                or payload.get("workspace_id") != workspace_id
            ):
                raise _conflict(f"cache file {path.name} belongs to another scope")
            records = {
                record.ref: record
                for record in (
                    PromptVersionRecord.model_validate(item) for item in payload["closure"]
                )
            }
            record = records[f"{prompt_id}:{version}"]
            if "sha256:" + path.stem != record.content_digest:
                raise _conflict(f"cache file {path.name} does not match its digest")
            return ManagedPromptVersion.from_record(
                record,
                workspace_id=workspace_id,
                lookup=lambda pid, number: records.get(f"{pid}:{number}"),
            )
        except PromptIntegrityError as error:
            if error.code == "cache_conflict":
                raise
            raise _conflict(f"cache file {path.name} failed verification") from error
        except (AttributeError, KeyError, TypeError, ValidationError, ValueError) as error:
            raise _conflict(f"cache file {path.name} failed verification") from error

    def _index_version(self, workspace_id: str, version: ManagedPromptVersion) -> None:
        index_key = (workspace_id, version.ref.prompt_id, version.ref.version)
        with self._lock:
            current = self._index.get(index_key)
            if current is not None and current != version.content_digest:
                raise _conflict(f"two digests cached for {version.ref}")
            self._index[index_key] = version.content_digest
        self._remember(("v", *index_key, version.content_digest), version)

    def put_version(self, workspace_id: str, version: ManagedPromptVersion) -> None:
        _uuid(workspace_id)
        _prompt_id(version.ref.prompt_id)
        _version(version.ref.version)
        if version.workspace_id != workspace_id:
            raise _conflict(f"{version.ref} belongs to another workspace")
        version.verify()
        self._index_version(workspace_id, version)
        if self._root is None:
            return
        directory = self._version_dir(workspace_id, version.ref.prompt_id, version.ref.version)
        target = directory / f"{_hex(version.content_digest)}.json"
        others = [p for p in directory.glob("*.json") if p != target] if directory.is_dir() else []
        if others:
            raise _conflict(f"two digests cached for {version.ref}")
        payload = {
            "schema": _VERSION_FILE,
            "workspace_id": workspace_id,
            "closure": [record.model_dump(mode="json") for record in version.closure_records()],
        }
        self._write(target, payload)

    def _closure_path(self, workspace_id: str, digest: str) -> Path:
        assert self._root is not None
        return self._root / workspace_id / "closures" / f"{_hex(digest)}.json"

    def get_closure(
        self, workspace_id: str, manifest_digest_value: str
    ) -> Optional[ResolvedClosure]:
        key = ("m", _uuid(workspace_id), manifest_digest_value)
        _hex(manifest_digest_value)
        found = self._recall(key)
        if isinstance(found, ResolvedClosure):
            return found
        if self._root is None:
            return None
        path = self._closure_path(workspace_id, manifest_digest_value)
        if not path.is_file():
            return None
        try:
            closure = ResolvedClosure.model_validate(self._read(path))
        except ValidationError as error:
            raise _conflict(f"cache file {path.name} failed verification") from error
        self._check_closure(closure, manifest_digest_value)
        self._remember(key, closure)
        return closure

    def _check_closure(self, closure: ResolvedClosure, expected: str) -> None:
        try:
            consistent = (
                closure.prompt_manifest_digest == expected
                and manifest_digest(closure.manifest) == expected
                and artifact_set_digest(closure.model_dump()) == closure.prompt_bundle_digest
                and all(
                    content_digest(entry["content"]) == entry["content_digest"]
                    for entry in closure.prompts.values()
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise _conflict("cached closure failed verification") from error
        if not consistent:
            raise _conflict("cached closure failed verification")

    def put_closure(self, workspace_id: str, closure: ResolvedClosure) -> None:
        _uuid(workspace_id)
        self._check_closure(closure, closure.prompt_manifest_digest)
        self._remember(("m", workspace_id, closure.prompt_manifest_digest), closure)
        if self._root is not None:
            self._write(
                self._closure_path(workspace_id, closure.prompt_manifest_digest),
                closure.model_dump(),
            )

    def _binding_path(self, workspace_id: str, agent_id: str, thread_key: str) -> Path:
        assert self._root is not None
        return self._root / workspace_id / "bindings" / agent_id / f"{_thread_hex(thread_key)}.json"

    def get_binding(
        self, workspace_id: str, agent_id: str, thread_key: str
    ) -> Optional[ExecutionBinding]:
        key = (_uuid(workspace_id), _uuid(agent_id), thread_key)
        _thread_hex(thread_key)
        with self._lock:
            found = self._bindings.get(key)
            if found is not None:
                self._bindings.move_to_end(key)
                return found
        if self._root is None:
            return None
        path = self._binding_path(*key)
        if not path.is_file():
            return None
        try:
            binding = ExecutionBinding.model_validate(self._read(path))
        except ValidationError as error:
            raise _conflict(f"cache file {path.name} failed verification") from error
        if (binding.workspace_id, binding.agent_id, binding.thread_key) != key:
            raise _conflict(f"cache file {path.name} belongs to another binding")
        self._remember_binding(key, binding)
        return binding

    def _remember_binding(self, key: tuple[str, str, str], binding: ExecutionBinding) -> None:
        with self._lock:
            self._bindings[key] = binding
            self._bindings.move_to_end(key)
            while len(self._bindings) > self._max_bindings:
                self._bindings.popitem(last=False)

    def put_binding(self, workspace_id: str, binding: ExecutionBinding) -> None:
        _uuid(workspace_id)
        if binding.workspace_id != workspace_id:
            raise _conflict(f"binding {binding.binding_id} belongs to another workspace")
        key = (workspace_id, _uuid(binding.agent_id), binding.thread_key)
        self._remember_binding(key, binding)
        if self._root is not None:
            self._write(self._binding_path(*key), binding.to_document())

    def evict_binding(self, workspace_id: str, agent_id: str, thread_key: str) -> None:
        key = (_uuid(workspace_id), _uuid(agent_id), thread_key)
        with self._lock:
            self._bindings.pop(key, None)
        if self._root is not None:
            self._binding_path(*key).unlink(missing_ok=True)

    async def _offload(self, call: Callable[..., _T], *args: Any) -> _T:
        if self._root is None:
            return call(*args)
        return await asyncio.to_thread(call, *args)

    async def aget_version(
        self, workspace_id: str, prompt_id: str, version: int
    ) -> Optional[ManagedPromptVersion]:
        return await self._offload(self.get_version, workspace_id, prompt_id, version)

    async def aput_version(self, workspace_id: str, version: ManagedPromptVersion) -> None:
        await self._offload(self.put_version, workspace_id, version)

    async def aget_closure(
        self, workspace_id: str, manifest_digest_value: str
    ) -> Optional[ResolvedClosure]:
        return await self._offload(self.get_closure, workspace_id, manifest_digest_value)

    async def aput_closure(self, workspace_id: str, closure: ResolvedClosure) -> None:
        await self._offload(self.put_closure, workspace_id, closure)

    async def aget_binding(
        self, workspace_id: str, agent_id: str, thread_key: str
    ) -> Optional[ExecutionBinding]:
        return await self._offload(self.get_binding, workspace_id, agent_id, thread_key)

    async def aput_binding(self, workspace_id: str, binding: ExecutionBinding) -> None:
        await self._offload(self.put_binding, workspace_id, binding)

    async def aevict_binding(self, workspace_id: str, agent_id: str, thread_key: str) -> None:
        await self._offload(self.evict_binding, workspace_id, agent_id, thread_key)
