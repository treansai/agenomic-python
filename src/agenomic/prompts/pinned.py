from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, Optional

from agenomic.prompts.errors import binding_error
from agenomic.prompts.refs import is_uuid

if TYPE_CHECKING:
    from agenomic.prompts.bundle import PromptBundle
    from agenomic.prompts.models import ManagedPromptVersion

THREAD_KEY_DOMAIN = b"agenomic.thread_key/v1"


def _key_hash(workspace_id: str, identifier: str) -> str:
    if not is_uuid(workspace_id):
        raise ValueError("workspace_id must be a lowercase uuid")
    if not isinstance(identifier, str):
        raise ValueError("thread and execution identifiers must be strings")
    payload = (
        THREAD_KEY_DOMAIN
        + b"\0"
        + workspace_id.encode("utf-8")
        + b"\0"
        + identifier.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def thread_key(workspace_id: str, thread_id: str) -> str:
    return "thread:sha256:" + _key_hash(workspace_id, thread_id)


def execution_key(workspace_id: str, execution_id: str) -> str:
    return "exec:sha256:" + _key_hash(workspace_id, execution_id)


class PinnedPromptSet:
    __slots__ = ("_binding_id", "_bundle", "_node_children")
    _binding_id: str
    _bundle: PromptBundle
    _node_children: Mapping[str, str]

    def __init__(
        self,
        bundle: PromptBundle,
        *,
        binding_id: str,
        node_children: Optional[Mapping[str, str]] = None,
    ) -> None:
        object.__setattr__(self, "_bundle", bundle)
        object.__setattr__(self, "_binding_id", binding_id)
        object.__setattr__(self, "_node_children", MappingProxyType(dict(node_children or {})))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("PinnedPromptSet is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("PinnedPromptSet is immutable")

    def __repr__(self) -> str:
        return (
            f"PinnedPromptSet(binding_id={self._binding_id}, "
            f"digest={self._bundle.prompt_manifest_digest})"
        )

    def __copy__(self) -> PinnedPromptSet:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> PinnedPromptSet:
        return self

    def _refuse_serialization(self) -> NoReturn:
        raise binding_error(
            "prompt_set_not_serializable",
            "a pinned prompt set never leaves the process; persist the binding id instead",
            binding_id=self._binding_id,
        )

    def __reduce__(self) -> NoReturn:
        self._refuse_serialization()

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        self._refuse_serialization()

    def __getstate__(self) -> NoReturn:
        self._refuse_serialization()

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def workspace_id(self) -> str:
        return self._bundle.workspace_id

    @property
    def agent_id(self) -> str:
        return self._bundle.agent_id

    @property
    def prompt_manifest_digest(self) -> str:
        return self._bundle.prompt_manifest_digest

    @property
    def release_id(self) -> str:
        return self._bundle.release_id

    @property
    def genome_version(self) -> Optional[str]:
        value = self._bundle.release.get("genome_version")
        return value if isinstance(value, str) else None

    @property
    def children(self) -> dict[str, str]:
        return self._bundle.child_manifest_digests

    @property
    def child_agent_ids(self) -> list[str]:
        return self._bundle.child_agent_ids

    @property
    def node_children(self) -> Mapping[str, str]:
        return self._node_children

    def is_pinned(self, agent_id: str) -> bool:
        return agent_id == self.agent_id or agent_id in self.children

    def _child(self, agent_id: str) -> dict[str, Any]:
        if agent_id not in self.children:
            raise binding_error(
                "child_agent_not_pinned",
                "the agent is not pinned by this binding",
                child_agent_id=agent_id,
            )
        child: dict[str, Any] = self._bundle.document["children"][agent_id]
        return child

    def manifest_digest_for(self, agent_id: str) -> str:
        if agent_id == self.agent_id:
            return self.prompt_manifest_digest
        return str(self._child(agent_id)["prompt_manifest_digest"])

    def release_id_for(self, agent_id: str) -> str:
        if agent_id == self.agent_id:
            return self.release_id
        return str(self._child(agent_id)["release_id"])

    def genome_version_for(self, agent_id: str) -> Optional[str]:
        if agent_id == self.agent_id:
            return self.genome_version
        value = self._child(agent_id).get("genome_version")
        return value if isinstance(value, str) else None

    def agent_for_node(self, node_path: str) -> str:
        best: Optional[str] = None
        for prefix in self._node_children:
            if node_path.startswith(prefix + "|") and (best is None or len(prefix) > len(best)):
                best = prefix
        return self.agent_id if best is None else self._node_children[best]

    def version(self, slot_path: str, *, agent_id: Optional[str] = None) -> ManagedPromptVersion:
        return self._bundle.version(slot_path, agent_id=agent_id)
