from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

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

    def __init__(
        self,
        bundle: PromptBundle,
        *,
        binding_id: str,
        node_children: Optional[Mapping[str, str]] = None,
    ) -> None:
        routes = dict(node_children or {})
        pinned = set(bundle.child_agent_ids)
        for node_path, child in routes.items():
            if child not in pinned:
                raise binding_error(
                    "child_agent_not_pinned",
                    f"node {node_path} names a child agent absent from the binding",
                    child_agent_id=child,
                )
        self._bundle = bundle
        self._binding_id = binding_id
        self._node_children = routes

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
    def children(self) -> dict[str, str]:
        return self._bundle.child_manifest_digests

    def agent_for_node(self, node_path: str) -> str:
        return self._node_children.get(node_path, self.agent_id)

    def version(self, slot_path: str, *, agent_id: Optional[str] = None) -> ManagedPromptVersion:
        return self._bundle.version(slot_path, agent_id=agent_id)
