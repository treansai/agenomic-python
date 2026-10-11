from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional

from agenomic.knowledge.models import VersionLike

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from agenomic._client import Client

__all__ = ["knowledge_tool"]

_LAZY = frozenset({"KnowledgeQuery", "KnowledgeRetriever", "KnowledgeTool"})


def knowledge_tool(
    knowledge_base: Optional[str] = None,
    version: Optional[VersionLike] = None,
    *,
    client: Optional[Client] = None,
    top_k: int = 5,
    agent_id: Optional[str] = None,
    execution: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    include_context: bool = True,
) -> BaseTool:
    from agenomic.knowledge._langchain import build_tool

    return build_tool(
        client=client,
        knowledge_base=knowledge_base,
        version=version,
        top_k=top_k,
        agent_id=agent_id,
        execution=execution,
        name=name,
        description=description,
        include_context=include_context,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module("agenomic.knowledge._langchain"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
