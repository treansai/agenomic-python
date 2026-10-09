"""Optional, lazy-imported integrations.

Importing this package does NOT import openai, anthropic, langgraph or mcp.
Each ``instrument_*`` helper imports its dependency lazily and raises a
helpful ImportError when the extra is not installed.
"""

from importlib import import_module
from typing import Any

from agenomic.integrations.anthropic import (
    instrument_anthropic,
    instrument_anthropic_async,
)
from agenomic.integrations.huggingface import (
    instrument_huggingface,
    trace_huggingface_call,
)
from agenomic.integrations.langgraph import instrument_langgraph
from agenomic.integrations.mcp import trace_mcp_call
from agenomic.integrations.openai import (
    instrument_openai,
    instrument_openai_async,
)

_LANGGRAPH_BINDING = frozenset(
    {
        "AgentFactory",
        "LocalBindingStore",
        "ManagedGraph",
        "PinnedPrompts",
        "bind_langgraph",
        "managed_prompt",
        "prompts_for",
        "scope_config",
    }
)


_KNOWLEDGE = frozenset({"KnowledgeRetriever", "knowledge_tool"})


def __getattr__(name: str) -> Any:
    if name in _LANGGRAPH_BINDING:
        return getattr(import_module("agenomic.integrations.langgraph_binding"), name)
    if name in _KNOWLEDGE:
        return getattr(import_module("agenomic.knowledge.tools"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "instrument_anthropic",
    "instrument_anthropic_async",
    "instrument_huggingface",
    "instrument_langgraph",
    "instrument_openai",
    "instrument_openai_async",
    "trace_huggingface_call",
    "trace_mcp_call",
]
