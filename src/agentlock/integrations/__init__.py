"""Optional, lazy-imported integrations.

Importing this package does NOT import openai, anthropic, langgraph or mcp.
Each ``instrument_*`` helper imports its dependency lazily and raises a
helpful ImportError when the extra is not installed.
"""
from agentlock.integrations.anthropic import (
    instrument_anthropic,
    instrument_anthropic_async,
)
from agentlock.integrations.langgraph import instrument_langgraph
from agentlock.integrations.mcp import trace_mcp_call
from agentlock.integrations.openai import (
    instrument_openai,
    instrument_openai_async,
)

__all__ = [
    "instrument_anthropic",
    "instrument_anthropic_async",
    "instrument_langgraph",
    "instrument_openai",
    "instrument_openai_async",
    "trace_mcp_call",
]
