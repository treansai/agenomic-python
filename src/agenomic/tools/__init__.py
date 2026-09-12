"""Tool execution for replays: the Tool Gateway client and a lightweight router.

The runtime keeps orchestrating its own agent. Each tool call is handed to
:class:`ToolRouter`, which posts a normalized invocation to Agenomic Cloud.
The cloud routes it to a real backend (with server-side credentials) or to
the Tool Mock Engine according to the run's explicit per-tool bindings, and
returns the tool's native result plus a technical envelope.
"""

from agenomic.tools.resources import (
    TOOL_EXECUTION_SCHEMA_VERSION,
    ToolCallError,
    ToolCallResult,
    ToolExecutionError,
    ToolRouter,
    ToolsResource,
)

__all__ = [
    "TOOL_EXECUTION_SCHEMA_VERSION",
    "ToolCallError",
    "ToolCallResult",
    "ToolExecutionError",
    "ToolRouter",
    "ToolsResource",
]
