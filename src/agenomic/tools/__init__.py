"""Tool execution for replays: the Tool Gateway client, a local engine and routers.

The runtime keeps orchestrating its own agent. Each tool call is handed to
:class:`ToolRouter` (or :class:`AsyncToolRouter`), which routes a normalized
invocation to Agenomic Cloud in cloud mode or to the in-process engine in
local mode. Real backends (with server-side credentials) and the Tool Mock
Engine are chosen by the run's explicit per-tool bindings; the runtime
receives the tool's native result plus a typed technical envelope.
"""

from agenomic.tools.local import LocalToolEngine
from agenomic.tools.models import (
    TOOL_EXECUTION_SCHEMA_VERSION,
    ToolCallError,
    ToolCallResult,
    ToolCallStatus,
    ToolEffect,
    ToolExecutionError,
    ToolExternalState,
    ToolProvenance,
    ToolResultSource,
)
from agenomic.tools.resources import AsyncToolRouter, ToolRouter, ToolsResource, dumps_config

__all__ = [
    "TOOL_EXECUTION_SCHEMA_VERSION",
    "AsyncToolRouter",
    "LocalToolEngine",
    "ToolCallError",
    "ToolCallResult",
    "ToolCallStatus",
    "ToolEffect",
    "ToolExecutionError",
    "ToolExternalState",
    "ToolProvenance",
    "ToolResultSource",
    "ToolRouter",
    "ToolsResource",
    "dumps_config",
]
