from __future__ import annotations

from typing import Any

from agentlock.client import AgentLockClient
from agentlock.redaction import Redactor
from agentlock.tracing import TraceRecorder


class InstrumentedLangGraph:
    def __init__(
        self,
        graph: Any,
        client: AgentLockClient | None = None,
        agent_id: str = "langgraph-agent",
        release: str = "dev",
        redactor: Redactor | None = None,
    ) -> None:
        self.graph = graph
        self.client = client
        self.agent_id = agent_id
        self.release = release
        self.redactor = redactor

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.graph, "invoke"):
            msg = "LangGraph placeholder expects an object with invoke(...)"
            raise NotImplementedError(msg)

        trace = TraceRecorder(
            agent_id=self.agent_id,
            release=self.release,
            input=input,
            metadata={"integration": "langgraph"},
            redactor=self.redactor,
        )

        try:
            result = self.graph.invoke(input, *args, **kwargs)
        except Exception as exc:
            self._emit(trace.complete(success=False, error=type(exc).__name__))
            raise

        self._emit(trace.complete(final_output=result, success=True))
        return result

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.graph, "ainvoke"):
            msg = "LangGraph placeholder expects an object with ainvoke(...)"
            raise NotImplementedError(msg)

        trace = TraceRecorder(
            agent_id=self.agent_id,
            release=self.release,
            input=input,
            metadata={"integration": "langgraph"},
            redactor=self.redactor,
        )

        try:
            result = await self.graph.ainvoke(input, *args, **kwargs)
        except Exception as exc:
            await self._emit_async(trace.complete(success=False, error=type(exc).__name__))
            raise

        await self._emit_async(trace.complete(final_output=result, success=True))
        return result

    def _emit(self, trace: Any) -> None:
        if self.client is None:
            return
        try:
            self.client.emit_trace(trace)
        except Exception:
            return

    async def _emit_async(self, trace: Any) -> None:
        if self.client is None:
            return
        try:
            await self.client.emit_trace_async(trace)
        except Exception:
            return


def instrument_langgraph(
    graph: Any,
    client: AgentLockClient | None = None,
    agent_id: str = "langgraph-agent",
    release: str = "dev",
    redactor: Redactor | None = None,
) -> InstrumentedLangGraph:
    if not hasattr(graph, "invoke") and not hasattr(graph, "ainvoke"):
        msg = (
            "Planned LangGraph integration currently supports only graph-like "
            "objects with invoke/ainvoke"
        )
        raise NotImplementedError(msg)
    return InstrumentedLangGraph(
        graph=graph,
        client=client,
        agent_id=agent_id,
        release=release,
        redactor=redactor,
    )
