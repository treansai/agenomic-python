"""LangGraph state graph traced into JSONL.

Requires `pip install agenomic[langgraph]`. This example uses a tiny
mock-shaped graph object so it runs without LangGraph itself; replace with
a real ``StateGraph`` once installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agenomic.exporters.jsonl import JsonlExporter
from agenomic.integrations.langgraph import instrument_langgraph
from agenomic.trace.decorator import trace_agent_run


def upper(state: dict[str, Any]) -> dict[str, Any]:
    return {"text": state["text"].upper()}


def reverse(state: dict[str, Any]) -> dict[str, Any]:
    return {"text": state["text"][::-1]}


def main() -> None:
    # Smallest possible langgraph-compatible shim
    try:
        instrument_langgraph(SimpleNamespace(nodes={"upper": upper, "reverse": reverse}))
    except ImportError as e:
        print(f"langgraph not installed: {e}")
        return

    graph = SimpleNamespace(nodes={"upper": upper, "reverse": reverse})
    instrument_langgraph(graph)

    with JsonlExporter("/tmp/langgraph-traces.jsonl") as exporter:

        @trace_agent_run(agent_id="agent://acme/lang", exporter=exporter)
        def run(text: str) -> dict[str, Any]:
            state = {"text": text}
            state = graph.nodes["upper"](state)
            return graph.nodes["reverse"](state)

        print(run("hello"))


if __name__ == "__main__":
    main()
