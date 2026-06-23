"""Canonical Trace SDK — the LangGraph adapter auto-emits canonical events."""

from __future__ import annotations

from typing import Any

import pytest

from agenomic.canonical import start_run
from agenomic.integrations.langgraph import instrument_langgraph_canonical


class FakeGraph:
    """Duck-typed stand-in for a compiled LangGraph (a `.nodes` mapping)."""

    def __init__(self, nodes: dict[str, Any]) -> None:
        self.nodes = nodes


def test_adapter_maps_nodes_to_canonical_events_without_manual_instrumentation() -> None:
    run = start_run("agent://acme/x")

    # Plain node functions — no Agenomic calls inside them.
    def agent(state: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "search", **state}

    def search(state: dict[str, Any]) -> dict[str, Any]:
        return {"results": [1, 2]}

    graph = FakeGraph({"agent": agent, "search": search})
    instrument_langgraph_canonical(graph, run, llm_nodes=["agent"])

    # Drive the (instrumented) graph as the engine would.
    s1 = graph.nodes["agent"]({"q": "x"})
    graph.nodes["search"](s1)
    trace = run.complete_run(output={"done": True})

    types = [e["type"] for e in trace["events"]]
    assert types[0] == "run.started"
    assert types[-1] == "run.completed"
    # The agent (LLM) node fires before the tool node.
    assert types.index("llm.requested") < types.index("tool.call.proposed")
    assert "tool.call.executed" in types


def test_adapter_records_node_errors() -> None:
    run = start_run("agent://acme/x")

    def boom(state: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("node failed")

    graph = FakeGraph({"boom": boom})
    instrument_langgraph_canonical(graph, run)
    with pytest.raises(ValueError):
        graph.nodes["boom"]({})
    trace = run.complete_run(output={}, status="error")
    assert any(e["type"] == "error.raised" for e in trace["events"])
