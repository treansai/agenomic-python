"""Tests for LangGraph integration with a mocked StateGraph."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest

from agenomic.integrations.langgraph import instrument_langgraph
from agenomic.trace.context import set_current_recorder
from agenomic.trace.recorder import TraceRecorder


def test_module_imports_without_langgraph() -> None:
    import agenomic.integrations.langgraph as mod  # noqa: F401


def test_instrument_records_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace(graph=SimpleNamespace()))

    def my_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"out": state["in"] * 2}

    fake_graph = SimpleNamespace(nodes={"doubler": my_node})

    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        instrument_langgraph(fake_graph)
        result = fake_graph.nodes["doubler"]({"in": 21})
    finally:
        set_current_recorder(None)
    assert result == {"out": 42}
    assert len(rec.tool_calls) == 1
    tc = rec.tool_calls[0]
    assert tc.tool == "doubler"
    assert tc.protocol == "local"
    assert tc.server == "langgraph"


def test_instrument_raises_without_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", None)
    with pytest.raises(ImportError, match="agenomic\\[langgraph\\]"):
        instrument_langgraph(SimpleNamespace(nodes={}))


def test_instrument_no_recorder_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace(graph=SimpleNamespace()))
    fake_graph = SimpleNamespace(nodes={"f": lambda s: s})
    instrument_langgraph(fake_graph)
    assert fake_graph.nodes["f"]({"x": 1}) == {"x": 1}


def test_instrument_warns_when_no_node_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace(graph=SimpleNamespace()))
    runnable_like = SimpleNamespace(invoke=lambda state: state)
    graph = SimpleNamespace(
        nodes={"a": runnable_like, "b": SimpleNamespace(runnable=runnable_like)}
    )
    with caplog.at_level("WARNING", logger="agenomic.integrations.langgraph"):
        instrument_langgraph(graph)
    assert "wrapped 0 of 2 nodes" in caplog.text
    assert graph.nodes["a"] is runnable_like


def test_instrument_does_not_warn_when_nodes_are_wrapped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace(graph=SimpleNamespace()))
    with caplog.at_level("WARNING", logger="agenomic.integrations.langgraph"):
        instrument_langgraph(SimpleNamespace(nodes={"f": lambda s: s}))
        instrument_langgraph(SimpleNamespace(nodes={}))
    assert caplog.text == ""


class _State(TypedDict):
    x: int


def _real_graph(**nodes: Any) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(_State)
    previous = START
    for name, fn in nodes.items():
        graph.add_node(name, fn)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph


def _recorded(run: Any) -> Any:
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        return run(), rec
    finally:
        set_current_recorder(None)


def test_real_state_graph_nodes_are_recorded(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("langgraph")
    from langchain_core.runnables import RunnableConfig

    def inc(state: _State) -> dict[str, int]:
        return {"x": state["x"] + 1}

    def double(state: _State, config: RunnableConfig) -> dict[str, int]:
        return {"x": state["x"] * 2}

    graph = _real_graph(inc=inc, double=double)
    with caplog.at_level("WARNING", logger="agenomic.integrations.langgraph"):
        instrument_langgraph(graph)
    assert caplog.text == ""
    result, rec = _recorded(lambda: graph.compile().invoke({"x": 1}))
    assert result == {"x": 4}
    assert [tc.tool for tc in rec.tool_calls] == ["inc", "double"]
    assert all(tc.output_hash for tc in rec.tool_calls)


def test_real_state_graph_async_node_is_recorded() -> None:
    pytest.importorskip("langgraph")

    async def inc(state: _State) -> dict[str, int]:
        return {"x": state["x"] + 1}

    graph = _real_graph(inc=inc)
    instrument_langgraph(graph)
    result, rec = _recorded(lambda: asyncio.run(graph.compile().ainvoke({"x": 1})))
    assert result == {"x": 2}
    assert [tc.tool for tc in rec.tool_calls] == ["inc"]


def test_real_state_graph_failing_node_records_error() -> None:
    pytest.importorskip("langgraph")

    def boom(state: _State) -> dict[str, int]:
        raise RuntimeError("nope")

    graph = _real_graph(boom=boom)
    instrument_langgraph(graph)
    with pytest.raises(RuntimeError, match="nope"):
        _recorded(lambda: graph.compile().invoke({"x": 1}))


def test_compiled_graph_nodes_are_recorded() -> None:
    pytest.importorskip("langgraph")

    def inc(state: _State) -> dict[str, int]:
        return {"x": state["x"] + 1}

    compiled = _real_graph(inc=inc).compile()
    instrument_langgraph(compiled)
    result, rec = _recorded(lambda: compiled.invoke({"x": 1}))
    assert result == {"x": 2}
    assert [tc.tool for tc in rec.tool_calls if tc.tool == "inc"] == ["inc"]
