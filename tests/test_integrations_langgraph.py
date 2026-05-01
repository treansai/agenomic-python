"""Tests for LangGraph integration with a mocked StateGraph."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from agentlock.integrations.langgraph import instrument_langgraph
from agentlock.trace.context import set_current_recorder
from agentlock.trace.recorder import TraceRecorder


def test_module_imports_without_langgraph() -> None:
    import agentlock.integrations.langgraph as mod  # noqa: F401


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
    with pytest.raises(ImportError, match="agentlock\\[langgraph\\]"):
        instrument_langgraph(SimpleNamespace(nodes={}))


def test_instrument_no_recorder_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace(graph=SimpleNamespace()))
    fake_graph = SimpleNamespace(nodes={"f": lambda s: s})
    instrument_langgraph(fake_graph)
    assert fake_graph.nodes["f"]({"x": 1}) == {"x": 1}
