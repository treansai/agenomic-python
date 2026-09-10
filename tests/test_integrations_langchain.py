"""Tests for the LangChain live-tracking callback handler."""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agenomic import Client
from agenomic.exceptions import CloudError
from agenomic.integrations.langchain import TrackingCallbackHandler, flush
from agenomic.tracking import TrackingSession


def _session() -> TrackingSession:
    return Client().tracking.start(agent="agent://acme/support", environment="test")


def _llm_result(model: str = "gpt-test", tokens: tuple[int, int] = (10, 5)) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )
    return LLMResult(
        generations=[[ChatGeneration(message=message)]], llm_output={"model_name": model}
    )


async def test_turn_node_model_and_tool_form_one_hierarchy() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session, capture_turn_title=True)
    root, node, seq, llm, tool = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()

    await handler.on_chain_start(
        {"name": "LangGraph"},
        {"messages": [HumanMessage(content="  boire un verre   à Lyon ")]},
        run_id=root,
    )
    await handler.on_chain_start(
        {},
        {},
        run_id=node,
        parent_run_id=root,
        tags=["graph:step:1"],
        metadata={"langgraph_node": "orchestrator_agent"},
        name="orchestrator_agent",
    )
    await handler.on_chain_start(
        {"name": "RunnableSequence"}, {}, run_id=seq, parent_run_id=node, tags=["seq:step:1"]
    )
    await handler.on_chat_model_start(
        {},
        [[HumanMessage(content="hi")]],
        run_id=llm,
        parent_run_id=seq,
        metadata={"ls_provider": "openai", "ls_model_name": "gpt-test"},
        invocation_params={"model": "gpt-test", "temperature": 0.7},
    )
    await handler.on_llm_end(_llm_result(), run_id=llm, parent_run_id=seq)
    await handler.on_tool_start(
        {"name": "search_activities"},
        "{}",
        run_id=tool,
        parent_run_id=node,
        inputs={"query": "Lyon"},
    )
    await handler.on_tool_end({"result": []}, run_id=tool, parent_run_id=node)
    await handler.on_chain_end({}, run_id=seq, parent_run_id=node)
    await handler.on_chain_end({}, run_id=node, parent_run_id=root)
    await handler.on_chain_end({}, run_id=root)
    assert flush(session)

    events = session.events
    assert [e["type"] for e in events] == [
        "turn.started",
        "agent.step.started",
        "model.call.started",
        "model.call.completed",
        "tool.call.started",
        "tool.call.completed",
        "agent.step.completed",
        "turn.completed",
    ]
    assert [e["sequence_number"] for e in events] == list(range(8))
    turn, step, m_start, m_end, t_start, t_end, step_end, turn_end = events
    assert turn["turn_id"] == str(root)
    assert turn["turn_title"] == "boire un verre à Lyon"
    assert "span_id" not in turn
    assert "span_id" not in turn_end
    assert step["span_id"] == str(node)
    assert "parent_span_id" not in step
    assert step["workflow_step_id"] == "orchestrator_agent"
    assert m_start["parent_span_id"] == str(node), "internal runnables are skipped, not linked"
    assert m_start["model"] == {"provider": "openai", "model": "gpt-test", "temperature": 0.7}
    assert len(m_start["input_hash"]) == 64
    assert m_end["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "scope": "self",
        "source": "runtime",
    }
    assert m_end["duration_ms"] >= 0
    assert m_end["end_time"].endswith("Z")
    assert m_end["timestamp"] == m_end["end_time"]
    assert m_start["timestamp"] == m_start["start_time"]
    assert t_start["tool"] == {"name": "search_activities"}
    assert t_end["status"] == "ok"
    assert all(e["turn_id"] == str(root) and e["trace_id"] == str(root) for e in events[1:-1])
    assert turn_end["duration_ms"] >= 0


async def test_errors_become_failed_events() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    root, tool, llm = uuid4(), uuid4(), uuid4()
    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_tool_start({"name": "search"}, "x", run_id=tool, parent_run_id=root)
    await handler.on_tool_error(TimeoutError("slow"), run_id=tool, parent_run_id=root)
    await handler.on_llm_start(
        {},
        ["p"],
        run_id=llm,
        parent_run_id=root,
        invocation_params={"model": "m", "_type": "openai"},
    )
    await handler.on_llm_error(RuntimeError("boom"), run_id=llm, parent_run_id=root)
    await handler.on_chain_error(ValueError("bad"), run_id=root)
    assert flush(session)
    types = [e["type"] for e in session.events]
    assert types == [
        "turn.started",
        "tool.call.started",
        "tool.call.failed",
        "model.call.started",
        "model.call.failed",
        "turn.failed",
    ]
    assert session.events[2]["error"] == {"type": "TimeoutError"}
    assert session.events[-1]["error"] == {"type": "ValueError"}
    assert session.events[-1]["status"] == "error"


async def test_turn_title_is_off_by_default_and_unknown_runs_are_ignored() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    root = uuid4()
    await handler.on_chain_start({}, {"messages": [HumanMessage(content="secret")]}, run_id=root)
    await handler.on_chain_end({}, run_id=uuid4())
    await handler.on_llm_end(_llm_result(), run_id=uuid4())
    await handler.on_chain_end({}, run_id=root)
    assert flush(session)
    assert [e["type"] for e in session.events] == ["turn.started", "turn.completed"]
    assert "turn_title" not in session.events[0]


async def test_transport_failures_never_reach_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()

    def explode(event_type: str, **fields: object) -> dict[str, object]:
        raise CloudError("gateway down")

    monkeypatch.setattr(session, "event", explode)
    handler = TrackingCallbackHandler(session)
    root = uuid4()
    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_chain_end({}, run_id=root)
    assert flush(session)
    assert handler._dispatcher.failures == 2
    assert session.events == []
