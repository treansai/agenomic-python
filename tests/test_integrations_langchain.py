"""Tests for the LangChain live-tracking callback handler."""

from __future__ import annotations

import logging
import threading
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agenomic import Client
from agenomic.exceptions import CloudError
from agenomic.integrations.langchain import (
    TrackingCallbackHandler,
    _dispatchers,
    dropped_events,
    flush,
    shutdown,
)
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
    assert flush(session) is False, "a drained-but-dropped flush is not a success"
    assert dropped_events(session) == 2
    assert session.events == []


async def test_worker_survives_an_exception_outside_the_transport_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    calls: list[str] = []

    real_event = session.event

    def flaky(event_type: str, **fields: object) -> dict[str, object]:
        calls.append(event_type)
        if event_type == "turn.started":
            raise TypeError("Object of type Foo is not JSON serializable")
        return real_event(event_type, **fields)

    monkeypatch.setattr(session, "event", flaky)
    handler = TrackingCallbackHandler(session)
    root = uuid4()
    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_chain_end({}, run_id=root)

    assert flush(session) is False
    assert handler._dispatcher._thread.is_alive(), "one bad event must not kill the worker"
    assert calls == ["turn.started", "turn.completed"], "the next event is still attempted"
    assert [e["type"] for e in session.events] == ["turn.completed"]
    assert dropped_events(session) == 1


async def test_stopping_the_session_drains_and_tears_down_the_worker() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    root = uuid4()
    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_chain_end({}, run_id=root)

    thread = handler._dispatcher._thread
    assert session.session_id in _dispatchers

    session.stop()

    assert session.session_id not in _dispatchers, "the registry must not retain stopped sessions"
    thread.join(timeout=2.0)
    assert thread.is_alive() is False, "the worker thread must exit on shutdown"
    assert [e["type"] for e in session.events] == ["turn.started", "turn.completed"]

    assert shutdown(session) is True, "shutdown is idempotent"
    assert flush(session) is True


async def test_usage_counts_one_generation_per_batch() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    root, llm = uuid4(), uuid4()
    usage = {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}
    response = LLMResult(
        generations=[
            [
                ChatGeneration(message=AIMessage(content="a", usage_metadata=usage)),
                ChatGeneration(message=AIMessage(content="b", usage_metadata=usage)),
            ]
        ]
    )

    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_llm_start(
        {}, ["p"], run_id=llm, parent_run_id=root, invocation_params={"model": "m"}
    )
    await handler.on_llm_end(response, run_id=llm, parent_run_id=root)
    assert flush(session)

    completed = next(e for e in session.events if e["type"] == "model.call.completed")
    assert completed["usage"]["input_tokens"] == 1000, "n>1 must not multiply the response usage"
    assert completed["usage"]["output_tokens"] == 200
    assert completed["usage"]["total_tokens"] == 1200


async def test_usage_skips_choices_without_metadata_and_sums_batches() -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    root, llm = uuid4(), uuid4()
    first = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    second = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    response = LLMResult(
        generations=[
            [
                ChatGeneration(message=AIMessage(content="a")),
                ChatGeneration(message=AIMessage(content="b", usage_metadata=first)),
            ],
            [ChatGeneration(message=AIMessage(content="c", usage_metadata=second))],
        ]
    )

    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_llm_start(
        {}, ["p", "q"], run_id=llm, parent_run_id=root, invocation_params={"model": "m"}
    )
    await handler.on_llm_end(response, run_id=llm, parent_run_id=root)
    assert flush(session)

    completed = next(e for e in session.events if e["type"] == "model.call.completed")
    assert completed["usage"]["input_tokens"] == 17, "each batch is a separate request"
    assert completed["usage"]["output_tokens"] == 8
    assert completed["usage"]["total_tokens"] == 25


async def test_flush_times_out_when_the_gateway_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    released = threading.Event()

    def block(event_type: str, **fields: object) -> dict[str, object]:
        released.wait(5.0)
        return {}

    monkeypatch.setattr(session, "event", block)
    handler = TrackingCallbackHandler(session)
    await handler.on_chain_start({}, {}, run_id=uuid4())
    try:
        assert flush(session, timeout=0.05) is False, "a wedged worker must not report success"
    finally:
        released.set()


async def test_stopping_a_session_that_dropped_events_logs_a_total(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    session = _session()

    def explode(event_type: str, **fields: object) -> dict[str, object]:
        raise CloudError("gateway down")

    monkeypatch.setattr(session, "event", explode)
    handler = TrackingCallbackHandler(session)
    root = uuid4()
    await handler.on_chain_start({}, {}, run_id=root)
    await handler.on_chain_end({}, run_id=root)

    with caplog.at_level(logging.WARNING, logger="agenomic.integrations.langchain"):
        session.stop()
    assert "2 event(s) dropped" in caplog.text


async def test_emitting_after_stop_is_counted_and_logged_not_queued(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session()
    handler = TrackingCallbackHandler(session)
    dispatcher = handler._dispatcher
    await handler.on_chain_start({}, {}, run_id=uuid4())
    session.stop()

    with caplog.at_level(logging.WARNING, logger="agenomic.integrations.langchain"):
        await handler.on_chain_start({}, {}, run_id=uuid4())
    assert dispatcher.failures == 1, "a closed emitter counts what it refuses"
    assert "emitter closed with the session" in caplog.text
    assert dispatcher._queue.unfinished_tasks == 0, "nothing is queued behind a dead worker"


async def test_a_handler_built_after_stop_leaves_no_worker_behind() -> None:
    session = _session()
    session.stop()

    handler = TrackingCallbackHandler(session)
    await handler.on_chain_start({}, {}, run_id=uuid4())

    assert session.session_id not in _dispatchers
    handler._dispatcher._thread.join(timeout=2.0)
    assert handler._dispatcher._thread.is_alive() is False
    assert dropped_events(session) == 0, "an unregistered emitter is not reported on the session"


async def test_a_failing_stop_callback_does_not_break_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session()
    session.on_stop(lambda: (_ for _ in ()).throw(RuntimeError("teardown exploded")))

    with caplog.at_level(logging.WARNING, logger="agenomic.tracking"):
        session.stop()
    assert session.stopped is True
    assert "stop callback failed" in caplog.text


def test_stop_callbacks_run_once_across_a_retried_stop() -> None:
    session = _session()
    calls: list[int] = []
    session.on_stop(lambda: calls.append(1))

    session.stop()
    session.stop()
    assert calls == [1]


async def test_handlers_on_one_session_share_a_single_emitter() -> None:
    session = _session()
    first, second = TrackingCallbackHandler(session), TrackingCallbackHandler(session)
    assert first._dispatcher is second._dispatcher, "one worker per session, not per handler"

    await first.on_chain_start({}, {}, run_id=uuid4())
    await second.on_chain_start({}, {}, run_id=uuid4())
    thread = first._dispatcher._thread
    session.stop()

    assert session.session_id not in _dispatchers
    thread.join(timeout=2.0)
    assert thread.is_alive() is False, "one teardown is registered, not one per handler"
