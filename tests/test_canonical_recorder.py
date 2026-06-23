"""Canonical Trace SDK — a run produces a schema-valid, hash-chained v0.3 trace."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agenomic.canonical import (
    GENESIS_PREV_EVENT_HASH,
    event_hash,
    merkle_root,
    start_run,
)


def build_trace() -> dict[str, Any]:
    run = start_run("agent://acme/support", provider="openai", model="gpt-4o")
    run.log_llm(prompt="hi", response="hello", input_tokens=2, output_tokens=1)
    run.log_tool_call(tool="search", arguments={"q": "x"}, result=[1, 2])
    run.log_policy_check(policy="pii-guard", outcome="allow")
    run.log_memory(store="kv", operation="write", key="k", value="v")
    run.request_human_review(reason="sign-off")
    return run.complete_run(output={"answer": "ok"})


def test_trace_is_v03_schema_valid(v03_errors: Callable[[dict[str, Any]], list[str]]) -> None:
    assert v03_errors(build_trace()) == []


def test_event_sequence_brackets_run() -> None:
    types = [e["type"] for e in build_trace()["events"]]
    assert types[0] == "run.started"
    assert types[-1] == "run.completed"
    for expected in ("llm.requested", "llm.responded", "tool.call.proposed", "tool.call.executed"):
        assert expected in types


def test_event_hash_chain_recomputes() -> None:
    trace = build_trace()
    prev = GENESIS_PREV_EVENT_HASH
    for ev in trace["events"]:
        assert ev["prev_event_hash"] == prev
        body = {k: v for k, v in ev.items() if k != "event_hash"}
        assert ev["event_hash"] == event_hash(body)
        prev = ev["event_hash"]
    roots = [e["event_hash"] for e in trace["events"]]
    assert trace["integrity"]["run_merkle_root"] == merkle_root(roots)


def test_completed_run_is_sealed() -> None:
    run = start_run("agent://acme/x")
    run.complete_run(output={})
    with pytest.raises(RuntimeError):
        run.log_error(message="too late")
