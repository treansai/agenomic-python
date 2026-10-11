"""LangGraph: state and checkpoints carry opaque references, never a secret value."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from agenomic import Client
from agenomic.client.retry import RetryPolicy
from agenomic.crypto.canonical import canonical_cbor
from agenomic.vault import Sensitive

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

BASE = "https://api.test"
CANARY = "canary-secret-3b9e7a10-must-never-reach-a-checkpoint"
RUNTIME_TOKEN = "vrt_runtime_token_for_langgraph"
EXEC = f"{BASE}/v1/vault/runtime/executions"
STORED = re.compile(re.escape(f"{BASE}/v1/vault/runtime/executions/") + r"[0-9a-f-]{36}")
EXAMPLE = Path(__file__).parent.parent / "examples" / "12_vault_langgraph.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vault_langgraph_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


def _runtime_client() -> Client:
    return Client(
        base_url=BASE,
        runtime_token=RUNTIME_TOKEN,
        vault_retry=RetryPolicy(max_retries=1, base_delay=0.0),
    )


def _create_the_secret(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/secrets",
        status_code=201,
        json={"secret": {"id": "s-1", "name": "crm", "state": "active"}, "versions": []},
    )
    admin = Client(api_key="agm_key_for_langgraph", base_url=BASE)
    admin.vault.secrets.create(
        environment="prod",
        name="crm",
        secret_type="api_key",
        provider_id="p-1",
        value=Sensitive(CANARY),
    )


def _serialized(saver: InMemorySaver) -> list[bytes]:
    """Everything the checkpointer persists, through its own serializer and raw."""
    blobs: list[bytes] = []
    for item in saver.list(None):
        for part in (item.checkpoint, item.metadata, *(item.pending_writes or [])):
            blobs.append(saver.serde.dumps_typed(part)[1])
        blobs.append(canonical_cbor(item.checkpoint["channel_values"]))
        blobs.append(json.dumps(item.checkpoint["channel_values"], default=str).encode())
    for store in (saver.storage, saver.writes, saver.blobs):
        blobs.append(repr(store).encode())
    return blobs


def _contains(blobs: list[bytes], needle: str) -> bool:
    return any(needle.encode() in blob for blob in blobs)


def _run(
    client: Client, saver: InMemorySaver, thread: str, extra: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any], list[Any]]:
    results: list[Any] = []
    graph = example.build_graph(client, results.append, saver)
    config = {"configurable": {"thread_id": thread}}
    state = graph.invoke({"binding": "crm-read", "customer_id": "c_1", **(extra or {})}, config)
    return graph, state, results


def test_a_secret_created_with_a_canary_never_reaches_state_or_checkpoints(
    httpx_mock: HTTPXMock,
) -> None:
    _create_the_secret(httpx_mock)
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        json={
            "status": "finished",
            "action_id": "x",
            "state": "succeeded",
            "receipt_id": "rcpt-1",
            "result": {"name": "Ada"},
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=STORED,
        json={
            "status": "finished",
            "action_id": "x",
            "state": "succeeded",
            "receipt_id": "rcpt-1",
            "result": {"name": "Ada"},
        },
    )
    saver = InMemorySaver()
    graph, state, results = _run(_runtime_client(), saver, "t1")
    assert state["status"] == "done"
    assert state["receipt_id"] == "rcpt-1"
    assert results == [{"name": "Ada"}]
    assert set(state) <= example.REFERENCE_KEYS
    history = list(graph.get_state_history({"configurable": {"thread_id": "t1"}}))
    assert len(history) >= 4
    for snapshot in history:
        assert set(snapshot.values) <= example.REFERENCE_KEYS
    blobs = _serialized(saver)
    assert len(blobs) > 10
    for secret in (CANARY, RUNTIME_TOKEN):
        assert not _contains(blobs, secret)
    assert _contains(blobs, state["action_id"])
    assert _contains(blobs, "rcpt-1")
    requests = httpx_mock.get_requests()
    assert [CANARY.encode() in r.content for r in requests] == [True, False, False]


def test_the_scan_is_able_to_see_a_canary_that_does_leak() -> None:
    def leaky(state: dict[str, Any]) -> dict[str, Any]:
        return {"status": CANARY}

    graph = StateGraph(dict)
    graph.add_node("leaky", leaky)
    graph.add_edge(START, "leaky")
    graph.add_edge("leaky", END)
    saver = InMemorySaver()
    graph.compile(checkpointer=saver).invoke({}, {"configurable": {"thread_id": "leak"}})
    assert _contains(_serialized(saver), CANARY)


def test_an_unknown_outcome_leaves_only_the_action_id_in_the_checkpoint(
    httpx_mock: HTTPXMock,
) -> None:
    _create_the_secret(httpx_mock)
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        json={
            "status": "finished",
            "action_id": "x",
            "state": "outcome_unknown",
            "status_code": 504,
        },
    )
    saver = InMemorySaver()
    _, state, results = _run(_runtime_client(), saver, "t2")
    assert state["status"] == "outcome_unknown"
    assert results == []
    assert "receipt_id" not in state
    assert not _contains(_serialized(saver), CANARY)
    assert len(httpx_mock.get_requests()) == 2


def test_approval_pending_is_resumed_with_the_checkpointed_action_id(httpx_mock: HTTPXMock) -> None:
    _create_the_secret(httpx_mock)
    httpx_mock.add_response(
        method="POST",
        url=EXEC,
        status_code=202,
        json={"status": "approval_required", "action_id": "x", "approval_id": "apr-7"},
    )
    finished = {
        "status": "finished",
        "action_id": "x",
        "state": "succeeded",
        "receipt_id": "rcpt-2",
        "result": {"name": "Ada"},
    }
    httpx_mock.add_response(method="POST", url=EXEC, json=finished)
    httpx_mock.add_response(method="GET", url=STORED, json=finished)
    saver = InMemorySaver()
    client = _runtime_client()
    graph, pending, _ = _run(client, saver, "t3")
    assert (pending["status"], pending["approval_id"]) == ("approval_pending", "apr-7")
    resumed = graph.invoke(
        {"binding": "crm-read", "customer_id": "c_1", "action_id": pending["action_id"]},
        {"configurable": {"thread_id": "t3"}},
    )
    assert resumed["status"] == "done"
    assert resumed["action_id"] == pending["action_id"]
    sent = [
        json.loads(r.content)
        for r in httpx_mock.get_requests()
        if r.url.path.endswith("/executions")
    ]
    assert [body["action_id"] for body in sent] == [pending["action_id"]] * 2
    assert not _contains(_serialized(saver), CANARY)


def test_the_offline_example_runs_from_fixtures_and_a_missing_one_is_an_error() -> None:
    client, replay = example.offline_client()
    outcomes = example.run(client, "crm-read", ["c_1", "c_2", "c_3", "c_4"])
    assert outcomes == {
        "c_1": "done",
        "c_2": "approval_pending",
        "c_3": "outcome_unknown",
        "c_4": "denied",
    }
    assert [call.fixture_id for call in replay.calls] == ["fx-c_1", "fx-c_3", "fx-c_4"]
    from agenomic.vault import ReplayFixtureMissing

    with pytest.raises(ReplayFixtureMissing):
        example.run(client, "crm-read", ["c_unknown"])
