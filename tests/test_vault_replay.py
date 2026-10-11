"""Replay: mock by default, a missing fixture is an explicit error, never a live call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from agenomic import Client
from agenomic.tools import ToolExecutionError
from agenomic.vault import (
    ReplayFixture,
    ReplayFixtureMissing,
    ReplayOutcome,
    ReplayUnsupported,
    VaultApprovalRequired,
    VaultConflict,
    VaultExecutionFailed,
    VaultNotFound,
    VaultOutcomeUnknown,
    VaultPolicyDenied,
    VaultReplay,
    VaultValidationError,
)

BASE = "https://api.test"
ACTION = "0a1b2c3d-0000-4000-8000-000000000001"
OTHER = "0a1b2c3d-0000-4000-8000-000000000002"
CANARY = "pii-canary-in-the-arguments-5521"


def _fixture(outcome: ReplayOutcome | None = None, **kw: Any) -> ReplayFixture:
    options: dict[str, Any] = {
        "fixture_id": "fx-1",
        "tool": "crm.get_customer",
        "binding": "crm-read",
        "arguments": {"id": "c_1"},
        "outcome": outcome
        or ReplayOutcome(result={"name": "Ada"}, receipt_id="rcpt-1", status_code=200),
    }
    options.update(kw)
    return ReplayFixture(**options)


def _client(*fixtures: ReplayFixture, **kw: Any) -> Client:
    return Client(vault_replay=VaultReplay(fixtures), **kw)


def test_a_fixture_answers_offline_and_is_marked_replayed() -> None:
    client = _client(_fixture())
    out = client.tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
    )
    assert (out.result, out.receipt_id, out.status_code, out.replayed) == (
        {"name": "Ada"},
        "rcpt-1",
        200,
        True,
    )
    assert out.action_id == ACTION
    assert client.vault_replay is not None
    assert [(c.fixture_id, c.repeated) for c in client.vault_replay.calls] == [("fx-1", False)]


def test_a_missing_fixture_is_an_explicit_error_and_never_a_live_call(
    httpx_mock: HTTPXMock,
) -> None:
    client = _client(_fixture(), base_url=BASE, runtime_token="vrt_live_token_0001")
    with pytest.raises(ReplayFixtureMissing) as excinfo:
        client.tools.execute(
            tool="crm.get_customer", binding="crm-read", arguments={"id": "c_unknown"}
        )
    error = excinfo.value
    assert error.code == "mock_unmatched"
    assert (error.tool, error.binding) == ("crm.get_customer", "crm-read")
    assert error.arguments_hash.startswith("blake3:")
    assert isinstance(error, ToolExecutionError)
    assert "never falls back to a live call" in str(error)
    assert httpx_mock.get_requests() == []


async def test_a_missing_fixture_is_an_explicit_error_on_the_async_path_too(
    httpx_mock: HTTPXMock,
) -> None:
    client = _client(base_url=BASE, runtime_token="vrt_live_token_0001")
    with pytest.raises(ReplayFixtureMissing):
        await client.tools.aexecute(tool="t", binding="b")
    assert httpx_mock.get_requests() == []


async def test_async_replay_serves_the_fixture(httpx_mock: HTTPXMock) -> None:
    out = await _client(_fixture(), base_url=BASE).tools.aexecute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}
    )
    assert out.result == {"name": "Ada"}
    assert httpx_mock.get_requests() == []


def test_the_match_is_exact_on_tool_binding_and_canonical_arguments() -> None:
    client = _client(_fixture(arguments={"b": 1, "a": {"y": 2, "x": 1}}))
    out = client.tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"a": {"x": 1, "y": 2}, "b": 1}
    )
    assert out.replayed
    for tool, binding, arguments in [
        ("other.tool", "crm-read", {"b": 1, "a": {"y": 2, "x": 1}}),
        ("crm.get_customer", "other-binding", {"b": 1, "a": {"y": 2, "x": 1}}),
        ("crm.get_customer", "crm-read", {"b": 2, "a": {"y": 2, "x": 1}}),
        ("crm.get_customer", "crm-read", {}),
    ]:
        with pytest.raises(ReplayFixtureMissing):
            client.tools.execute(tool=tool, binding=binding, arguments=arguments)


def test_the_missing_fixture_message_never_echoes_the_arguments() -> None:
    client = _client()
    with pytest.raises(ReplayFixtureMissing) as excinfo:
        client.tools.execute(tool="t", binding="b", arguments={"email": CANARY})
    assert CANARY not in str(excinfo.value) + repr(excinfo.value)


def test_the_same_action_id_replays_the_same_outcome_and_a_different_request_conflicts() -> None:
    client = _client(_fixture(), _fixture(fixture_id="fx-2", arguments={"id": "c_2"}))
    kwargs: dict[str, Any] = {
        "tool": "crm.get_customer",
        "binding": "crm-read",
        "action_id": ACTION,
    }
    first = client.tools.execute(arguments={"id": "c_1"}, **kwargs)
    again = client.tools.execute(arguments={"id": "c_1"}, **kwargs)
    assert first == again
    assert client.vault_replay is not None
    assert [c.repeated for c in client.vault_replay.calls] == [False, True]
    with pytest.raises(VaultConflict):
        client.tools.execute(arguments={"id": "c_2"}, **kwargs)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            ReplayOutcome(kind="failed", status_code=404, error_class="destination_error"),
            VaultExecutionFailed,
        ),
        (ReplayOutcome(kind="outcome_unknown", status_code=504), VaultOutcomeUnknown),
        (
            ReplayOutcome(
                kind="denied", reason_codes=["no_policy_bound"], explanation="none bound"
            ),
            VaultPolicyDenied,
        ),
        (ReplayOutcome(kind="approval_required", approval_id="apr-5"), VaultApprovalRequired),
    ],
)
def test_every_outcome_raises_the_same_error_class_as_a_live_execution(
    outcome: ReplayOutcome, expected: type[Exception]
) -> None:
    client = _client(_fixture(outcome))
    with pytest.raises(expected) as excinfo:
        client.tools.execute(
            tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
        )
    assert excinfo.value.action_id == ACTION  # type: ignore[attr-defined]


def test_replayed_errors_equal_live_errors(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/v1/vault/runtime/executions",
        json={
            "status": "finished",
            "action_id": ACTION,
            "state": "outcome_unknown",
            "status_code": 504,
        },
    )
    live = Client(base_url=BASE, runtime_token="vrt_t")
    replayed = _client(_fixture(ReplayOutcome(kind="outcome_unknown", status_code=504)))
    caught: list[VaultOutcomeUnknown] = []
    for client in (live, replayed):
        with pytest.raises(VaultOutcomeUnknown) as excinfo:
            client.tools.execute(
                tool="crm.get_customer",
                binding="crm-read",
                arguments={"id": "c_1"},
                action_id=ACTION,
            )
        caught.append(excinfo.value)
    assert (caught[0].code, caught[0].status, caught[0].status_code, caught[0].action_id) == (
        caught[1].code,
        caught[1].status,
        caught[1].status_code,
        caught[1].action_id,
    )


def test_an_approval_fixture_raises_every_time_until_it_is_replaced() -> None:
    client = _client(_fixture(ReplayOutcome(kind="approval_required")))
    for _ in range(2):
        with pytest.raises(VaultApprovalRequired) as excinfo:
            client.tools.execute(
                tool="crm.get_customer",
                binding="crm-read",
                arguments={"id": "c_1"},
                action_id=ACTION,
            )
        assert excinfo.value.approval_id == "replay-approval-fx-1"
    assert client.vault_replay is not None
    client.vault_replay.add(_fixture(), replace=True)
    out = client.tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
    )
    assert out.replayed


def test_get_execution_reads_the_stored_outcome_and_an_unknown_action_is_not_found() -> None:
    client = _client(_fixture(ReplayOutcome(kind="outcome_unknown", status_code=504)))
    with pytest.raises(VaultOutcomeUnknown):
        client.tools.execute(
            tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
        )
    status = client.tools.get_execution(ACTION)
    assert (status.state, status.replayed, status.status_code) == ("outcome_unknown", True, 504)
    with pytest.raises(VaultNotFound):
        client.tools.get_execution(OTHER)


async def test_async_get_execution_in_replay() -> None:
    client = _client(_fixture())
    await client.tools.aexecute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}, action_id=ACTION
    )
    assert (await client.tools.aget_execution(ACTION)).state == "succeeded"


def test_grants_have_no_replay_counterpart_and_are_not_sent_live(httpx_mock: HTTPXMock) -> None:
    runtime = _client(base_url=BASE, runtime_token="vrt_t").vault.runtime
    with pytest.raises(ReplayUnsupported):
        runtime.request_grant(binding_id="b", max_uses=1, ttl_seconds=60, reason="r")
    with pytest.raises(ReplayUnsupported):
        runtime.list_grants()
    with pytest.raises(ReplayUnsupported):
        runtime.delegate_grant("g", delegate_agent_id="a", max_uses=1, ttl_seconds=1, reason="r")
    assert httpx_mock.get_requests() == []


def test_fixtures_round_trip_through_a_file(tmp_path: Path) -> None:
    replay = VaultReplay(
        [
            _fixture(),
            _fixture(
                ReplayOutcome(kind="outcome_unknown"), fixture_id="fx-2", arguments={"id": "c_2"}
            ),
        ]
    )
    path = tmp_path / "fixtures.json"
    replay.save(path)
    document = json.loads(path.read_text())
    assert document["schema_version"] == "agenomic.vault_replay/v1"
    assert [f["fixture_id"] for f in document["fixtures"]] == ["fx-1", "fx-2"]
    loaded = VaultReplay.from_file(path)
    out = Client(vault_replay=loaded).tools.execute(
        tool="crm.get_customer", binding="crm-read", arguments={"id": "c_1"}
    )
    assert out.result == {"name": "Ada"}


@pytest.mark.parametrize(
    "document",
    [
        {"fixtures": []},
        {"schema_version": "agenomic.vault_replay/v1", "fixtures": "nope"},
        {"schema_version": "agenomic.vault_replay/v1", "fixtures": [{"fixture_id": "x"}]},
        {
            "schema_version": "agenomic.vault_replay/v1",
            "fixtures": [{"fixture_id": "x", "tool": "t", "binding": "b", "typo": 1}],
        },
    ],
)
def test_a_malformed_fixture_document_is_refused(document: dict[str, Any]) -> None:
    with pytest.raises(VaultValidationError) as excinfo:
        VaultReplay.from_mapping(document)
    assert excinfo.value.code == "invalid_fixtures"


def test_a_missing_or_corrupt_fixture_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VaultValidationError):
        VaultReplay.from_file(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2")
    with pytest.raises(VaultValidationError):
        VaultReplay.from_file(bad)


def test_a_duplicate_fixture_is_refused_unless_replaced() -> None:
    replay = VaultReplay([_fixture()])
    with pytest.raises(VaultValidationError) as excinfo:
        replay.add(_fixture(fixture_id="fx-dup"))
    assert excinfo.value.code == "duplicate_fixture"
    replay.add(_fixture(fixture_id="fx-dup"), replace=True)


def test_without_a_replay_set_or_a_base_url_nothing_is_silently_mocked() -> None:
    from agenomic.vault import VaultNotConfigured

    with pytest.raises(VaultNotConfigured) as excinfo:
        Client(runtime_token="vrt_t").tools.execute(tool="t", binding="b")
    assert excinfo.value.code == "cloud_required"
