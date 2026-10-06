from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agenomic.integrations.hermes import exporter as exporter_mod
from agenomic.integrations.hermes.exporter import (
    EVENT_SCHEMA,
    EventBuilder,
    EventExporter,
    is_secret_key,
    mask_text,
    redacted_preview,
)

#: The documented ``POST /v1/hermes/runtime/events`` body limit (cloud docs/hermes/api.md).
BODY_LIMIT = 1024 * 1024

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")

SECRETS = ("sk-livesecretvalue123", "agmhr_runtimesecret", "hunter2-password")


def raw_content() -> dict[str, Any]:
    return {
        "input": {
            "command": f"curl -H 'Authorization: Bearer {SECRETS[1]}' https://x",
            "api_key": SECRETS[0],
            "nested": {"password": SECRETS[2]},
        },
        "output": f"done with {SECRETS[0]}",
    }


def test_builder_shape_and_metadata_default() -> None:
    builder = EventBuilder()
    e1 = builder.build(
        "tool.call.requested", hermes_session_id="s", tool={"name": "t"}, content=raw_content()
    )
    e2 = builder.build("tool.call.completed", hermes_session_id="s")
    assert e1["schema_version"] == EVENT_SCHEMA
    assert len(e1["event_id"]) == 26
    assert e1["event_id"] != e2["event_id"]
    assert (e1["seq"], e2["seq"]) == (1, 2)
    assert e1["occurred_at"].endswith("Z")
    assert set(e1["extra"]["content_hashes"]) == {"input", "output"}
    assert "previews" not in e1["extra"]
    blob = json.dumps(e1)
    assert not any(s in blob for s in SECRETS)
    with pytest.raises(ValueError, match="unknown event field"):
        builder.build("x", prompt="raw")


def test_builder_redacted_preview_masks_and_truncates() -> None:
    builder = EventBuilder("redacted_preview", preview_chars=80)
    event = builder.build("tool.call.requested", content=raw_content())
    blob = json.dumps(event)
    assert not any(s in blob for s in SECRETS)
    assert all(len(p) <= 80 for p in event["extra"]["previews"].values())
    assert mask_text("token agmhs_abc and ghp_" + "a" * 20) == "token *** and ***"
    assert redacted_preview(42, 10) == "42"


def collect() -> tuple[list[list[dict[str, Any]]], Any]:
    batches: list[list[dict[str, Any]]] = []

    def post(batch: list[dict[str, Any]]) -> dict[str, int]:
        batches.append(list(batch))
        return {"accepted": len(batch)}

    return batches, post


def test_batching_and_dedup() -> None:
    batches, post = collect()
    builder = EventBuilder()
    exporter = EventExporter(post, batch_size=3, flush_interval_s=0.05)
    events = [builder.build("x") for _ in range(7)]
    for e in events:
        assert exporter.submit(e)
    assert exporter.submit(events[0])  # duplicate accepted silently, not resent
    assert exporter.flush(3.0)
    sent = [e["event_id"] for b in batches for e in b]
    assert sorted(sent) == sorted(e["event_id"] for e in events)
    assert all(len(b) <= 3 for b in batches)
    assert exporter.stats() == {
        "buffered": 0,
        "dropped": 0,
        "buffer_full": False,
        "last_flush_error": None,
    }
    assert exporter.delivered == 7
    assert exporter.close(1.0)
    assert not exporter.submit(EventBuilder().build("late"))
    assert exporter.stats()["dropped"] == 1


def test_backpressure_drops_and_counts_without_blocking() -> None:
    gate = threading.Event()
    sending = threading.Event()

    def stalled(batch: list[dict[str, Any]]) -> None:
        sending.set()
        gate.wait(5)

    builder = EventBuilder()
    exporter = EventExporter(
        stalled, max_events=2, batch_size=1, flush_interval_s=0.01, max_retries=0
    )
    # Park the worker inside a send first, so it cannot drain the buffer (and clear
    # buffer_full) while the agent loop floods it below.
    assert exporter.submit(builder.build("first"))
    assert sending.wait(2)
    started = time.monotonic()
    results = [exporter.submit(builder.build("x")) for _ in range(10)]
    assert time.monotonic() - started < 1.0  # never blocks the agent loop
    stats = exporter.stats()
    assert results.count(False) >= 7
    assert stats["dropped"] == results.count(False)
    assert stats["buffer_full"] is True
    gate.set()
    exporter.close(2.0)


def test_retry_limit_then_drop() -> None:
    attempts: list[int] = []

    def failing(batch: list[dict[str, Any]]) -> None:
        attempts.append(len(batch))
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=2, backoff_s=0.01, flush_interval_s=0.01)
    exporter.submit(EventBuilder().build("x"))
    assert not exporter.flush(3.0)
    assert len(attempts) == 3
    stats = exporter.stats()
    assert stats["dropped"] == 1
    assert stats["last_flush_error"] == "ConnectionError"
    exporter.close(1.0)


def test_spool_is_bounded_redacted_and_replayed(tmp_path: Path) -> None:
    spool = tmp_path / "spool" / "events.jsonl"
    up = threading.Event()
    delivered: list[dict[str, Any]] = []

    def post(batch: list[dict[str, Any]]) -> None:
        if not up.is_set():
            raise ConnectionError("down")
        delivered.extend(batch)

    builder = EventBuilder("redacted_preview")
    exporter = EventExporter(
        post, max_retries=0, flush_interval_s=0.02, spool_path=str(spool), spool_max_bytes=4096
    )
    for _ in range(40):
        exporter.submit(builder.build("tool.call.requested", content=raw_content()))
    exporter.flush(3.0)
    assert spool.exists()
    text = spool.read_text()
    assert 0 < spool.stat().st_size <= 4096
    assert not any(s in text for s in SECRETS)
    if os.name == "posix":  # Windows has no POSIX mode bits
        assert oct(spool.stat().st_mode & 0o777) == "0o600"
    stats = exporter.stats()
    spooled = len(text.splitlines())
    assert stats["dropped"] == 40 - spooled  # what did not fit is counted, never silent
    up.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(delivered) < spooled:
        exporter.submit(builder.build("nudge"))
        time.sleep(0.05)
    assert len({e["event_id"] for e in delivered if e["type"] == "tool.call.requested"}) == spooled
    exporter.close(1.0)


def test_close_spools_leftovers(tmp_path: Path) -> None:
    spool = tmp_path / "s.jsonl"

    def failing(batch: list[dict[str, Any]]) -> None:
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=0, flush_interval_s=10, spool_path=str(spool))
    exporter.submit(EventBuilder().build("x"))
    exporter.close(0.2)
    assert len(spool.read_text().splitlines()) == 1


def test_oversized_and_invalid_events() -> None:
    batches, post = collect()
    exporter = EventExporter(post, flush_interval_s=0.01)
    big = EventBuilder().build("x", extra={"blob": "a" * 100_000})
    assert exporter.submit(big)
    assert not exporter.submit({"type": "no-id"})
    exporter.flush(2.0)
    assert batches[0][0]["extra"] == {"truncated": True}
    assert exporter.stats()["dropped"] == 1
    exporter.close(1.0)


def test_preview_masks_credential_keys_whatever_their_case() -> None:
    text = redacted_preview(
        {
            "API_KEY": "plainsecret",
            "Authorization": "Basic plainsecret",
            "nested": [{"X-Api-Key": "v", "DB_Password": "p", "Auth_Token": "t"}],
            "max_tokens": 5,
        },
        500,
    )
    assert "plainsecret" not in text
    for value in ('"v"', '"p"', '"t"'):
        assert value not in text
    assert '"max_tokens":5' in text


@pytest.mark.parametrize(
    "key",
    [
        "auth",
        "AUTH",
        "Auth",
        "pass",
        "PWD",
        "passphrase",
        "ssh_passphrase",
        "Bearer",
        "jwt",
        "JWT",
        "otp",
        "totp",
        "csrf",
        "X-CSRF-Token",
        "xsrf",
        "csrf_token",
        "X-XSRF-TOKEN",
    ],
)
def test_exact_credential_key_aliases_are_masked(key: str) -> None:
    assert is_secret_key(key)
    text = redacted_preview({key: "plainsecret"}, 500)
    assert "plainsecret" not in text


@pytest.mark.parametrize(
    "key",
    [
        "author",
        "authority",
        "authenticated",
        "auth_method",
        "oauth_provider",
        "bypass",
        "passed",
        "pass_count",
        "session",
        "session_id",
        "sessionId",
        "tokens",
        "max_tokens",
        "input_tokens",
        "jwt_issuer",
        "otp_length",
        "path",
        "output",
    ],
)
def test_ordinary_keys_near_credential_aliases_are_kept(key: str) -> None:
    assert not is_secret_key(key)
    assert "plainvalue" in redacted_preview({key: "plainvalue"}, 500)


def test_password_named_policy_keys_stay_masked() -> None:
    # Any key containing ``password`` is masked, ``password_policy`` included: masking a
    # non-secret is a safe failure, a substring exemption could leak ``password_old``.
    assert is_secret_key("password_policy")


@pytest.mark.parametrize(
    "text",
    [
        "auth=plainsecret",
        "?user=bob&auth=plainsecret",
        "pwd: plainsecret",
        "PASS='plainsecret'",
        '"jwt": "plainsecret"',
        "otp=plainsecret",
        "passphrase=plainsecret",
    ],
)
def test_exact_credential_key_aliases_are_masked_in_text(text: str) -> None:
    assert "plainsecret" not in mask_text(text)


@pytest.mark.parametrize(
    "text",
    ["author=plainvalue", "bypass=plainvalue", "passed=plainvalue", "auth_method=plainvalue"],
)
def test_ordinary_keys_near_credential_aliases_are_kept_in_text(text: str) -> None:
    assert mask_text(text) == text


def test_free_form_fields_are_redacted_in_metadata_mode() -> None:
    event = EventBuilder("metadata").build(
        "api.request.failed",
        reason="Authorization: Bearer sk-abcdefghijklmnop",
        extra={"detail": {"api_key": "plain", "msg": "token agmhr_abc123"}},
    )
    text = json.dumps(event)
    for secret in ("sk-abcdefghijklmnop", "agmhr_abc123", '"plain"'):
        assert secret not in text


def test_every_json_container_is_redacted() -> None:
    from types import MappingProxyType

    event = EventBuilder("redacted_preview").build(
        "tool.call.requested",
        extra={
            "headers": ({"Authorization": "plainsecret"},),
            "view": MappingProxyType({"api_key": "plainsecret"}),
            "tags": frozenset({"sk-abcdefghijklmnop"}),
        },
        content={"input": ({"api_key": "plainsecret"},)},
    )
    assert event["extra"]["headers"] == [{"Authorization": "***"}]
    assert event["extra"]["view"] == {"api_key": "***"}
    assert event["extra"]["tags"] == ["***"]
    text = json.dumps(event, default=str)
    for secret in ("plainsecret", "sk-abcdefghijklmnop"):
        assert secret not in text


def test_top_level_credential_keys_are_masked() -> None:
    event = EventBuilder("redacted_preview").build(
        "tool.call.requested",
        extra={"Authorization": "plainsecret", "X-Api-Key": 42, "status_code": 200},
        content={"api_key": "plainsecret", "input": {"q": "x"}},
        usage={"input_tokens": 3, "auth_token": "plainsecret"},
    )
    assert event["extra"]["Authorization"] == "***"
    assert event["extra"]["X-Api-Key"] == "***"
    assert event["extra"]["status_code"] == 200
    assert event["extra"]["previews"]["api_key"] == "***"
    assert event["extra"]["previews"]["input"] == '{"q":"x"}'
    assert event["usage"] == {"input_tokens": 3, "auth_token": "***"}
    assert "plainsecret" not in json.dumps(event, default=str)


def test_spool_is_replayed_while_live_traffic_continues(tmp_path: Path) -> None:
    builder = EventBuilder()
    spool = tmp_path / "events.jsonl"
    spooled = [builder.build("spooled") for _ in range(3)]
    spool.write_text("".join(json.dumps(e) + "\n" for e in spooled))
    feed_limit = 200
    live = {"batches": 0}
    replayed_after: list[int] = []
    done = threading.Event()
    holder: dict[str, EventExporter] = {}

    def post(batch: list[dict[str, Any]]) -> None:
        if any(e["type"] == "spooled" for e in batch):
            replayed_after.append(live["batches"])
            done.set()
            return
        live["batches"] += 1
        if live["batches"] < feed_limit:
            # Feed the next live event before returning: the buffer is never empty.
            holder["exporter"].submit(builder.build("live"))
        else:
            done.set()

    # A long flush interval: only the live batch count can trigger the replay.
    exporter = EventExporter(post, flush_interval_s=30.0, batch_size=1, spool_path=str(spool))
    holder["exporter"] = exporter
    exporter._last_replay_at = time.monotonic()
    exporter.submit(builder.build("live"))
    assert done.wait(10.0)
    assert replayed_after, "the spool is replayed"
    assert replayed_after[0] < feed_limit, "while live traffic was still flowing"
    exporter.close(2.0)


@pytest.mark.parametrize(
    ("text", "kept", "secret"),
    [
        ("Authorization: Basic cGxhaW5zZWNyZXQ=", "Authorization: Basic ***", "cGxhaW5zZWNyZXQ"),
        ("Authorization: Bearer abc.def-ghi", "Authorization: Bearer ***", "abc.def"),
        ("authorization: Token tok123secret", "authorization: Token ***", "tok123secret"),
        ("Authorization: ApiKey key123secret", "Authorization: ApiKey ***", "key123secret"),
        (
            'Authorization: Digest username="bob", response="6629fae49393a0539"',
            "Authorization: Digest ***",
            "6629fae49393a0539",
        ),
        (
            '{"authorization": "Digest username=\\"bob\\", response=\\"6629fae4\\""}',
            '{"authorization": "Digest ***"}',
            "6629fae4",
        ),
        (
            "curl -H 'Proxy-Authorization: Basic dXNlcjpwYXNz' https://x",
            "curl -H 'Proxy-Authorization: Basic ***' https://x",
            "dXNlcjpwYXNz",
        ),
        (
            "Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/s3, Signature=fe5f80f7",
            "Authorization: AWS4-HMAC-SHA256 ***",
            "fe5f80f7",
        ),
        (
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/s3, SignedHeaders=host, Signature=fe5f80f7",
            "AWS4-HMAC-SHA256 Credential=***, SignedHeaders=host, Signature=***",
            "fe5f80f7",
        ),
        (
            "curl -H 'x-api-key: value12345' https://x",
            "curl -H 'x-api-key: ***' https://x",
            "value12345",
        ),
        ("GET /v1?api_key=plainkey&page=2", "GET /v1?api_key=***&page=2", "plainkey"),
        ("login password=hunter2 ok", "login password=*** ok", "hunter2"),
        ("token: t0k3nvalue", "token: ***", "t0k3nvalue"),
        ('{"client_secret": "s3cr3t value"}', '{"client_secret": "***"}', "s3cr3t"),
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI", "AWS_SECRET_ACCESS_KEY=***", "wJalrXUtnFEMI"),
        ("Cookie: session=abc123; theme=dark", "Cookie: ***", "abc123"),
        (
            "postgres://admin:pa55word@db.example:5432/app",
            "postgres://admin:***@db.example:5432/app",
            "pa55word",
        ),
    ],
)
def test_credential_schemes_are_masked_but_stay_readable(text: str, kept: str, secret: str) -> None:
    assert mask_text(text) == kept
    event = EventBuilder("metadata").build("api.request.failed", reason=text)
    assert secret not in json.dumps(event)


@pytest.mark.parametrize(
    "prose",
    [
        "the token budget was exceeded",
        "max_tokens=512",
        "input_tokens: 3, output_tokens: 7",
        "authorization required for this tool",
        "a secret is not shared here",
        "ssh://git@github.com/org/repo.git",
        "if token == expected: ok",
    ],
)
def test_ordinary_prose_is_not_masked(prose: str) -> None:
    assert mask_text(prose) == prose


class _Opaque:
    def __str__(self) -> str:
        return "client with x-api-key: plainsecret"


def test_object_leaves_are_masked_before_serialization(tmp_path: Path) -> None:
    error = RuntimeError("upstream said Authorization: Basic cGxhaW5zZWNyZXQ=")
    event = EventBuilder("metadata").build(
        "api.request.failed",
        reason=error,
        extra={"error": error, "client": _Opaque(), "nested": [error], "latency": float("nan")},
        usage={"ratio": float("inf"), "input_tokens": 3, "ok": True},
    )
    assert event["reason"] == "upstream said Authorization: Basic ***"
    assert event["extra"]["client"] == "client with x-api-key: ***"
    assert event["extra"]["nested"] == ["upstream said Authorization: Basic ***"]
    assert event["extra"]["latency"] is None
    assert event["usage"] == {"ratio": None, "input_tokens": 3, "ok": True}
    # Strict JSON (no NaN) and no credential, in the exporter's own serialization too.
    text = json.dumps(event, allow_nan=False)
    assert "cGxhaW5zZWNyZXQ=" not in text
    assert "plainsecret" not in text
    spool = tmp_path / "spool.jsonl"

    def failing(batch: list[dict[str, Any]]) -> None:
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=0, flush_interval_s=10, spool_path=str(spool))
    exporter.submit(event)
    exporter.close(0.2)
    spooled = spool.read_text()
    assert "Basic ***" in spooled
    assert "cGxhaW5zZWNyZXQ=" not in spooled
    assert "plainsecret" not in spooled


@posix_only
def test_existing_spool_is_narrowed_to_owner_only(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text("")
    os.chmod(spool, 0o644)

    def failing(batch: list[dict[str, Any]]) -> None:
        raise ConnectionError("down")

    exporter = EventExporter(failing, max_retries=0, flush_interval_s=10, spool_path=str(spool))
    exporter.submit(EventBuilder().build("x"))
    exporter.close(0.2)
    assert len(spool.read_text().splitlines()) == 1
    assert oct(spool.stat().st_mode & 0o777) == "0o600"


@posix_only
def test_spool_directory_it_creates_is_owner_only(tmp_path: Path) -> None:
    spool = tmp_path / "fresh" / "spool.jsonl"
    exporter = EventExporter(lambda batch: None, spool_path=str(spool))
    exporter.close(0.2)
    assert oct(spool.parent.stat().st_mode & 0o777) == "0o700"


@posix_only
def test_spool_rewrite_never_reuses_a_wide_temporary_file(tmp_path: Path) -> None:
    from agenomic.integrations.hermes.exporter import _Spool

    spool = _Spool(tmp_path / "spool.jsonl", 1 << 20)
    spool.append([EventBuilder().build("a"), EventBuilder().build("b")])
    leftover = tmp_path / "spool.jsonl.tmp"
    leftover.write_text("stale")
    os.chmod(leftover, 0o666)
    lines, events = spool.head(1)
    assert len(events) == 1
    spool.remove(lines)
    assert len(spool.path.read_text().splitlines()) == 1
    assert oct(spool.path.stat().st_mode & 0o777) == "0o600"


def _spooled_exporter(spool: Path, post: Any) -> EventExporter:
    """An exporter whose worker stays parked, so the test drives ``_replay_spool`` itself."""
    return EventExporter(post, max_retries=0, flush_interval_s=30.0, spool_path=str(spool))


@posix_only
def test_replay_refuses_a_symlinked_spool(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.jsonl"
    content = json.dumps(EventBuilder().build("planted")) + "\n"
    target.write_text(content)
    os.chmod(target, 0o644)
    spool = tmp_path / "spool.jsonl"
    spool.symlink_to(target)
    sent: list[dict[str, Any]] = []
    exporter = _spooled_exporter(spool, sent.extend)
    exporter._replay_spool()
    assert sent == []
    assert spool.is_symlink(), "the link is not replaced"
    assert target.read_text() == content, "the target is neither rewritten nor truncated"
    assert oct(target.stat().st_mode & 0o777) == "0o644"
    # Spooling to that path stops: an undelivered event is counted, not written.
    exporter._overflow([EventBuilder().build("later")], "test")
    assert target.read_text() == content
    assert exporter.stats()["dropped"] == 1
    exporter.close(0.2)


@posix_only
def test_replay_refuses_a_spool_owned_by_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool = tmp_path / "spool.jsonl"
    content = json.dumps(EventBuilder().build("planted")) + "\n"
    spool.write_text(content)
    foreign = spool.stat().st_ino
    real_fstat = os.fstat

    def fstat(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        if st.st_ino != foreign:
            return st
        fields = list(st[:10])
        fields[4] = os.geteuid() + 1  # st_uid
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", fstat)
    sent: list[dict[str, Any]] = []
    exporter = _spooled_exporter(spool, sent.extend)
    exporter._replay_spool()
    assert sent == []
    assert spool.read_text() == content
    assert spool.stat().st_ino == foreign, "the foreign file is not replaced"
    exporter.close(0.2)


def test_replayed_records_are_validated_and_redacted_again(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    tampered = EventBuilder().build("tampered")
    tampered["extra"] = {"note": "Authorization: Basic cGxhaW5zZWNyZXQ=", "api_key": "k-123"}
    tampered["reason"] = "curl -H 'Authorization: Bearer abcdefghijklmnop'"
    lines = [
        json.dumps(tampered),
        json.dumps({"event_id": "x", "type": "t"}),  # no schema_version
        json.dumps({**EventBuilder().build("ok"), "unexpected": 1}),  # key outside the schema
        json.dumps(["not", "an", "event"]),
        "{not json",
    ]
    spool.write_text("".join(line + "\n" for line in lines))
    sent: list[dict[str, Any]] = []
    exporter = _spooled_exporter(spool, sent.extend)
    exporter._replay_spool()
    assert [e["event_id"] for e in sent] == [tampered["event_id"]]
    text = json.dumps(sent)
    assert "cGxhaW5zZWNyZXQ=" not in text
    assert "abcdefghijklmnop" not in text
    assert "k-123" not in text
    assert sent[0]["extra"]["note"] == "Authorization: Basic ***"
    assert exporter.stats()["dropped"] == 4
    assert spool.read_text() == "", "the batch, invalid lines included, left the spool"
    exporter.close(0.2)


def test_replayed_batch_stays_on_disk_until_acknowledged(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    events = [EventBuilder().build("spooled") for _ in range(3)]
    original = "".join(json.dumps(e) + "\n" for e in events)
    spool.write_text(original)
    on_disk_during_post: list[str] = []

    def failing(batch: list[dict[str, Any]]) -> None:
        on_disk_during_post.append(spool.read_text())
        raise ConnectionError("process dies before the acknowledgement")

    exporter = _spooled_exporter(spool, failing)
    exporter._replay_spool()
    assert on_disk_during_post == [original], "a crash during delivery loses nothing"
    assert spool.read_text() == original, "a failed replay neither loses nor duplicates"
    assert exporter.stats()["dropped"] == 0
    exporter.close(0.2)


def test_acknowledged_replay_is_removed_and_concurrent_appends_survive(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    events = [EventBuilder().build("spooled") for _ in range(3)]
    spool.write_text("".join(json.dumps(e) + "\n" for e in events))
    late = EventBuilder().build("late")
    holder: dict[str, EventExporter] = {}
    sent: list[dict[str, Any]] = []

    def post(batch: list[dict[str, Any]]) -> None:
        # Another producer overflows into the spool while the replay is in flight.
        holder["exporter"]._overflow([late], "buffer full")
        sent.extend(batch)

    exporter = _spooled_exporter(spool, post)
    holder["exporter"] = exporter
    exporter._replay_spool()
    assert [e["event_id"] for e in sent] == [e["event_id"] for e in events]
    remaining = [json.loads(line) for line in spool.read_text().splitlines()]
    assert [e["event_id"] for e in remaining] == [late["event_id"]]
    exporter.close(0.2)


@posix_only
def test_spool_rewrite_ignores_a_link_planted_at_the_old_temporary_name(tmp_path: Path) -> None:
    from agenomic.integrations.hermes.exporter import _Spool

    spool = _Spool(tmp_path / "spool.jsonl", 1 << 20)
    spool.append([EventBuilder().build("a"), EventBuilder().build("b")])
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    (tmp_path / "spool.jsonl.tmp").symlink_to(victim)
    lines, _ = spool.head(1)
    spool.remove(lines)
    assert not spool.refused
    assert victim.read_text() == "precious"
    assert len(spool.path.read_text().splitlines()) == 1
    assert oct(spool.path.stat().st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------- worker resilience


def test_spool_stat_error_keeps_the_worker_delivering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(self: Any) -> int:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(exporter_mod._Spool, "size", unreadable)
    batches, post = collect()
    exporter = EventExporter(post, flush_interval_s=0.01, spool_path=str(tmp_path / "s.jsonl"))
    for _ in range(3):
        assert exporter.submit(EventBuilder().build("live"))
        assert exporter.flush(5.0), "live events are still delivered"
    assert sum(len(b) for b in batches) == 3
    assert exporter._thread.is_alive()
    assert exporter._spool_errors >= 1
    exporter.close(1.0)


def test_unexpected_worker_error_is_logged_and_the_worker_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: Any) -> None:
        raise RuntimeError("bug in a replay step")

    monkeypatch.setattr(EventExporter, "_replay_spool", broken)
    batches, post = collect()
    exporter = EventExporter(post, flush_interval_s=0.01)
    with caplog.at_level("ERROR", logger="agenomic.integrations.hermes.exporter"):
        for _ in range(3):
            assert exporter.submit(EventBuilder().build("live"))
            assert exporter.flush(5.0)
    assert sum(len(b) for b in batches) == 3
    assert exporter._thread.is_alive()
    assert any("worker continues" in r.getMessage() for r in caplog.records)
    exporter.close(1.0)


# ---------------------------------------------------------------- request size


def _within_body_limit(batch: list[dict[str, Any]]) -> bool:
    """Whatever the transport's JSON encoder (compact UTF-8, or ASCII escapes)."""
    body = {"events": batch}
    compact = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    escaped = json.dumps(body).encode("utf-8")
    return max(len(compact), len(escaped)) <= BODY_LIMIT


def _large_event(builder: EventBuilder, i: int) -> dict[str, Any]:
    # Near the 64 KiB per event limit; half of them non-ASCII (6 bytes once escaped).
    blob = ("é" * 25_000) if i % 2 else ("a" * 60_000)
    return builder.build("tool.call.completed", extra={"blob": blob})


def test_live_batches_stay_under_the_request_body_limit() -> None:
    batches, post = collect()
    # A long interval: every event is buffered before the first batch is taken.
    exporter = EventExporter(post, flush_interval_s=30.0)
    builder = EventBuilder()
    events = [_large_event(builder, i) for i in range(60)]
    for event in events:
        assert exporter.submit(event)
    assert exporter.flush(10.0)
    assert len(batches) > 1
    assert all(_within_body_limit(b) for b in batches)
    delivered = [e["event_id"] for b in batches for e in b]
    assert delivered == [e["event_id"] for e in events], "all delivered, in order"
    assert all(b[0]["extra"] != {"truncated": True} for b in batches)
    exporter.close(1.0)


def test_replayed_spool_batches_stay_under_the_request_body_limit(tmp_path: Path) -> None:
    builder = EventBuilder()
    spooled = [_large_event(builder, i) for i in range(40)]
    oversized = builder.build("x", extra={"blob": "b" * 100_000})
    hopeless = builder.build("x", reason="r" * 100_000)  # too large even without extra
    spool = tmp_path / "events.jsonl"
    spool.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in [*spooled, oversized, hopeless]),
        encoding="utf-8",
    )
    batches: list[list[dict[str, Any]]] = []
    done = threading.Event()

    def post(batch: list[dict[str, Any]]) -> None:
        batches.append(list(batch))
        if sum(len(b) for b in batches) >= len(spooled) + 1:
            done.set()

    exporter = EventExporter(post, flush_interval_s=0.01, spool_path=str(spool))
    assert done.wait(20.0)
    exporter.close(2.0)
    assert len(batches) > 1
    assert all(_within_body_limit(b) for b in batches)
    by_id = {e["event_id"]: e for b in batches for e in b}
    assert all(e["event_id"] in by_id for e in spooled)
    assert by_id[oversized["event_id"]]["extra"] == {"truncated": True}
    assert hopeless["event_id"] not in by_id
    assert exporter.stats()["dropped"] == 1, "the record the gateway would refuse is counted"
