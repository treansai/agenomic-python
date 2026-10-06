from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from agenomic.integrations.hermes import guard

PAYLOAD = json.dumps(
    {
        "hook_event_name": "pre_tool_call",
        "tool_name": "write_file",
        "tool_input": {"path": "/tmp/x"},
        "session_id": "s",
        "cwd": "/",
        "extra": {},
    }
)


def iso(delta_s: float = 0.0) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat().replace("+00:00", "Z")
    )


EPOCH = "epoch-of-this-hermes-process"


def run(
    home: Path, status: Optional[Any] = None, stdin: str = PAYLOAD, **env: str
) -> tuple[int, str]:
    """Run the guard as a hook of the Hermes process whose plugin wrote ``status``: the
    epoch is in the environment and, unless the test sets it, in a dict status."""
    if status is not None:
        if isinstance(status, dict):
            status = {"epoch": EPOCH, **status}
        path = home / "agenomic" / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(status if isinstance(status, str) else json.dumps(status))
    out = io.StringIO()
    environ = {"HERMES_HOME": str(home), guard.GUARD_EPOCH_ENV: EPOCH, **env}
    code = guard.main(stdin=io.StringIO(stdin), stdout=out, environ=environ)
    return code, out.getvalue()


@pytest.mark.parametrize(
    ("status_epoch", "env_epoch"),
    [("other-process", EPOCH), (None, EPOCH), (EPOCH, ""), ("", "")],
    ids=["other_process", "status_without_epoch", "hook_without_epoch", "both_empty"],
)
def test_status_of_another_hermes_process_blocks(
    tmp_path: Path, status_epoch: Optional[str], env_epoch: str
) -> None:
    # A fresh loaded status left by a Hermes process killed with SIGKILL and restarted
    # without the plugin: the new process's hooks do not carry that epoch.
    status: dict[str, Any] = {"loaded": True, "instance_status": "active", "updated_at": iso()}
    status["epoch"] = status_epoch
    code, out = run(tmp_path, status, **{guard.GUARD_EPOCH_ENV: env_epoch})
    assert_block(code, out, "another Hermes process")


def assert_block(code: int, out: str, fragment: str) -> None:
    assert code == 2
    doc = json.loads(out)
    assert doc["action"] == "block"
    assert fragment in doc["message"]


def test_allows_when_loaded_and_fresh(tmp_path: Path) -> None:
    code, out = run(
        tmp_path,
        {
            "loaded": True,
            "instance_status": "active",
            "effective_state": "enforce",
            "updated_at": iso(),
        },
    )
    assert (code, out) == (0, "")


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (None, "plugin is not loaded"),
        ("{not json", "unreadable"),
        ("[1]", "unreadable"),
        ({"loaded": False, "updated_at": iso()}, "not loaded"),
        ({"loaded": True, "instance_status": "active", "updated_at": iso(-600)}, "stale"),
        ({"loaded": True, "instance_status": "active", "updated_at": iso(3600)}, "future"),
        (
            {"loaded": True, "instance_status": "active", "updated_at": "yesterday"},
            "no valid timestamp",
        ),
        ({"loaded": True, "instance_status": "paused", "updated_at": iso()}, "is paused"),
        ({"loaded": True, "instance_status": "revoked", "updated_at": iso()}, "is revoked"),
        (
            {
                "loaded": True,
                "instance_status": "active",
                "effective_state": "quarantined",
                "updated_at": iso(),
            },
            "is quarantined",
        ),
    ],
)
def test_each_failure_blocks_with_json_and_exit_2(
    tmp_path: Path, status: Any, fragment: str
) -> None:
    assert_block(*run(tmp_path, status), fragment)


def test_bad_stdin_blocks(tmp_path: Path) -> None:
    fresh = {"loaded": True, "instance_status": "active", "updated_at": iso()}
    assert_block(*run(tmp_path, fresh, stdin="[]"), "not an object")
    code, out = run(tmp_path, fresh, stdin="{garbage")
    assert_block(code, out, "guard failed")


def test_max_age_env(tmp_path: Path) -> None:
    status = {"loaded": True, "instance_status": "active", "updated_at": iso(-30)}
    assert run(tmp_path, status)[0] == 0
    assert_block(*run(tmp_path, status, AGENOMIC_HERMES_GUARD_MAX_AGE_S="10"), "stale")


def test_internal_error_still_prints_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(guard, "evaluate", boom)
    code, out = run(tmp_path, {"loaded": True, "updated_at": iso()})
    assert_block(code, out, "Agenomic guard failed")


def test_console_script_process(tmp_path: Path) -> None:
    # The real process path: missing status -> block JSON on stdout, exit 2.
    proc = subprocess.run(
        [sys.executable, "-c", "from agenomic.integrations.hermes.guard import cli; cli()"],
        input=PAYLOAD,
        capture_output=True,
        text=True,
        env={
            "HERMES_HOME": str(tmp_path),
            "PATH": "/usr/bin",
            # A Windows Python cannot start without SYSTEMROOT.
            **({"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")} if sys.platform == "win32" else {}),
        },
        timeout=60,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["action"] == "block"


def test_status_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert guard.status_path({}).parts[-3:] == (".hermes", "agenomic", "status.json")


@pytest.mark.parametrize("value", ["nan", "inf", "-5", "0", "soon"])
def test_invalid_deadline_override_blocks(tmp_path: Path, value: str) -> None:
    code, out = run(
        tmp_path,
        {"loaded": True, "instance_status": "active", "updated_at": iso()},
        AGENOMIC_HERMES_GUARD_MAX_AGE_S=value,
    )
    assert_block(code, out, "AGENOMIC_HERMES_GUARD_MAX_AGE_S")


def test_status_read_retries_a_transient_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"loaded": True}), encoding="utf-8")
    real_read = Path.read_text
    calls: list[int] = []

    def flaky_read(self: Path, *args: Any, **kwargs: Any) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "sharing violation")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert guard._read_status(path) == {"loaded": True}
    assert len(calls) == 2


def test_status_read_gives_up_on_a_persistent_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(guard, "_SHARING_BACKOFF_S", 0.0)
    with pytest.raises(PermissionError):
        guard._read_status(tmp_path / "status.json")


@pytest.mark.parametrize("value", ["0.5", "2.99"])
def test_deadline_below_the_minimum_blocks(tmp_path: Path, value: str) -> None:
    # The adapter refreshes at most once a second and within a third of the deadline, so a
    # deadline below MIN_MAX_AGE_S would make a healthy adapter look stale between updates.
    fresh = {"loaded": True, "instance_status": "active", "updated_at": iso()}
    assert_block(*run(tmp_path, fresh, AGENOMIC_HERMES_GUARD_MAX_AGE_S=value), "minimum")
    assert run(tmp_path, fresh, AGENOMIC_HERMES_GUARD_MAX_AGE_S="3")[0] == 0
