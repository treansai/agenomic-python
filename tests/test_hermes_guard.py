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


def run(
    home: Path, status: Optional[Any] = None, stdin: str = PAYLOAD, **env: str
) -> tuple[int, str]:
    if status is not None:
        path = home / "agenomic" / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(status if isinstance(status, str) else json.dumps(status))
    out = io.StringIO()
    code = guard.main(
        stdin=io.StringIO(stdin), stdout=out, environ={"HERMES_HOME": str(home), **env}
    )
    return code, out.getvalue()


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
