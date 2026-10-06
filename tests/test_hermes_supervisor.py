from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agenomic.integrations.hermes import supervisor as sup
from agenomic.integrations.hermes.client import HermesApiError
from agenomic.integrations.hermes.supervisor import (
    Supervisor,
    SupervisorSettings,
    build_child_env,
    egress_restricted,
    isolation_report,
    provider_secrets_absent,
    sync_skills,
    writable_by,
)

# The supervisor targets POSIX hosts: mode bits, uids, process groups and signal exit codes.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only supervisor behaviour")

PARENT_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/agent",
    "HERMES_HOME": "/srv/hermes",
    "OPENAI_API_KEY": "sk-openai",
    "ANTHROPIC_API_KEY": "sk-ant",
    "OPENROUTER_API_KEY": "or",
    "GITHUB_TOKEN": "ghp",
    "AWS_SECRET_ACCESS_KEY": "aws",
    "AGENOMIC_HERMES_SUPERVISOR_TOKEN": "agmhs_secret",
    "AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_runtime",
    "DB_PASSWORD": "pw",
    "RANDOM_VAR": "x",
}

# A Windows child cannot start Python without SYSTEMROOT; it is no secret.
PROCESS_ENV = (
    {**PARENT_ENV, "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    if sys.platform == "win32"
    else PARENT_ENV
)


class FakeApi:
    def __init__(self) -> None:
        self.heartbeats: list[dict[str, Any]] = []
        self.acks: list[tuple[str, str, dict[str, Any]]] = []
        self.commands: list[dict[str, Any]] = []
        self.skills: list[dict[str, Any]] = []
        self.fail = False

    def heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            raise HermesApiError("unavailable", "down", 503)
        self.heartbeats.append(body)
        commands, self.commands = self.commands, []
        return {"effective_state": "enforce", "commands": commands}

    def ack_command(self, command_id: str, status: str, detail: dict[str, Any]) -> dict[str, Any]:
        self.acks.append((command_id, status, detail))
        return {}

    def approved_skills(self) -> dict[str, Any]:
        return {"skills": self.skills}


def refuse(addr: tuple[str, int], timeout: float) -> socket.socket:
    raise OSError("network unreachable")


def test_child_env_is_allowlisted_and_scrubbed() -> None:
    env = build_child_env(PARENT_ENV, allow=["RANDOM_VAR", "GITHUB_TOKEN", "OPENAI_API_KEY"])
    assert env == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/agent",
        "HERMES_HOME": "/srv/hermes",
        "AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_runtime",
        "RANDOM_VAR": "x",
    }
    assert provider_secrets_absent(env)
    assert not provider_secrets_absent(PARENT_ENV)
    assert not provider_secrets_absent({"AGENOMIC_HERMES_SUPERVISOR_TOKEN": "x"})
    custom = build_child_env(
        {**PARENT_ENV, "AGENOMIC_RT": "agmhr_custom"}, runtime_token_env="AGENOMIC_RT"
    )
    assert custom["AGENOMIC_RT"] == "agmhr_custom"
    assert "AGENOMIC_HERMES_RUNTIME_TOKEN" not in custom


def test_egress_check() -> None:
    assert egress_restricted(["api.openai.com:443", "[::1]:443"], connect=refuse)
    opened: list[tuple[str, int]] = []

    class Sock:
        def close(self) -> None:
            pass

    def accept(addr: tuple[str, int], timeout: float) -> Sock:
        opened.append(addr)
        return Sock()

    assert not egress_restricted(["api.openai.com:443"], connect=accept)
    assert opened == [("api.openai.com", 443)]
    assert not egress_restricted([])
    assert not egress_restricted(["host:notaport"], connect=refuse)


@posix_only
def test_writability_by_mode_bits(tmp_path: Path) -> None:
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    cfg = ro_dir / "config.yaml"
    cfg.write_text("x")
    os.chmod(cfg, 0o444)
    os.chmod(ro_dir, 0o555)
    other_uid = 65534 if os.getuid() != 65534 else 65533
    try:
        assert not writable_by(cfg, other_uid, [other_uid])
        assert writable_by(cfg, 0, [0])  # root on a writable mount
        assert not writable_by(cfg, os.getuid(), [os.getgid()]) or os.getuid() == 0
        os.chmod(cfg, 0o446)
        assert writable_by(cfg, other_uid, [other_uid])  # world writable file
        os.chmod(cfg, 0o444)
        os.chmod(ro_dir, 0o757)
        assert writable_by(cfg, other_uid, [other_uid])  # file replaceable through its directory
        os.chmod(ro_dir, 0o1777)
        if os.getuid() != other_uid and cfg.stat().st_uid != other_uid:
            assert not writable_by(
                cfg, other_uid, [other_uid]
            )  # sticky: not the owner of the entry
    finally:
        os.chmod(ro_dir, 0o755)


def test_isolation_report(tmp_path: Path) -> None:
    sock_path = tmp_path / "docker.sock"
    report = isolation_report(
        {"PATH": "/bin"},
        child_uid=0,
        child_gids=[0],
        config_paths=[tmp_path / "config.yaml"],
        skills_paths=[],
        forbidden_hosts=["api.openai.com:443"],
        connect=refuse,
        docker_sockets=[str(sock_path)],
    )
    assert report["provider_secrets_absent"] is True
    assert report["egress_restricted"] is True
    assert report["skills_readonly"] is False  # nothing configured cannot be attested
    assert report["docker_socket_absent"] is True
    assert report["runs_as_non_root"] is False
    assert report["config_readonly"] is False  # root on a writable tmp dir
    sock_path.write_text("")
    again = isolation_report(
        {},
        child_uid=1000,
        child_gids=[1000],
        config_paths=[],
        skills_paths=[],
        forbidden_hosts=[],
        docker_sockets=[str(sock_path)],
    )
    assert again["docker_socket_absent"] is False
    assert set(report) == {
        "provider_secrets_absent",
        "egress_restricted",
        "config_readonly",
        "skills_readonly",
        "docker_socket_absent",
        "runs_as_non_root",
        "checked_at",
    }


def test_sync_skills(tmp_path: Path) -> None:
    body = "---\nname: demo\n---\n"
    sha = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
    skills = [
        {"target": "skills/demo/SKILL.md", "version": 1, "digest": sha, "content": body},
        {"target": "skills/../../escape.md", "version": 1, "digest": sha, "content": body},
        {"target": "/etc/passwd", "version": 1, "digest": sha, "content": body},
        {"target": "skills/bad/SKILL.md", "version": 1, "digest": "sha256:00", "content": body},
    ]
    out = tmp_path / "skills"
    assert sync_skills(skills, out) == {"written": 1, "unchanged": 0, "removed": 0, "rejected": 3}
    assert (out / "demo" / "SKILL.md").read_text() == body
    assert not (tmp_path / "escape.md").exists()
    assert sync_skills(skills[:1], out)["unchanged"] == 1
    assert sync_skills([], out)["removed"] == 1
    assert not (out / "demo" / "SKILL.md").exists()


def _skill(name: str, body: str) -> dict[str, Any]:
    return {
        "target": f"skills/{name}/SKILL.md",
        "version": 1,
        "digest": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
        "content": body,
    }


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc/self/fd")
def test_failed_manifest_write_keeps_the_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "skills"
    sync_skills([_skill("a", "A"), _skill("b", "B")], out)
    manifest = out / ".agenomic_manifest.json"
    before = manifest.read_bytes()
    real_fsync = os.fsync

    def full_disk(fd: int) -> None:
        if "agenomic_manifest.json" in os.readlink(f"/proc/self/fd/{fd}"):
            raise OSError(28, "No space left on device")
        real_fsync(fd)

    monkeypatch.setattr(sup.os, "fsync", full_disk)
    with pytest.raises(OSError):
        sync_skills([_skill("a", "A")], out)
    monkeypatch.undo()
    assert manifest.read_bytes() == before, "the previous manifest is intact"
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]
    sync_skills([_skill("a", "A")], out)
    assert not (out / "b" / "SKILL.md").exists()
    assert json.loads(manifest.read_text()) == {"files": ["a/SKILL.md"]}


def make_supervisor(tmp_path: Path, api: FakeApi, argv: list[str]) -> Supervisor:
    settings = SupervisorSettings(
        argv=argv,
        hermes_home=tmp_path / "home",
        skills_dir=tmp_path / "skills",
        grace_s=2.0,
        interval_s=0.05,
        forbidden_hosts=["api.openai.com:443"],
        restart=False,
    )
    return Supervisor(settings, api, environ=PROCESS_ENV, connect=refuse)


SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
# SIGTERM ends a POSIX child with -15; on Windows it is TerminateProcess with exit code 1.
TERM_EXIT = 1 if sys.platform == "win32" else -15


def test_quarantine_stops_and_refuses_restart_then_resume(tmp_path: Path) -> None:
    api = FakeApi()
    s = make_supervisor(tmp_path, api, SLEEPER)
    assert "AGENOMIC_HERMES_SUPERVISOR_TOKEN" not in s.child_env
    assert "OPENAI_API_KEY" not in s.child_env
    assert s.start_child()
    s.heartbeat()
    first = api.heartbeats[-1]
    assert first["process"]["state"] == "running"
    assert first["process"]["pid"]
    assert first["isolation"]["egress_restricted"] is True
    api.commands = [
        {"id": "q1", "kind": "quarantine", "target_kind": "instance", "status": "requested"}
    ]
    s.heartbeat()
    assert [(c, st) for c, st, _ in api.acks] == [("q1", "received"), ("q1", "applied")]
    detail = api.acks[-1][2]
    assert detail["process_state"] == "stopped"
    assert detail["exit_code"] == TERM_EXIT
    assert s.refuse_restart
    assert not s.start_child()
    api.commands = [{"id": "q1", "kind": "quarantine", "status": "received"}]
    s.heartbeat()  # replayed command is not executed twice
    assert len(api.acks) == 2
    api.commands = [
        {"id": "r1", "kind": "resume", "target_kind": "instance", "status": "requested"}
    ]
    s.heartbeat()
    assert api.acks[-1][:2] == ("r1", "applied")
    assert api.acks[-1][2]["restarted"] is True
    assert s.state == "running"
    api.commands = [{"id": "x1", "kind": "pause", "status": "requested"}]
    s.heartbeat()
    assert api.acks[-1][:2] == ("x1", "refused")
    assert s.stop_child() == TERM_EXIT


@posix_only
def test_sigkill_after_grace(tmp_path: Path) -> None:
    api = FakeApi()
    stubborn = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); "
        "time.sleep(60)",
    ]
    s = make_supervisor(tmp_path, api, stubborn)
    s.settings.grace_s = 0.5
    s.start_child()
    import time

    time.sleep(0.5)
    assert s.stop_child() == -9


def test_tick_poll_and_skill_sync(tmp_path: Path) -> None:
    api = FakeApi()
    body = "# s"
    api.skills = [
        {
            "target": "skills/a/SKILL.md",
            "digest": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
            "content": body,
        }
    ]
    s = make_supervisor(tmp_path, api, [sys.executable, "-c", "raise SystemExit(3)"])
    s.start_child()
    assert s.proc is not None
    s.proc.wait()
    s.tick()
    assert s.state == "exited"
    assert s.exit_code == 3
    assert (tmp_path / "skills" / "a" / "SKILL.md").read_text() == body
    assert api.heartbeats[-1]["process"] == {
        "state": "exited",
        "pid": None,
        "exit_code": 3,
        "restarts": 0,
    }
    api.fail = True
    s.heartbeat()  # failure is logged, not raised


def test_main_argument_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sup.main(["--endpoint", "https://a.example"], environ={}) == 2
    assert sup.main(["--", "hermes"], environ={}) == 2
    assert sup.main(["--endpoint", "https://a.example", "--", "hermes"], environ={}) == 2


@posix_only
def test_main_runs_child_until_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_fakes import FakeAgenomic

    server = FakeAgenomic()
    try:
        runs: list[int] = []
        original_tick = Supervisor.tick

        def tick_then_stop(self: Supervisor) -> None:
            original_tick(self)
            runs.append(1)
            if self.state != "running":
                self.request_stop()

        monkeypatch.setattr(Supervisor, "tick", tick_then_stop)
        code = sup.main(
            [
                "--endpoint",
                server.url,
                "--hermes-home",
                str(tmp_path),
                "--interval-s",
                "0.05",
                "--no-restart",
                "--forbidden-host",
                "127.0.0.1:9",
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            environ={
                "AGENOMIC_HERMES_SUPERVISOR_TOKEN": "agmhs_t",
                "PATH": os.environ.get("PATH", ""),
            },
        )
        assert code == 0
        beats = [r for r in server.requests if r.path == "/v1/hermes/supervisor/heartbeat"]
        assert beats
        assert beats[-1].headers["Authorization"] == "Bearer agmhs_t"
        assert beats[-1].body["isolation"]["egress_restricted"] is True
    finally:
        server.close()


@posix_only
def test_no_restart_supervisor_exits_with_its_child(tmp_path: Path) -> None:
    from hermes_fakes import FakeAgenomic

    server = FakeAgenomic()
    try:
        started = time.monotonic()
        code = sup.main(
            [
                "--endpoint",
                server.url,
                "--hermes-home",
                str(tmp_path),
                "--interval-s",
                "0.05",
                "--no-restart",
                "--forbidden-host",
                "127.0.0.1:9",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(3)",
            ],
            environ={
                "AGENOMIC_HERMES_SUPERVISOR_TOKEN": "agmhs_t",
                "PATH": os.environ.get("PATH", ""),
            },
        )
        assert code == 3
        assert time.monotonic() - started < 30
        states = [
            r.body["process"]["state"]
            for r in server.requests
            if r.path == "/v1/hermes/supervisor/heartbeat"
        ]
        assert states[-1] in ("exited", "stopped")
    finally:
        server.close()


@posix_only
def test_missing_protected_path_is_writable_through_a_writable_ancestor(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "config.yaml"
    assert writable_by(deep, os.getuid(), [os.getgid()]), "the child can create a/b/config.yaml"


def test_supervisor_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        SupervisorSettings(argv=["hermes"], hermes_home=Path("/h"), interval_s=-1)
    with pytest.raises(ValidationError):
        SupervisorSettings(argv=[], hermes_home=Path("/h"))
    settings = SupervisorSettings(argv=["hermes"], hermes_home=Path("/h"))
    with pytest.raises(ValidationError):
        settings.grace_s = 0


def test_configured_hermes_home_wins_over_the_inherited_one(tmp_path: Path) -> None:
    settings = SupervisorSettings(argv=["hermes"], hermes_home=tmp_path / "configured")
    s = Supervisor(settings, FakeApi(), environ={**PROCESS_ENV, "HERMES_HOME": "/elsewhere"})
    assert s.child_env["HERMES_HOME"] == str(tmp_path / "configured")


def test_child_drops_supplementary_groups(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class FakeProc:
        pid = 4242

        def poll(self) -> None:
            return None

    def fake_popen(argv: list[str], **kwargs: Any) -> FakeProc:
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    settings = SupervisorSettings(
        argv=["hermes"], hermes_home=tmp_path, child_uid=10001, child_gid=10001
    )
    s = Supervisor(settings, FakeApi(), environ=PROCESS_ENV)
    assert s.start_child()
    assert seen["extra_groups"] == []
    assert s._gids() == [10001]


def test_launch_failure_follows_the_restart_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sup.time, "sleep", lambda _s: None)
    settings = SupervisorSettings(
        argv=[str(tmp_path / "missing-binary")],
        hermes_home=tmp_path,
        max_restarts=2,
        interval_s=0.01,
    )
    s = Supervisor(settings, FakeApi(), environ=PROCESS_ENV, connect=refuse)
    monkeypatch.setattr(s, "sync_skills", lambda: None)
    monkeypatch.setattr(sup.signal, "signal", lambda *_a: None)
    assert s.run() == 1, "the supervisor stops with a failure instead of idling forever"
    assert s.restarts == 2
    assert s.gave_up


def test_failed_supervisor_ack_is_retried(tmp_path: Path) -> None:
    api = FakeApi()
    real = api.ack_command
    failures = {"left": 2}

    def flaky(command_id: str, status: str, detail: dict[str, Any]) -> dict[str, Any]:
        if failures["left"]:
            failures["left"] -= 1
            raise HermesApiError("unreachable", "connection refused", 0)
        return real(command_id, status, detail)

    api.ack_command = flaky  # type: ignore[method-assign]
    api.commands = [{"id": "q9", "kind": "quarantine", "status": "requested"}]
    s = make_supervisor(tmp_path, api, [sys.executable, "-c", "raise SystemExit(0)"])
    s.heartbeat()
    assert api.acks == []
    api.commands = []
    s.tick()
    assert [(c, st) for c, st, _ in api.acks][-1] == ("q9", "applied")


@posix_only
def test_failed_tick_still_stops_the_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = SupervisorSettings(
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
        hermes_home=tmp_path,
        interval_s=0.01,
    )
    s = Supervisor(settings, FakeApi(), environ=PROCESS_ENV, connect=refuse)
    monkeypatch.setattr(s, "sync_skills", lambda: None)
    monkeypatch.setattr(sup.signal, "signal", lambda *_a: None)

    def broken_tick() -> None:
        raise OSError("skill write failed")

    monkeypatch.setattr(s, "tick", broken_tick)
    try:
        code = s.run()
    except OSError:
        code = None
    proc = s.proc
    assert proc is not None
    try:
        assert proc.poll() is not None, "the child is stopped when supervision fails"
        assert code == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)


def test_final_heartbeat_is_report_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class ResumeAfterStop(FakeApi):
        def heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
            if body["process"]["state"] == "stopped":
                # Only the report sent after the child was stopped carries the command.
                self.commands = [{"id": "r9", "kind": "resume", "status": "requested"}]
            return super().heartbeat(body)

    api = ResumeAfterStop()
    s = make_supervisor(tmp_path, api, [sys.executable, "-c", "pass"])
    monkeypatch.setattr(s, "sync_skills", lambda: None)
    monkeypatch.setattr(sup.signal, "signal", lambda *_a: None)
    real_start = s.start_child
    starts: list[bool] = []

    def counting_start() -> bool:
        started = real_start()
        starts.append(started)
        return started

    monkeypatch.setattr(s, "start_child", counting_start)
    try:
        assert s.run() == 0
        assert api.heartbeats[-1]["process"]["state"] == "stopped"
        assert starts == [True], "no child is started once supervision has ended"
        assert s.state == "stopped"
        assert api.acks == [], "the command stays pending for the next supervisor"
    finally:
        proc = s.proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(10)


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GITHUB_TOKEN",
        "DB_PASSWORD",
        "AGENOMIC_HERMES_SUPERVISOR_TOKEN",
        "AGENOMIC_PROVIDER_API_KEY",
    ],
)
def test_runtime_token_env_cannot_name_another_credential(name: str) -> None:
    with pytest.raises(ValidationError):
        SupervisorSettings(argv=["hermes"], hermes_home=Path("/h"), runtime_token_env=name)
    # Even when called directly, the child does not get that credential and the
    # isolation self check does not exempt it.
    env = build_child_env({**PARENT_ENV, name: "agmhr_lookalike"}, runtime_token_env=name)
    assert name not in env
    assert not provider_secrets_absent({name: "agmhr_lookalike"}, runtime_token_env=name)
    assert (
        SupervisorSettings(
            argv=["hermes"], hermes_home=Path("/h"), runtime_token_env="AGENOMIC_RT"
        ).runtime_token_env
        == "AGENOMIC_RT"
    )


def test_runtime_token_must_be_an_agenomic_runtime_token() -> None:
    env = {"PATH": "/bin", "AGENOMIC_HERMES_RUNTIME_TOKEN": "sk-provider-key"}
    assert "AGENOMIC_HERMES_RUNTIME_TOKEN" not in build_child_env(env)
    assert not provider_secrets_absent(env)
    assert provider_secrets_absent({"AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_x"})
    assert sup.runtime_token_problem(env) is not None
    assert sup.runtime_token_problem({"PATH": "/bin"}) is None


@posix_only
@pytest.mark.parametrize(
    ("extra_args", "environ"),
    [
        (["--runtime-token-env", "OPENAI_API_KEY"], {"OPENAI_API_KEY": "sk-openai"}),
        ([], {"AGENOMIC_HERMES_RUNTIME_TOKEN": "sk-openai"}),
    ],
)
def test_main_refuses_a_provider_credential_as_runtime_token(
    tmp_path: Path, extra_args: list[str], environ: dict[str, str]
) -> None:
    from hermes_fakes import FakeAgenomic

    server = FakeAgenomic()
    try:
        code = sup.main(
            [
                "--endpoint",
                server.url,
                "--hermes-home",
                str(tmp_path),
                "--no-restart",
                "--forbidden-host",
                "127.0.0.1:9",
                *extra_args,
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            environ={
                "AGENOMIC_HERMES_SUPERVISOR_TOKEN": "agmhs_t",
                "PATH": os.environ.get("PATH", ""),
                **environ,
            },
        )
        assert code == 2
        assert not [r for r in server.requests if r.path == "/v1/hermes/supervisor/heartbeat"]
    finally:
        server.close()


def test_manifest_encoding_failure_keeps_the_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "skills"
    sync_skills([_skill("a", "A"), _skill("b", "B")], out)
    manifest = out / ".agenomic_manifest.json"
    before = manifest.read_bytes()

    class _Json:
        loads = staticmethod(json.loads)

        @staticmethod
        def dumps(value: object) -> str:
            return "\ud800"  # cannot be encoded: the write fails part way

    monkeypatch.setattr(sup, "json", _Json)
    with pytest.raises(UnicodeEncodeError):
        sync_skills([_skill("a", "A")], out)
    monkeypatch.undo()
    assert manifest.read_bytes() == before
    sync_skills([_skill("a", "A")], out)
    assert not (out / "b" / "SKILL.md").exists()
    assert json.loads(manifest.read_text()) == {"files": ["a/SKILL.md"]}


@pytest.mark.parametrize("corrupt", ["", '{"files": ["a/SKI', '{"files": [1]}', "[]"])
def test_corrupt_manifest_still_removes_unapproved_skills(tmp_path: Path, corrupt: str) -> None:
    out = tmp_path / "skills"
    sync_skills([_skill("a", "A"), _skill("b", "B")], out)
    (out / ".agenomic_manifest.json").write_text(corrupt)
    counts = sync_skills([_skill("a", "A")], out)
    assert counts["removed"] == 1
    assert (out / "a" / "SKILL.md").read_text() == "A"
    assert not (out / "b" / "SKILL.md").exists()
    assert json.loads((out / ".agenomic_manifest.json").read_text()) == {"files": ["a/SKILL.md"]}


def test_skill_named_like_the_manifest_is_rejected(tmp_path: Path) -> None:
    body = '{"files": []}'
    skill = {
        "target": "skills/.agenomic_manifest.json",
        "digest": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
        "content": body,
    }
    assert sync_skills([skill, _skill("a", "A")], tmp_path)["rejected"] == 1
    assert json.loads((tmp_path / ".agenomic_manifest.json").read_text()) == {
        "files": ["a/SKILL.md"]
    }


def test_interrupted_sync_still_names_the_files_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "skills"
    sync_skills([_skill("a", "A")], out)
    written: list[str] = []
    if sys.platform == "win32":
        real = sup._write_atomic

        def crash_after_one_skill(path: Path, data: bytes, mode: int) -> None:
            if path.name == "SKILL.md" and written:
                raise OSError(28, "No space left on device")
            real(path, data, mode)
            if path.name == "SKILL.md":
                written.append(str(path))

        monkeypatch.setattr(sup, "_write_atomic", crash_after_one_skill)
    else:  # POSIX writes go through the descriptor of the skills directory
        real_at = sup._write_atomic_at

        def crash_after_one_skill_at(dir_fd: int, name: str, data: bytes, mode: int) -> None:
            if name == "SKILL.md" and written:
                raise OSError(28, "No space left on device")
            real_at(dir_fd, name, data, mode)
            if name == "SKILL.md":
                written.append(name)

        monkeypatch.setattr(sup, "_write_atomic_at", crash_after_one_skill_at)
    with pytest.raises(OSError):
        sync_skills([_skill("a", "A"), _skill("b", "B"), _skill("c", "C")], out)
    monkeypatch.undo()
    assert (out / "b" / "SKILL.md").exists()
    # b is no longer approved: the manifest written before it named it, so it goes.
    sync_skills([_skill("a", "A")], out)
    assert not (out / "b" / "SKILL.md").exists()


@posix_only
def test_unchanged_skill_with_a_wider_mode_is_narrowed(tmp_path: Path) -> None:
    sync_skills([_skill("a", "A")], tmp_path)
    path = tmp_path / "a" / "SKILL.md"
    os.chmod(path, 0o666)
    assert sync_skills([_skill("a", "A")], tmp_path)["unchanged"] == 1
    assert oct(path.stat().st_mode & 0o777) == "0o644"


# ---------------------------------------------------------------- planted links


@posix_only
@pytest.mark.parametrize("name", ["SKILL.md", ".agenomic_manifest.json"])
def test_link_planted_at_the_old_temporary_name_is_never_followed(
    tmp_path: Path, name: str
) -> None:
    out = tmp_path / "skills"
    (out / "a").mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    folder = out / "a" if name == "SKILL.md" else out
    # The temporary name the supervisor used to derive from its pid alone.
    (folder / f".{name}.{os.getpid()}.tmp").symlink_to(victim)
    assert sync_skills([_skill("a", "A")], out)["written"] == 1
    assert victim.read_text() == "precious", "the planted link's target is untouched"
    assert (out / "a" / "SKILL.md").read_text() == "A"
    assert json.loads((out / ".agenomic_manifest.json").read_text()) == {"files": ["a/SKILL.md"]}


@posix_only
@pytest.mark.parametrize("link", ["file", "directory"])
def test_skill_destination_through_a_symlink_is_not_followed(tmp_path: Path, link: str) -> None:
    out = tmp_path / "skills"
    out.mkdir()
    victim_dir = out / "other"
    victim_dir.mkdir()
    victim = victim_dir / "SKILL.md"
    victim.write_text("precious")
    if link == "file":
        (out / "a").mkdir()
        (out / "a" / "SKILL.md").symlink_to(victim)
    else:
        (out / "a").symlink_to(victim_dir)
    counts = sync_skills([_skill("a", "A")], out)
    assert counts["written"] == 0
    assert counts["rejected"] == 1
    assert victim.read_text() == "precious"
    assert (out / "a" / "SKILL.md").is_symlink() or (out / "a").is_symlink()


@posix_only
def test_symlinked_skills_dir_is_refused_and_reported(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere"
    real.mkdir()
    api = FakeApi()
    api.skills = [_skill("a", "A")]
    s = make_supervisor(tmp_path, api, SLEEPER)
    assert s.settings.skills_dir is not None
    s.settings.skills_dir.symlink_to(real)
    assert s.sync_skills() == {"written": 0, "unchanged": 0, "removed": 0, "rejected": 1}
    assert list(real.iterdir()) == []
    assert s.isolation()["skills_readonly"] is False


@posix_only
@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() != 0, reason="needs chown")
def test_skills_dir_owned_by_another_user_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "skills"
    out.mkdir()
    os.chown(out, 65534, 65534)
    assert sync_skills([_skill("a", "A")], out)["rejected"] == 1
    assert list(out.iterdir()) == []
    # A subdirectory planted by another user is refused too, file by file.
    os.chown(out, 0, 0)
    (out / "a").mkdir()
    os.chown(out / "a", 65534, 65534)
    counts = sync_skills([_skill("a", "A"), _skill("b", "B")], out)
    assert counts == {"written": 1, "unchanged": 0, "removed": 0, "rejected": 1}
    assert not (out / "a" / "SKILL.md").exists()
    assert (out / "b" / "SKILL.md").read_text() == "B"


@posix_only
def test_skills_dir_swapped_for_a_link_after_the_check_is_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "skills"
    victim = tmp_path / "victim"
    victim.mkdir()
    real_problem = sup.skills_dir_problem

    def check_then_swap(path: Path) -> str | None:
        problem = real_problem(path)  # the directory is missing: no problem
        path.symlink_to(victim)  # the agent plants a link right after the check
        return problem

    monkeypatch.setattr(sup, "skills_dir_problem", check_then_swap)
    counts = sync_skills([_skill("a", "A")], out)
    assert counts == {"written": 0, "unchanged": 0, "removed": 0, "rejected": 1}
    assert list(victim.iterdir()) == [], "nothing is written through the planted link"


@posix_only
def test_skills_dir_swapped_for_a_link_at_creation_is_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "skills"
    victim = tmp_path / "victim"
    victim.mkdir()
    real_mkdir = os.mkdir
    planted: list[str] = []

    def mkdir_after_a_plant(path: Any, mode: int = 0o777, **kw: Any) -> None:
        if os.path.basename(os.fspath(path)) == "skills" and not planted:
            out.symlink_to(victim)  # the agent wins the race against the creation
            planted.append("skills")
        real_mkdir(path, mode, **kw)

    monkeypatch.setattr(os, "mkdir", mkdir_after_a_plant)
    counts = sync_skills([_skill("a", "A")], out)
    monkeypatch.undo()
    assert planted
    assert counts["written"] == 0
    assert counts["rejected"] == 1
    assert list(victim.iterdir()) == [], "nothing is written through the planted link"


@posix_only
def test_descriptor_sync_writes_nested_skills_and_removes_stale_ones(tmp_path: Path) -> None:
    out = tmp_path / "missing" / "skills"
    deep = {
        "target": "skills/team/deep/SKILL.md",
        "digest": "sha256:" + hashlib.sha256(b"D").hexdigest(),
        "content": "D",
    }
    counts = sync_skills([_skill("a", "A"), deep], out)
    assert counts == {"written": 2, "unchanged": 0, "removed": 0, "rejected": 0}
    assert (out / "team" / "deep" / "SKILL.md").read_text() == "D"
    assert json.loads((out / ".agenomic_manifest.json").read_text()) == {
        "files": ["a/SKILL.md", "team/deep/SKILL.md"]
    }
    counts = sync_skills([_skill("a", "A")], out)
    assert counts == {"written": 0, "unchanged": 1, "removed": 1, "rejected": 0}
    assert not (out / "team" / "deep" / "SKILL.md").exists()
    # A stale file behind a directory link is kept, and its target untouched.
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "SKILL.md").write_text("precious")
    sync_skills([_skill("a", "A"), deep], out)
    (out / "team" / "deep" / "SKILL.md").unlink()
    (out / "team" / "deep").rmdir()
    (out / "team" / "deep").symlink_to(victim)
    assert sync_skills([_skill("a", "A")], out)["removed"] == 0
    assert (victim / "SKILL.md").read_text() == "precious"


@posix_only
def test_skills_dir_whose_parent_is_a_link_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "parent").symlink_to(real)
    counts = sync_skills([_skill("a", "A")], tmp_path / "parent" / "skills")
    assert counts["rejected"] == 1
    assert counts["written"] == 0
    assert list(real.iterdir()) == []
