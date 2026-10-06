"""Live demo: Hermes Agent under Agenomic control, end to end, no fakes in the control path.

Real components: the agenomic-cloud gateway (Postgres), the pinned Hermes
Agent (v2026.9.24) with the Agenomic adapter loaded by entry point, the
supervisor, the guard shell hook. The only test double is the upstream model
(``fake_model.py``), reachable by the gateway and NOT by the agent.

Isolation is real: Hermes runs as uid 10001 inside a Linux network namespace
whose only route is a veth to the gateway port, in a private mount namespace
that hides /run (no Docker socket), with HERMES_HOME config and skills owned
by root. The supervisor attests this from inside the namespaces.

Run as root on Linux with iproute2 and iptables, the gateway listening on
0.0.0.0:18080 with AGENOMIC_HERMES_ALLOW_PRIVATE_UPSTREAMS=1 (the scripted model
upstream is on loopback; production upstreams must be public https) and the
venv at /opt/agm-hermes-venv (docs/hermes.md):

    python live_demo.py --out ./evidence

Steps: connect -> allowed action -> blocked action -> human approval ->
controlled delegation -> control incidents (bypass attempt, pause,
quarantine) -> evidence in RMP and Protect.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from agenomic.integrations.hermes.config import render_hermes_config

HERE = Path(__file__).parent
VENV = Path("/opt/agm-hermes-venv")
ROOT = Path("/var/lib/agm-hermes")
HOME = ROOT / "home"
WORKSPACE = ROOT / "workspace"
NETNS = "agm-hermes"
HOST_IP = "10.231.0.1"
NS_IP = "10.231.0.2"
PORT = 18080
MODEL_PORT = 18090
UID = 10001

POLICY = """
policy_id: hermes-demo
version: "1.0.0"
schema_version: agenomic.policy/v1
scope:
  tools: [read_file, search_files, write_file, patch, terminal, process_manage, delegate_task]
default_decision: deny
rules:
  - rule_id: reads
    match: { action_type: tool.call, tool_id: read_file }
    decision: allow
  - rule_id: search
    match: { action_type: tool.call, tool_id: search_files }
    decision: allow
  - rule_id: delegation
    match: { action_type: tool.call, tool_id: delegate_task }
    decision: allow
  - rule_id: writes-need-a-human
    match: { action_type: tool.call, tool_id: write_file }
    decision: require_approval
  - rule_id: no-shell
    match: { action_type: tool.call, tool_id: terminal }
    decision: deny
    non_derogable: true
"""

EFFECTS = {
    "read_file": "read",
    "search_files": "read",
    "write_file": "reversible_write",
    "patch": "reversible_write",
    "terminal": "irreversible_write",
    "process_manage": "irreversible_write",
    "delegate_task": "irreversible_write",
}


class Demo:
    def __init__(self, out: Path, base: str) -> None:
        self.out = out
        self.base = base
        self.log: list[dict[str, Any]] = []
        self.secrets: list[str] = []
        self.user = httpx.Client(base_url=base, timeout=30)
        self.api_key = ""

    def scrub(self, value: Any) -> Any:
        text = json.dumps(value, default=str)
        for s in self.secrets:
            text = text.replace(s, "<redacted-token>")
        return json.loads(text)

    def record(self, step: str, what: str, data: Any) -> None:
        entry = {
            "step": step,
            "what": what,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": self.scrub(data),
        }
        self.log.append(entry)
        print(f"[{step}] {what}")

    def call(
        self,
        method: str,
        path: str,
        body: Optional[Any] = None,
        *,
        token: Optional[str] = None,
        expect: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, Any]:
        h = dict(headers or {})
        if token:
            h["authorization"] = f"Bearer {token}"
            r = httpx.request(method, self.base + path, json=body, headers=h, timeout=30)
        else:
            h["x-api-key"] = self.api_key
            r = self.user.request(method, path, json=body, headers=h)
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text[:500]}
        if expect is not None and r.status_code != expect:
            raise SystemExit(
                f"{method} {path} -> {r.status_code}, expected {expect}: {self.scrub(data)}"
            )
        return r.status_code, data

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "steps.json").write_text(
            json.dumps(self.log, indent=2) + "\n", encoding="utf-8"
        )


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), check=check, text=True, capture_output=True)


def setup_namespaces() -> None:
    sh("ip", "netns", "del", NETNS, check=False)
    sh("ip", "link", "del", "veth-agm", check=False)
    sh("ip", "netns", "add", NETNS)
    sh("ip", "link", "add", "veth-agm", "type", "veth", "peer", "name", "veth-hermes")
    sh("ip", "link", "set", "veth-hermes", "netns", NETNS)
    sh("ip", "addr", "add", f"{HOST_IP}/30", "dev", "veth-agm")
    sh("ip", "link", "set", "veth-agm", "up")
    for cmd in (
        ["ip", "addr", "add", f"{NS_IP}/30", "dev", "veth-hermes"],
        ["ip", "link", "set", "veth-hermes", "up"],
        ["ip", "link", "set", "lo", "up"],
    ):
        sh("ip", "netns", "exec", NETNS, *cmd)
    sh("iptables", "-D", "INPUT", "-i", "veth-agm", "-j", "AGM_HERMES", check=False)
    sh("iptables", "-F", "AGM_HERMES", check=False)
    sh("iptables", "-X", "AGM_HERMES", check=False)
    sh("iptables", "-N", "AGM_HERMES")
    sh("iptables", "-A", "AGM_HERMES", "-p", "tcp", "--dport", str(PORT), "-j", "ACCEPT")
    sh(
        "iptables",
        "-A",
        "AGM_HERMES",
        "-m",
        "state",
        "--state",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    )
    sh("iptables", "-A", "AGM_HERMES", "-j", "DROP")
    sh("iptables", "-I", "INPUT", "-i", "veth-agm", "-j", "AGM_HERMES")
    sh("iptables", "-D", "FORWARD", "-i", "veth-agm", "-j", "DROP", check=False)
    sh("iptables", "-I", "FORWARD", "-i", "veth-agm", "-j", "DROP")


def setup_filesystem(endpoint: str) -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for d in (HOME, HOME / "skills", HOME / "plugins", HOME / "hooks"):
        d.mkdir(parents=True, exist_ok=True)
        os.chown(d, 0, 0)
        d.chmod(0o755)
    HOME.chmod(0o1777)
    env_file = HOME / ".env"
    env_file.write_text("", encoding="utf-8")
    env_file.chmod(0o644)
    writable = (
        "backups",
        "logs",
        "logs/curator",
        "sessions",
        "cache",
        "pending",
        "agenomic",
        "memories",
        "cron",
        "pairing",
        "image_cache",
        "audio_cache",
    )
    for d in (WORKSPACE, ROOT / "user", *(HOME / w for w in writable)):
        d.mkdir(parents=True, exist_ok=True)
        os.chown(d, UID, UID)
    (WORKSPACE / "notes.txt").write_text(
        "Quarterly notes: revenue up 4 percent, two incidents closed.\n", encoding="utf-8"
    )
    os.chown(WORKSPACE / "notes.txt", UID, UID)
    allowlist = HOME / "shell-hooks-allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "approvals": [
                    {
                        "event": "pre_tool_call",
                        "command": "agenomic-hermes-guard",
                        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "script_mtime_at_approval": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    allowlist.chmod(0o644)
    soul = HOME / "SOUL.md"
    soul.write_text(default_soul(), encoding="utf-8")
    soul.chmod(0o644)
    rendered = render_hermes_config(
        endpoint,
        model="demo-model",
        settings={"timeouts": {"decision_s": 10}, "buffer": {"flush_interval_s": 0.5}},
    )
    rendered["model"]["default"] = "demo-model"
    config = HOME / "config.yaml"
    config.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    os.chown(config, 0, 0)
    config.chmod(0o644)


def default_soul() -> str:
    probe = subprocess.run(
        [
            str(VENV / "bin" / "python"),
            "-c",
            "from hermes_cli.config import DEFAULT_SOUL_MD; print(DEFAULT_SOUL_MD, end='')",
        ],
        env={"HERMES_HOME": "/tmp/agm-soul-probe", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return probe.stdout


def supervisor_cmd(endpoint: str, prompt: str) -> list[str]:
    inner = [
        str(VENV / "bin" / "agenomic-hermes-supervisor"),
        "--endpoint",
        endpoint,
        "--hermes-home",
        str(HOME),
        "--skills-dir",
        str(HOME / "skills"),
        "--config-path",
        str(HOME / "config.yaml"),
        "--readonly-path",
        str(HOME / "plugins"),
        "--forbidden-host",
        "api.openai.com:443",
        "--forbidden-host",
        f"{HOST_IP}:{MODEL_PORT}",
        "--forbidden-host",
        f"{HOST_IP}:5432",
        "--allow-env",
        "HERMES_HOME",
        "--child-uid",
        str(UID),
        "--child-gid",
        str(UID),
        "--interval-s",
        "2",
        "--no-restart",
        "--grace-s",
        "5",
        "--",
        str(VENV / "bin" / "python"),
        str(HERE / "run_hermes.py"),
        prompt,
    ]
    script = 'mount -t tmpfs tmpfs /run && mount -t tmpfs tmpfs /var/run 2>/dev/null; cd /var/lib/agm-hermes/workspace && exec "$@"'
    return ["ip", "netns", "exec", NETNS, "unshare", "-m", "sh", "-c", script, "sh", *inner]


def run_hermes(
    demo: Demo,
    endpoint: str,
    prompt: str,
    runtime: str,
    supervisor: str,
    *,
    background: bool = False,
) -> Any:
    env = {
        "PATH": f"{VENV}/bin:/usr/bin:/bin",
        "HOME": str(ROOT / "user"),
        "HERMES_HOME": str(HOME),
        "VIRTUAL_ENV": str(VENV),
        "AGENOMIC_HERMES_RUNTIME_TOKEN": runtime,
        "AGENOMIC_HERMES_SUPERVISOR_TOKEN": supervisor,
        "PYTHONUNBUFFERED": "1",
    }
    cmd = supervisor_cmd(endpoint, prompt)
    logs = demo.out / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{time.strftime('%H%M%S')}-{prompt.split()[0]}.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if background:
        return proc, log_path
    try:
        proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()
    return collect(demo, proc.returncode, log_path)


def collect(demo: Demo, code: Optional[int], log_path: Path) -> Any:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    scrubbed = demo.scrub(text)
    log_path.write_text(scrubbed, encoding="utf-8")
    result = None
    for line in text.splitlines():
        if line.startswith("RESULT:"):
            result = demo.scrub(json.loads(line[len("RESULT:") :]))
    tail = "\n".join(scrubbed.splitlines()[-15:])
    return {"exit_code": code, "result": result, "log": str(log_path.name), "log_tail": tail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway", default=f"http://127.0.0.1:{PORT}")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("run as root: the demo creates namespaces and a dedicated user")
    endpoint = f"http://{HOST_IP}:{PORT}"
    model_state = args.out / "model"
    if model_state.exists():
        shutil.rmtree(model_state)
    model = subprocess.Popen(
        [sys.executable, str(HERE / "fake_model.py"), str(MODEL_PORT), str(model_state)]
    )
    demo = Demo(args.out, args.gateway)
    try:
        setup_namespaces()
        setup_filesystem(endpoint)
        scenario(demo, endpoint, model_state)
    finally:
        demo.save()
        model.terminate()


def scenario(demo: Demo, endpoint: str, model_state: Path) -> None:
    email = f"owner-{uuid.uuid4().hex[:8]}@demo.example"
    r = demo.user.post(
        "/v1/orgs/bootstrap",
        json={"name": f"Hermes demo {uuid.uuid4().hex[:6]}", "owner_email": email},
    )
    if r.status_code != 201:
        raise SystemExit(f"bootstrap failed: {r.status_code} {r.text[:300]}")
    boot = r.json()
    demo.api_key = boot["bootstrap_api_key"]["value"]
    demo.secrets.append(demo.api_key)
    demo.record(
        "0",
        "workspace bootstrapped (owner API key, org signing key provisioned)",
        {"org": boot["organization"]["slug"]},
    )

    _, agent = demo.call("POST", "/v1/agents", {"name": "support-agent"}, expect=201)
    demo.call("POST", "/v1/policies", {"document_text": POLICY}, expect=201)
    demo.call("POST", "/v1/policies/hermes-demo@1.0.0/release", {}, expect=200)
    demo.call(
        "POST",
        "/v1/protect/bindings",
        {
            "policy_id": "hermes-demo",
            "version": "1.0.0",
            "scope_kind": "org",
            "scope_ref": "",
            "mode": "enforce",
        },
        expect=201,
    )
    profile = {
        "schema_version": "agenomic.hermes.profile/v1",
        "model": {
            "allowed": [{"provider": "custom", "model": "demo-model"}],
            "max_tokens_ceiling": 2048,
            "token_budget_per_root_session": 200000,
            "upstream": {"base_url": f"http://127.0.0.1:{MODEL_PORT}/v1"},
        },
        "delegation": {
            "max_depth": 1,
            "max_children_per_session": 2,
            "max_concurrent_children": 1,
            "max_descendants": 2,
        },
        "protected_paths": [
            "/var/lib/agm-hermes/home/skills",
            "/var/lib/agm-hermes/home/config.yaml",
            "/var/lib/agm-hermes/home/plugins",
        ],
    }
    _, created = demo.call(
        "POST",
        "/v1/hermes/instances",
        {
            "agent_id": agent["agent"]["id"],
            "name": "support-prod-1",
            "environment": "production",
            "profile": profile,
        },
        expect=201,
    )
    instance = created["instance"]
    iid = instance["id"]
    runtime = created["credentials"]["runtime_token"]
    supervisor = created["credentials"]["supervisor_token"]
    demo.secrets += [runtime, supervisor]
    demo.record(
        "1",
        "Hermes instance enrolled; runtime and supervisor credentials issued once",
        {"instance": instance},
    )
    _, rmp = demo.call(
        "POST",
        "/v1/rmp/sessions",
        {
            "spec_version": "agenomic.rmp/v0.1",
            "agent_id": instance["agent_ref"],
            "environment": "production",
        },
    )
    demo.record("1", "RMP session opened by the operator (grouping only; it triggers nothing)", rmp)

    first = run_hermes(demo, endpoint, "TASK_READ please read notes.txt", runtime, supervisor)
    demo.record(
        "2",
        "first run in OBSERVE: connection test, tool inventory reported, no Agenomic decision",
        first,
    )
    _, inst = demo.call("GET", f"/v1/hermes/instances/{iid}")
    demo.record(
        "2",
        "instance after hello: version, connection, protection state",
        {
            "connection": inst["connection"],
            "hermes_version": inst["hermes_version"],
            "hermes_commit": inst["hermes_commit"],
            "adapter_version": inst["adapter_version"],
            "protection": inst["protection"],
            "isolation": inst["isolation"],
        },
    )

    _, catalog = demo.call("GET", f"/v1/hermes/instances/{iid}/catalog")
    approved = []
    for entry in catalog["entries"]:
        effect = EFFECTS.get(entry["tool_name"])
        if effect and entry["status"] == "discovered":
            demo.call(
                "POST",
                f"/v1/hermes/instances/{iid}/catalog/{entry['id']}/decide",
                {"decision": "approve", "effect": effect},
                expect=200,
            )
            approved.append(entry["tool_name"])
    demo.record(
        "3",
        "operator approved the tool contracts it knows (others stay discovered, hence refused in enforce)",
        {"approved": approved, "discovered": [e["tool_name"] for e in catalog["entries"]]},
    )

    status, mode = demo.call("PATCH", f"/v1/hermes/instances/{iid}", {"requested_mode": "enforce"})
    demo.record("3", f"switch to ENFORCE -> HTTP {status}", mode.get("protection", mode))
    if status != 200:
        raise SystemExit("enforce refused; see steps.json")

    allowed = run_hermes(demo, endpoint, "TASK_READ please read notes.txt", runtime, supervisor)
    demo.record(
        "4", "allowed action: read_file admitted with a signed permit, executed, reported", allowed
    )

    blocked = run_hermes(demo, endpoint, "TASK_SHELL run a command", runtime, supervisor)
    demo.record(
        "5",
        "blocked action: terminal denied before execution",
        {**blocked, "external_check": {"PWNED_exists": (WORKSPACE / "PWNED").exists()}},
    )

    skill = run_hermes(demo, endpoint, "TASK_SKILL write a skill", runtime, supervisor)
    demo.record(
        "5",
        "persistence write into the skills directory denied (read only mount is the prevention)",
        {**skill, "external_check": {"skill_exists": (HOME / "skills" / "evil").exists()}},
    )

    proc, proc_log = run_hermes(
        demo, endpoint, "TASK_WRITE write the report", runtime, supervisor, background=True
    )
    approval_id = None
    for _ in range(120):
        time.sleep(1)
        _, approvals = demo.call("GET", "/v1/protect/approvals?status=pending")
        pending = approvals.get("approvals", [])
        if pending:
            approval_id = pending[0]["id"]
            break
    demo.record(
        "6",
        "write_file needs a human: approval pending, file not written yet",
        {"approval_id": approval_id, "report_exists_before": (WORKSPACE / "report.txt").exists()},
    )
    if approval_id:
        status, decided = demo.call(
            "POST",
            f"/v1/protect/approvals/{approval_id}/decide",
            {"decision": "approve", "comment": "reviewed in the demo"},
        )
        demo.record("6", f"human approved -> HTTP {status}", decided)
        model_state.mkdir(parents=True, exist_ok=True)
        (model_state / "approved").write_text("1", encoding="utf-8")
    try:
        proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    result = collect(demo, proc.returncode, proc_log)
    report = WORKSPACE / "report.txt"
    demo.record(
        "6",
        "controlled retry after approval: executed exactly once",
        {
            "result": result,
            "report_exists": report.exists(),
            "report_content": report.read_text() if report.exists() else None,
        },
    )

    delegated = run_hermes(
        demo, endpoint, "TASK_DELEGATE summarize notes with a helper", runtime, supervisor
    )
    _, sessions = demo.call("GET", f"/v1/hermes/instances/{iid}/sessions")
    children = [s for s in sessions["sessions"] if s.get("parent_id")]
    demo.record(
        "7",
        "controlled delegation: reservation, child session restricted to the parent tool set",
        {"run": delegated, "child_sessions": children},
    )

    bypass = sh(
        "ip",
        "netns",
        "exec",
        NETNS,
        str(VENV / "bin" / "python"),
        "-c",
        "import socket\nfor h,p in [('api.openai.com',443),('10.231.0.1',18090),('10.231.0.1',5432)]:\n    try:\n        socket.create_connection((h,p),timeout=3); print(h,p,'REACHABLE')\n    except OSError as e: print(h,p,'refused:',type(e).__name__)",
        check=False,
    )
    demo.record(
        "8",
        "incident: direct provider and internal service access from the agent namespace",
        {"stdout": bypass.stdout, "stderr": bypass.stderr[-300:]},
    )

    _, paused = demo.call(
        "POST",
        f"/v1/hermes/instances/{iid}/commands",
        {"kind": "pause", "target_kind": "instance", "reason": "demo incident"},
        headers={"idempotency-key": f"pause-{uuid.uuid4().hex}"},
    )
    after_pause = run_hermes(demo, endpoint, "TASK_READ please read notes.txt", runtime, supervisor)
    demo.record(
        "8",
        "incident: instance paused, every protected action refused",
        {"command": paused, "run": after_pause},
    )
    demo.call(
        "POST",
        f"/v1/hermes/instances/{iid}/commands",
        {"kind": "resume", "target_kind": "instance", "reason": "demo"},
    )

    long_run, long_log = run_hermes(
        demo, endpoint, "TASK_WRITE write the report", runtime, supervisor, background=True
    )
    (model_state / "approved").unlink(missing_ok=True)
    time.sleep(8)
    _, quarantine = demo.call(
        "POST",
        f"/v1/hermes/instances/{iid}/commands",
        {"kind": "quarantine", "target_kind": "instance", "reason": "demo quarantine"},
    )
    for _ in range(60):
        time.sleep(1)
        _, cmds = demo.call("GET", f"/v1/hermes/instances/{iid}/commands")
        q = next(c for c in cmds["commands"] if c["id"] == quarantine["id"])
        if q["status"] in ("applied", "refused", "expired"):
            break
    try:
        long_run.wait(timeout=60)
    except subprocess.TimeoutExpired:
        long_run.kill()
        long_run.wait()
    demo.record(
        "8",
        "incident: quarantine executed by the supervisor, applied only after the process exited",
        {"command": q, "run": collect(demo, long_run.returncode, long_log)},
    )

    _, instance_after = demo.call("GET", f"/v1/hermes/instances/{iid}")
    _, events = demo.call("GET", f"/v1/hermes/instances/{iid}/events?limit=500")
    _, decisions = demo.call("GET", "/v1/protect/decisions?limit=200")
    _, calls = demo.call("GET", f"/v1/hermes/instances/{iid}/model-calls")
    _, commands = demo.call("GET", f"/v1/hermes/instances/{iid}/commands")
    _, rmp_session = (
        demo.call("GET", f"/v1/rmp/sessions/{rmp.get('session', rmp).get('session_id', '')}")
        if isinstance(rmp, dict)
        else (0, {})
    )
    upstream = (
        [json.loads(line) for line in (model_state / "requests.jsonl").read_text().splitlines()]
        if (model_state / "requests.jsonl").exists()
        else []
    )
    by_source: dict[str, int] = {}
    for e in events["events"]:
        key = f"{e['source']}/{e['trust']}"
        by_source[key] = by_source.get(key, 0) + 1
    demo.record(
        "9",
        "evidence: decisions (Protect), events by source and trust, model calls, commands, RMP",
        {
            "instance_state": instance_after["protection"]["effective_state"],
            "events_by_source_trust": by_source,
            "decisions": [
                {
                    k: d.get(k)
                    for k in (
                        "tool",
                        "outcome",
                        "effective_mode",
                        "reason_codes",
                        "policy_snapshot_digest",
                        "permit_ref",
                        "approval_id",
                    )
                }
                for d in decisions.get("decisions", [])
            ],
            "model_calls": calls["model_calls"],
            "upstream_requests_received": len(upstream),
            "upstream_saw_runtime_token": any(u["authorization_header_present"] for u in upstream),
            "commands": commands["commands"],
            "rmp_session": rmp_session,
        },
    )
    (demo.out / "events.json").write_text(
        json.dumps(demo.scrub(events), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
