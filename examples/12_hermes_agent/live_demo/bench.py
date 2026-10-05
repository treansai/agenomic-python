"""Latency and throughput of the Hermes runtime API against a running gateway.

Measures, on the real gateway: authorize (Protect admission, decision row,
signed permit), report (permit verification, settlement), event ingestion
(batches of 500). Shadow mode is used so no isolation attestation is needed;
the admission path is the same Protect pipeline as enforce. Prints JSON.

    python bench.py --gateway http://127.0.0.1:18080 --n 500 --concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import uuid

import httpx

POLICY = """
policy_id: bench
version: "1.0.0"
schema_version: agenomic.policy/v1
scope: { tools: [read_file] }
default_decision: deny
rules:
  - rule_id: reads
    match: { action_type: tool.call, tool_id: read_file }
    decision: allow
"""


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1)))))
    return round(ordered[k], 2)


def summary(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50_ms": pct(values, 50),
        "p95_ms": pct(values, 95),
        "p99_ms": pct(values, 99),
        "mean_ms": round(statistics.mean(values), 2),
    }


def rss_kib(pid: int) -> int:
    with open(f"/proc/{pid}/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:18080")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--gateway-pid", type=int, default=0)
    a = ap.parse_args()
    c = httpx.Client(base_url=a.gateway, timeout=30)
    boot = c.post(
        "/v1/orgs/bootstrap",
        json={
            "name": f"bench {uuid.uuid4().hex[:6]}",
            "owner_email": f"b-{uuid.uuid4().hex[:6]}@bench.example",
        },
    ).json()
    key = {"x-api-key": boot["bootstrap_api_key"]["value"]}
    agent = c.post("/v1/agents", json={"name": "bench-agent"}, headers=key).json()["agent"]
    c.post("/v1/policies", json={"document_text": POLICY}, headers=key).raise_for_status()
    c.post("/v1/policies/bench@1.0.0/release", json={}, headers=key).raise_for_status()
    c.post(
        "/v1/protect/bindings",
        json={
            "policy_id": "bench",
            "version": "1.0.0",
            "scope_kind": "org",
            "scope_ref": "",
            "mode": "shadow",
        },
        headers=key,
    ).raise_for_status()
    created = c.post(
        "/v1/hermes/instances",
        json={"agent_id": agent["id"], "name": "bench", "environment": "bench"},
        headers=key,
    ).json()
    iid = created["instance"]["id"]
    rt = {"authorization": f"Bearer {created['credentials']['runtime_token']}"}
    h = "blake3:" + "a" * 64
    entry = c.post(
        "/v1/hermes/runtime/tools/discovered",
        json={"tools": [{"tool_name": "read_file", "source": "builtin", "schema_hash": h}]},
        headers=rt,
    ).json()["entries"][0]
    c.post(
        f"/v1/hermes/instances/{iid}/catalog/{entry['id']}/decide",
        json={"decision": "approve", "effect": "read"},
        headers=key,
    ).raise_for_status()
    c.patch(
        f"/v1/hermes/instances/{iid}", json={"requested_mode": "shadow"}, headers=key
    ).raise_for_status()
    sid = f"bench_{uuid.uuid4().hex[:12]}"
    c.post(
        "/v1/hermes/runtime/sessions",
        json={"hermes_session_id": sid, "platform": "cli"},
        headers=rt,
    ).raise_for_status()
    rss_before = rss_kib(a.gateway_pid) if a.gateway_pid else None

    def one(i: int) -> tuple[float, float, bool]:
        with httpx.Client(base_url=a.gateway, timeout=30) as cc:
            args = {"path": f"notes/{i}.txt"}
            t0 = time.perf_counter()
            r = cc.post(
                f"/v1/hermes/runtime/sessions/{sid}/actions/authorize",
                json={
                    "tool_call_id": f"call_{i}",
                    "tool": "read_file",
                    "arguments": args,
                    "schema_hash": h,
                },
                headers=rt,
            )
            t1 = time.perf_counter()
            body = r.json()
            ok = r.status_code == 200 and body.get("permit") is not None
            r2 = cc.post(
                f"/v1/hermes/runtime/sessions/{sid}/actions/report",
                json={
                    "logical_call_id": f"call_{i}",
                    "tool": "read_file",
                    "arguments": args,
                    "permit": body.get("permit"),
                    "duration_ms": 1,
                },
                headers=rt,
            )
            t2 = time.perf_counter()
            return (t1 - t0) * 1000, (t2 - t1) * 1000, ok and r2.status_code == 200

    out: dict = {
        "conditions": {
            "gateway": "debug build (unoptimized), Postgres 16 on the same host, 4 vCPU, 15 GiB RAM, loopback HTTP, one client process",
            "mode": "shadow (full Protect admission, decision persisted, ed25519 permit)",
        }
    }
    for conc in (1, a.concurrency):
        start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=conc) as pool:
            results = list(pool.map(one, range(conc * 100000, conc * 100000 + a.n)))
        wall = time.perf_counter() - start
        out[f"concurrency_{conc}"] = {
            "authorize": summary([r[0] for r in results]),
            "report": summary([r[1] for r in results]),
            "successful_pairs": sum(1 for r in results if r[2]),
            "throughput_pairs_per_s": round(a.n / wall, 1),
        }
    batch_lat = []
    for b in range(20):
        events = [
            {
                "event_id": f"ev_{b}_{j}_{uuid.uuid4().hex[:8]}",
                "type": "tool.call.completed",
                "hermes_session_id": sid,
                "seq": j,
                "tool": {"name": "read_file"},
                "status": "ok",
            }
            for j in range(500)
        ]
        t0 = time.perf_counter()
        r = c.post("/v1/hermes/runtime/events", json={"events": events}, headers=rt)
        batch_lat.append((time.perf_counter() - t0) * 1000)
        assert r.json()["accepted"] == 500, r.text
    out["events_batch_500"] = {
        **summary(batch_lat),
        "events_per_s": round(500 * 1000 / statistics.mean(batch_lat), 1),
    }
    if a.gateway_pid:
        out["gateway_rss_kib"] = {"before": rss_before, "after": rss_kib(a.gateway_pid)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
