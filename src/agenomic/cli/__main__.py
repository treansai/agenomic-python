"""agenomic-py — small Python CLI utility for ATEP and trace inspection.

For full CLI features (bundle, sign, replay), use ``agenomic-cli`` (Rust).

Commands::

    agenomic-py atep verify <segment.atep> --public-key <key.pem>
    agenomic-py atep inspect <segment.atep>
    agenomic-py traces summarize <traces.jsonl>
    agenomic-py keys generate <out.pem> -
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from agenomic._version import __version__
from agenomic.atep.segment import SegmentReader
from agenomic.crypto.signing import SigningKey, VerifyingKey
from agenomic.exceptions import AtepError


def _cmd_atep_verify(args: argparse.Namespace) -> int:
    try:
        reader = SegmentReader(Path(args.segment))
    except AtepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not reader.verify_merkle_root():
        print("error: merkle root mismatch", file=sys.stderr)
        return 3
    vk = VerifyingKey.from_pem_file(Path(args.public_key))
    bad = 0
    total = 0
    for ev in reader.iter_events():
        total += 1
        if not ev.verify(vk):
            bad += 1
    if bad:
        print(f"error: {bad}/{total} events failed verification", file=sys.stderr)
        return 4
    print(f"ok: {total} events verified")
    return 0


def _cmd_atep_inspect(args: argparse.Namespace) -> int:
    try:
        reader = SegmentReader(Path(args.segment))
    except AtepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    types: Counter[str] = Counter()
    for ev in reader.iter_events():
        types[ev.header.event_type] += 1
    summary = {
        "version": reader.version,
        "event_count": reader.event_count,
        "first_hlc": {
            "physical_ms": reader.first_hlc.physical_ms,
            "logical": reader.first_hlc.logical,
            "node_id": reader.first_hlc.node_id,
        },
        "last_hlc": {
            "physical_ms": reader.last_hlc.physical_ms,
            "logical": reader.last_hlc.logical,
            "node_id": reader.last_hlc.node_id,
        },
        "merkle_root": reader.merkle_root.hex(),
        "event_types": dict(types),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_traces_summarize(args: argparse.Namespace) -> int:
    path = Path(args.traces)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    count = 0
    agents: Counter[str] = Counter()
    durations: list[int] = []
    errors = 0
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                env = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            agents[env.get("agent_id", "?")] += 1
            if env.get("error"):
                errors += 1
            d = env.get("duration_ms")
            if isinstance(d, int):
                durations.append(d)
    avg = sum(durations) / len(durations) if durations else 0
    summary = {
        "envelopes": count,
        "errors": errors,
        "agents": dict(agents),
        "average_duration_ms": avg,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_keys_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"error: {out} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    sk = SigningKey.generate()
    sk.write_pem_file(out)
    pub = out.with_suffix(out.suffix + ".pub")
    sk.write_public_pem_file(pub)
    print(f"wrote private={out} public={pub} key_id={sk.key_id}")
    return 0


def _cmd_benchmark_serve(args: argparse.Namespace) -> int:
    import importlib

    from agenomic._client import Client
    from agenomic.benchmarks import AgentTargetBridge, FixtureBridge, serve_bridge

    if args.bridge == "fixture":
        bridge: AgentTargetBridge = FixtureBridge()
    else:
        module_name, _, attr = args.bridge.partition(":")
        if not attr:
            print("error: --bridge must be module:attribute or 'fixture'", file=sys.stderr)
            return 2
        target = getattr(importlib.import_module(module_name), attr)
        bridge = target() if isinstance(target, type) else target
        if not isinstance(bridge, AgentTargetBridge):
            print(
                "error: --bridge must resolve to an AgentTargetBridge instance or class",
                file=sys.stderr,
            )
            return 2
    client = Client(api_key=args.api_key, base_url=args.base_url)
    answered = serve_bridge(
        client,
        bridge,
        agent=args.agent,
        release_id=args.release,
        max_turns=args.max_turns,
        idle_timeout=args.idle_timeout,
    )
    print(f"bridge stopped after {answered} turn(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agenomic-py",
        description="Agenomic Python utility CLI",
    )
    parser.add_argument("--version", action="version", version=f"agenomic-py {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    atep = sub.add_parser("atep", help="ATEP segment commands")
    atep_sub = atep.add_subparsers(dest="atep_command", required=True)

    verify = atep_sub.add_parser("verify", help="verify an ATEP segment")
    verify.add_argument("segment", type=str)
    verify.add_argument("--public-key", required=True, type=str)
    verify.set_defaults(func=_cmd_atep_verify)

    inspect = atep_sub.add_parser("inspect", help="inspect an ATEP segment")
    inspect.add_argument("segment", type=str)
    inspect.set_defaults(func=_cmd_atep_inspect)

    traces = sub.add_parser("traces", help="trace JSONL utilities")
    traces_sub = traces.add_subparsers(dest="traces_command", required=True)
    summarize = traces_sub.add_parser("summarize", help="summarize a JSONL file")
    summarize.add_argument("traces", type=str)
    summarize.set_defaults(func=_cmd_traces_summarize)

    keys = sub.add_parser("keys", help="ed25519 key utilities")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    gen = keys_sub.add_parser("generate", help="generate a new ed25519 PEM")
    gen.add_argument("out", type=str)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=_cmd_keys_generate)

    benchmark = sub.add_parser("benchmark", help="RMP benchmark bridge")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    serve = benchmark_sub.add_parser("serve", help="serve your agent to Agenomic benchmark turns")
    serve.add_argument("--agent", required=True, help="agent id as used by the RMP session")
    serve.add_argument("--release", default=None, help="release id served by this bridge")
    serve.add_argument(
        "--bridge", required=True, help="module:attribute of an AgentTargetBridge, or 'fixture'"
    )
    serve.add_argument(
        "--base-url",
        default=os.environ.get("AGENOMIC_BASE_URL"),
        required="AGENOMIC_BASE_URL" not in os.environ,
    )
    serve.add_argument("--api-key", default=os.environ.get("AGENOMIC_API_KEY"))
    serve.add_argument("--max-turns", type=int, default=None)
    serve.add_argument(
        "--idle-timeout", type=float, default=None, help="stop after this many idle seconds"
    )
    serve.set_defaults(func=_cmd_benchmark_serve)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
