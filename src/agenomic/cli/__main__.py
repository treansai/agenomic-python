"""agenomic-py: small Python CLI utility for ATEP and trace inspection.

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
from typing import TYPE_CHECKING, Any, Optional, Sequence

from agenomic._version import __version__
from agenomic.atep.segment import SegmentReader
from agenomic.crypto.signing import SigningKey, VerifyingKey
from agenomic.exceptions import ApiError, AtepError

if TYPE_CHECKING:
    from agenomic._client import Client


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


def _cloud_client() -> Optional[Client]:
    from agenomic._client import Client

    client = Client.from_env()
    return client if client.is_cloud and client.api_key else None


def _print_json(document: Any, out: Optional[str] = None) -> None:
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        Path(out).write_text(text, encoding="utf-8")


def _api_failure(error: ApiError) -> int:
    reason = f" ({error.reason})" if error.reason else ""
    print(f"error: {error.code}{reason}: {error.message}", file=sys.stderr)
    return 1


def _cmd_prompts_scan(args: argparse.Namespace) -> int:
    from agenomic.prompts.discovery import DEFAULT_EXCLUDES, scan_paths

    target = Path(args.path)
    if not target.exists():
        print(f"error: {target} not found", file=sys.stderr)
        return 2
    root = Path(args.root) if args.root else (target if target.is_dir() else target.parent)
    try:
        report = scan_paths(
            [target],
            root=root,
            max_files=args.max_files,
            max_file_bytes=args.max_file_bytes,
            exclude=(*DEFAULT_EXCLUDES, *args.exclude),
            label=args.label,
            commit=args.commit,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print_json(report, args.out)
    statuses = Counter(candidate["status"] for candidate in report["candidates"])
    scanned = sum(1 for entry in report["files"] if entry["status"] == "scanned")
    summary = ", ".join(f"{name} {statuses[name]}" for name in sorted(statuses)) or "none"
    print(f"scanned {scanned} file(s); candidates: {summary}", file=sys.stderr)
    return 0


def _cmd_prompts_import(args: argparse.Namespace) -> int:
    from agenomic.prompts.importer import apply_import, new_idempotency_key, upload_report

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: cannot read {args.report}: {error}", file=sys.stderr)
        return 2
    if args.declare_slots and args.slots_revision is None:
        print("error: --declare-slots needs --slots-revision", file=sys.stderr)
        return 2
    client = _cloud_client()
    if client is None:
        print("error: set AGENOMIC_ENDPOINT and AGENOMIC_API_KEY", file=sys.stderr)
        return 2
    try:
        record = upload_report(client, report, agent_id=args.agent_id)
        plan = record["plan"]
        _print_json(record)
        print(f"plan {plan['plan_id']} {plan['plan_digest']}", file=sys.stderr)
        if not args.apply:
            return 0
        result = apply_import(
            client,
            plan,
            idempotency_key=args.idempotency_key or new_idempotency_key(),
            agent_id=args.agent_id,
            mode=args.mode,
            declare_slots=args.declare_slots,
            expected_slots_revision=args.slots_revision,
        )
        _print_json(result)
    except ApiError as error:
        return _api_failure(error)
    finally:
        client.close()
    return 0


def _cmd_prompts_render(args: argparse.Namespace) -> int:
    from agenomic.prompts.importer import load_prompt_file
    from agenomic.prompts.render import RenderedMessage, render_content

    try:
        variables = json.loads(Path(args.vars).read_text(encoding="utf-8")) if args.vars else {}
    except (OSError, ValueError) as error:
        print(f"error: cannot read {args.vars}: {error}", file=sys.stderr)
        return 2
    if not isinstance(variables, dict):
        print("error: --vars must hold a JSON object", file=sys.stderr)
        return 2
    try:
        if Path(args.target).exists():
            document = load_prompt_file(Path(args.target))
            result = render_content(
                document["content"], variables, fragments=lambda prompt_id, version, digest: None
            )
        else:
            client = _cloud_client()
            if client is None:
                print("error: set AGENOMIC_ENDPOINT and AGENOMIC_API_KEY", file=sys.stderr)
                return 2
            try:
                result = client.prompts.get(args.target).render(variables)
            finally:
                client.close()
    except ApiError as error:
        return _api_failure(error)
    messages = None
    if result.messages is not None:
        messages = [
            item.to_dict() if isinstance(item, RenderedMessage) else item
            for item in result.messages
        ]
    _print_json(
        {
            "kind": result.kind,
            "text": result.text,
            "messages": messages,
            "content_digest": result.content_digest,
            "rendered_hash": result.rendered_hash,
        }
    )
    return 0


def _cmd_prompts_digest(args: argparse.Namespace) -> int:
    from agenomic.prompts.digest import prompt_digest
    from agenomic.prompts.errors import AjsError
    from agenomic.prompts.importer import load_prompt_file
    from agenomic.prompts.render import validate_content

    try:
        document = load_prompt_file(Path(args.file))
        report = validate_content(
            document["content"], fragments=lambda prompt_id, version, digest: None
        )
        if any(item.code != "fragment_not_found" for item in report.errors):
            report.raise_for_errors()
        print(prompt_digest(document["content"]))
    except ApiError as error:
        return _api_failure(error)
    except AjsError as error:
        print(f"error: {error.reason} at {error.value_path or '/'}", file=sys.stderr)
        return 1
    return 0


def _cmd_prompts_bundle_verify(args: argparse.Namespace) -> int:
    from agenomic.prompts.bundle import BundleTrust, PromptBundle

    try:
        trust = BundleTrust.from_pem_files(*args.trust_key) if args.trust_key else None
        bundle = PromptBundle.load(
            Path(args.bundle),
            expected_workspace_id=args.workspace,
            expected_agent_id=args.agent,
            trust=trust,
            expected_bundle_digest=args.expect_bundle_digest,
        )
    except ApiError as error:
        return _api_failure(error)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print_json(
        {
            "prompt_bundle_digest": bundle.prompt_bundle_digest,
            "prompt_manifest_digest": bundle.prompt_manifest_digest,
            "prompt_refs": bundle.prompt_refs,
            "slots": bundle.slots(),
        }
    )
    return 0


def _add_prompts_parser(sub: Any) -> None:
    prompts = sub.add_parser("prompts", help="managed prompt utilities")
    prompts_sub = prompts.add_subparsers(dest="prompts_command", required=True)

    scan = prompts_sub.add_parser(
        "scan", help="discover prompts statically; the code is never imported or run"
    )
    scan.add_argument("path", help="directory or Python file to scan")
    scan.add_argument("--out", default=None, help="write the report to this file")
    scan.add_argument("--root", default=None, help="repository root (default: PATH)")
    scan.add_argument("--label", default=None, help="root label recorded in the report")
    scan.add_argument("--commit", default=None, help="git commit recorded in the report")
    scan.add_argument("--exclude", action="append", default=[], help="glob to exclude")
    scan.add_argument("--max-files", type=int, default=4000)
    scan.add_argument("--max-file-bytes", type=int, default=512 * 1024)
    scan.set_defaults(func=_cmd_prompts_scan)

    importer = prompts_sub.add_parser(
        "import", help="upload a discovery report and print the import plan"
    )
    importer.add_argument("report", help="discovery report JSON")
    importer.add_argument("--agent-id", required=True)
    importer.add_argument("--apply", action="store_true", help="apply the plan (write key)")
    importer.add_argument("--mode", choices=("publish", "draft"), default="publish")
    importer.add_argument("--declare-slots", action="store_true")
    importer.add_argument("--slots-revision", type=int, default=None)
    importer.add_argument("--idempotency-key", default=None)
    importer.set_defaults(func=_cmd_prompts_import)

    render = prompts_sub.add_parser("render", help="render a prompt file or a prompt reference")
    render.add_argument("target", help="prompt file path, or a prm_x:N reference")
    render.add_argument("--vars", default=None, help="JSON file with the variables")
    render.set_defaults(func=_cmd_prompts_render)

    digest = prompts_sub.add_parser("digest", help="print the content digest of a prompt file")
    digest.add_argument("file")
    digest.set_defaults(func=_cmd_prompts_digest)

    verify = prompts_sub.add_parser("bundle-verify", help="verify an offline prompt bundle")
    verify.add_argument("bundle")
    verify.add_argument("--workspace", required=True)
    verify.add_argument("--agent", required=True)
    anchor = verify.add_mutually_exclusive_group(required=True)
    anchor.add_argument("--trust-key", action="append", default=None, help="trusted key PEM")
    anchor.add_argument("--expect-bundle-digest", default=None)
    verify.set_defaults(func=_cmd_prompts_bundle_verify)


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

    _add_prompts_parser(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
