"""Full offline flow: traced runs -> ATEP store -> signed release attestation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from agentlock.atep.store import AtepStore
from agentlock.crypto.hashing import LEAF_DOMAIN, hash_with_domain
from agentlock.crypto.signing import SigningKey
from agentlock.exporters.atep_local import AtepLocalExporter
from agentlock.trace.decorator import trace_agent_run
from agentlock.types.attestation import ReleaseAttestation


def _bundle_hash(directory: Path) -> str:
    """Crude bundle hash: BLAKE3 over sorted (relpath, sha256(file)) pairs."""
    parts: list[bytes] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            rel = p.relative_to(directory).as_posix()
            parts.append(f"{rel}:{digest}".encode())
    return hash_with_domain(LEAF_DOMAIN, *parts).hex()


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlock-release-"))
    bundle_dir = workdir / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "system_prompt.md").write_text("Be helpful.\n")
    (bundle_dir / "tools.json").write_text("[]\n")

    bundle_hash = _bundle_hash(bundle_dir)

    sk = SigningKey.generate()
    store = AtepStore.open_or_init(workdir / "store", "agent://acme/release-demo")
    exporter = AtepLocalExporter(store, sk)

    @trace_agent_run(agent_id="agent://acme/release-demo", exporter=exporter)
    def handle(q: str) -> dict[str, str]:
        return {"answer": q.upper()}

    handle("hi")
    handle("there")

    atep_root = store.compute_root_hash().hex()

    attestation = ReleaseAttestation(
        agent_id="agent://acme/release-demo",
        release_id="rel-2026-01-01",
        bundle_hash=bundle_hash,
        atep_root_hash=atep_root,
        signer_key_id=sk.key_id,
        signature_hex="",  # filled below
        notes="example offline signed release",
    )
    payload = attestation.model_dump_json(exclude={"signature_hex"}).encode()
    sig = sk.sign(payload).hex()
    attestation = attestation.model_copy(update={"signature_hex": sig})
    print(json.dumps(json.loads(attestation.model_dump_json()), indent=2))


if __name__ == "__main__":
    main()
