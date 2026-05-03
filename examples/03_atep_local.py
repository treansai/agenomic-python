"""ATEP local: produce a signed event log fully offline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentlock.atep.store import AtepStore
from agentlock.crypto.signing import SigningKey
from agentlock.exporters.atep_local import AtepLocalExporter
from agentlock.trace.decorator import trace_agent_run


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentlock-atep-"))
    key_path = workdir / "key.pem"
    pub_path = workdir / "key.pem.pub"
    sk = SigningKey.generate()
    sk.write_pem_file(key_path)
    sk.write_public_pem_file(pub_path)

    store = AtepStore.open_or_init(workdir / "store", "agent://acme/demo")
    exporter = AtepLocalExporter(store, sk)

    @trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
    def handle(q: str) -> dict[str, str]:
        return {"answer": q.upper()}

    handle("hello")
    handle("again")

    report = store.verify_all(sk.verifying_key())
    print(
        f"store={store.root}\nsegments={report.segments_checked}"
        f"\nevents={report.events_checked}\nok={report.ok}"
    )
    print("root_hash:", store.compute_root_hash().hex())


if __name__ == "__main__":
    main()
