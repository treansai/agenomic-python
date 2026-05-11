"""Decorator + JSONL exporter: trace each call to a function."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agenomic.exporters.jsonl import JsonlExporter
from agenomic.trace.decorator import trace_agent_run


def main() -> None:
    out = Path(tempfile.gettempdir()) / "agenomic-traces.jsonl"
    out.unlink(missing_ok=True)
    with JsonlExporter(out) as exporter:

        @trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
        def handle(query: str) -> dict[str, str]:
            return {"answer": query.upper()}

        handle("hello")
        handle("world")

    print(f"wrote {out}")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
