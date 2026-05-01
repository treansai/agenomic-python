"""Tests for the JSONL exporter."""
from __future__ import annotations

import json
from pathlib import Path

from agentlock.exporters.jsonl import JsonlExporter
from agentlock.types.envelope import TraceEnvelope
from agentlock.types.trace import TraceInput, TraceOutput


def _env(i: int) -> TraceEnvelope:
    return TraceEnvelope(
        trace_id=f"t{i}",
        run_id=f"r{i}",
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": i}),
        final_output=TraceOutput(payload_inline={"a": i}),
    )


def test_writes_one_line_per_envelope(tmp_path: Path) -> None:
    out = tmp_path / "traces.jsonl"
    with JsonlExporter(out) as exp:
        for i in range(5):
            exp.export(_env(i))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["trace_id"] == f"t{i}"


def test_round_trip_envelope(tmp_path: Path) -> None:
    out = tmp_path / "rt.jsonl"
    with JsonlExporter(out) as exp:
        exp.export(_env(7))
    line = out.read_text(encoding="utf-8").strip()
    restored = TraceEnvelope.model_validate_json(line)
    assert restored.trace_id == "t7"


def test_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "traces.jsonl"
    with JsonlExporter(out) as exp:
        exp.export(_env(0))
    assert out.exists()
