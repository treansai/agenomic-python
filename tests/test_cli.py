"""Tests for agentlock-py CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import ulid

from agentlock.atep.clock import Hlc
from agentlock.atep.event import AtepEvent, EventHeader, StreamId
from agentlock.atep.segment import SegmentWriter
from agentlock.cli.__main__ import main
from agentlock.crypto.signing import SigningKey

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_atep_segments"


def test_atep_verify_golden(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "atep",
            "verify",
            str(FIXTURE_DIR / "golden_v1.atep"),
            "--public-key",
            str(FIXTURE_DIR / "golden_pub.pem"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "verified" in out


def test_atep_verify_tampered(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    seg = tmp_path / "s.atep"
    hdr = EventHeader(
        event_id=ulid.new().bytes,
        agent_id="agent://a/b",
        stream=StreamId.IDENTITY,
        stream_seq=0,
        clock=Hlc(1, 0, 0),
        event_type="t",
        payload_schema_uri="atep://t",
    )
    with SegmentWriter(seg) as w:
        w.append(AtepEvent.seal(hdr, {"x": 1}, sk))

    pub = tmp_path / "k.pem.pub"
    sk.write_public_pem_file(pub)

    data = bytearray(seg.read_bytes())
    data[100] ^= 0xFF
    seg.write_bytes(bytes(data))

    rc = main(["atep", "verify", str(seg), "--public-key", str(pub)])
    assert rc != 0


def test_atep_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["atep", "inspect", str(FIXTURE_DIR / "golden_v1.atep")])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event_count"] == 3
    assert payload["event_types"]["identity.created"] == 3


def test_keys_generate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "k.pem"
    rc = main(["keys", "generate", str(out)])
    assert rc == 0
    assert out.exists()
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600
    pub = out.with_suffix(out.suffix + ".pub")
    assert pub.exists()
    assert "key_id=" in capsys.readouterr().out


def test_keys_generate_no_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "k.pem"
    out.write_text("existing")
    rc = main(["keys", "generate", str(out)])
    assert rc != 0


def test_traces_summarize(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"agent_id": "agent://a/b", "duration_ms": 10}),
                json.dumps({"agent_id": "agent://a/b", "duration_ms": 30}),
                json.dumps({"agent_id": "agent://x/y", "error": "boom"}),
                "",
            ]
        )
    )
    rc = main(["traces", "summarize", str(p)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["envelopes"] == 3
    assert payload["errors"] == 1
    assert payload["agents"]["agent://a/b"] == 2


def test_version() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
