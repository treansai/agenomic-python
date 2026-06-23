"""Canonical Trace SDK — redaction runs before export; secrets never leak."""

from __future__ import annotations

import json

from agenomic.canonical import start_run
from agenomic.redaction.engine import RedactionEngine
from agenomic.redaction.rules import RedactionMode, RedactionRule

SECRET = "SUPERSECRET-TOKEN-abc123"


def test_sensitive_payload_never_appears_in_cleartext() -> None:
    engine = RedactionEngine(
        [
            RedactionRule(path="**.api_key", mode=RedactionMode.MASK),
            RedactionRule(path="**.prompt", mode=RedactionMode.HASH),
        ]
    )
    run = start_run(
        "agent://acme/x",
        redaction=engine,
        input_payload={"api_key": SECRET, "q": "hello"},
    )
    run.log_llm(prompt=f"use the key {SECRET} now", response="done")
    run.log_tool_call(tool="send_email", arguments={"api_key": SECRET}, result="sent")
    trace = run.complete_run(output={"api_key": SECRET})

    blob = json.dumps(trace)
    # The explicit negative assertion: the raw secret is absent everywhere.
    assert SECRET not in blob
    # …and the redaction markers are present where it was.
    assert "***" in blob  # masked api_key
    assert "hash:" in blob  # hashed prompt


def test_no_redaction_keeps_payloads_but_still_hashes() -> None:
    run = start_run("agent://acme/x", input_payload={"q": "plain"})
    run.log_tool_call(tool="echo", arguments={"v": "plain"}, result="plain")
    trace = run.complete_run(output={"ok": True})
    # Without redaction rules the payloads pass through, but every event still
    # carries a content-addressed hash.
    assert all(e["payload_hash"].startswith("blake3:") for e in trace["events"])
