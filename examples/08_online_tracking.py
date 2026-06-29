"""Online tracking: instrument a production agent run.

With no ``base_url`` the session buffers spec-shaped events locally; point the
client at Agenomic Cloud to stream them for real-time drift / loop / intent
detection. Either way the wire format is the v0.3 ``tracking-event`` shape.

    python examples/08_online_tracking.py
"""

from __future__ import annotations

from agenomic import Client


def main() -> None:
    # base_url=... switches to cloud mode (no silent insecure fallback).
    client = Client()

    session = client.tracking.start(
        agent="agent://treans/claims-agent",
        release_id="release_123",
        environment="production",
    )

    with session.step("classify_claim"):
        session.model_call(provider="openai", model="gpt-4o", input_hash="blake3:" + "0" * 64)
        session.tool_call(
            tool="claims_db.lookup",
            input_hash="blake3:" + "1" * 64,
            output_hash="blake3:" + "2" * 64,
        )
        session.intent("verify_claim_validity")
        session.memory_write(schema_version="1.0.0")

    session.stop()

    if client.is_cloud:
        print(session.report())
    else:
        # Local mode: export events for `agenomic track report` to analyze.
        print(session.to_jsonl(), end="")


if __name__ == "__main__":
    main()
