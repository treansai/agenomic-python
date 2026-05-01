"""Minimal trace: build an envelope by hand, write to stdout."""
from __future__ import annotations

import ulid

from agentlock.types.envelope import TraceEnvelope
from agentlock.types.trace import TraceInput, TraceOutput


def main() -> None:
    env = TraceEnvelope(
        trace_id=ulid.new().str,
        run_id=ulid.new().str,
        agent_id="agent://acme/demo",
        input=TraceInput(payload_inline={"q": "hello"}),
        final_output=TraceOutput(payload_inline={"a": "world"}),
    )
    print(env.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
