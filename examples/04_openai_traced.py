"""OpenAI integration: model calls auto-recorded on TraceRecorder.

Requires `pip install agentlock[openai]` and an OPENAI_API_KEY in your env.
"""
from __future__ import annotations

import os

from agentlock.exporters.jsonl import JsonlExporter
from agentlock.integrations.openai import instrument_openai
from agentlock.trace.decorator import trace_agent_run


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping live call")
        return

    from openai import OpenAI

    oai = instrument_openai(OpenAI())

    with JsonlExporter("/tmp/openai-traces.jsonl") as exporter:

        @trace_agent_run(agent_id="agent://acme/qa", exporter=exporter)
        def answer(question: str) -> str:
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
            )
            return resp.choices[0].message.content or ""

        print(answer("what is the capital of France?"))


if __name__ == "__main__":
    main()
