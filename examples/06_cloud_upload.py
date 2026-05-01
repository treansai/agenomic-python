"""Upload a JSONL file of envelopes to AgentLock Cloud."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from agentlock.client.client import AgentLockClient
from agentlock.types.envelope import TraceEnvelope


async def upload(path: Path, endpoint: str, api_key: str) -> None:
    envs: list[TraceEnvelope] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            envs.append(TraceEnvelope.model_validate(json.loads(line)))
    client = AgentLockClient(endpoint, api_key)
    try:
        result = await client.upload_traces(envs)
        print(json.dumps(result, indent=2))
    finally:
        await client.aclose()


def main() -> int:
    endpoint = os.environ.get("AGENTLOCK_ENDPOINT")
    api_key = os.environ.get("AGENTLOCK_API_KEY")
    if not endpoint or not api_key:
        print("AGENTLOCK_ENDPOINT and AGENTLOCK_API_KEY must be set", file=sys.stderr)
        return 1
    if len(sys.argv) < 2:
        print("usage: 06_cloud_upload.py <traces.jsonl>", file=sys.stderr)
        return 1
    asyncio.run(upload(Path(sys.argv[1]), endpoint, api_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
