#!/usr/bin/env python
"""Regenerate the golden ATEP segment fixture.

Produces ``tests/fixtures/golden_atep_segments/golden_v1.atep`` and the
matching ``golden_pub.pem``, with exactly the content the golden tests
assert: three signed identity events for ``agent://acme/golden`` at
HLC(1000..1002, 0, 0), each chained on the previous event's causal hash.

The ed25519 signing key is generated fresh and intentionally discarded;
only the public half is written. Re-running this script therefore yields
a new valid fixture/key pair — commit both files together.
"""

from __future__ import annotations

from pathlib import Path

import ulid

from agenomic.atep.clock import Hlc
from agenomic.atep.event import AtepEvent, EventHeader, StreamId
from agenomic.atep.segment import SegmentWriter
from agenomic.crypto.signing import SigningKey

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden_atep_segments"
GOLDEN_SEGMENT = FIXTURE_DIR / "golden_v1.atep"
GOLDEN_PUB_PEM = FIXTURE_DIR / "golden_pub.pem"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    sk = SigningKey.generate()

    parents: list[bytes] = []
    with SegmentWriter(GOLDEN_SEGMENT) as writer:
        for i in range(3):
            header = EventHeader(
                event_id=ulid.new().bytes,
                agent_id="agent://acme/golden",
                stream=StreamId.IDENTITY,
                stream_seq=i,
                clock=Hlc(1000 + i, 0, 0),
                parents=parents,
                event_type="identity.created",
                payload_schema_uri="atep://schemas/v1/identity",
            )
            event = AtepEvent.seal(header, {"i": i, "label": f"event-{i}"}, sk)
            writer.append(event)
            parents = [event.causal_hash]

    sk.write_public_pem_file(GOLDEN_PUB_PEM)
    print(f"wrote {GOLDEN_SEGMENT}")
    print(f"wrote {GOLDEN_PUB_PEM} (key_id={sk.key_id})")


if __name__ == "__main__":
    main()
