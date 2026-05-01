"""Cross-implementation compatibility test against the golden ATEP fixture.

The golden fixture is the wire-format anchor for `agentlock-python`,
`agentlock-cli` and `agentlock-cloud`. If this test breaks, either the
wire format changed (regenerate the fixture and bump the schema version)
or one of the implementations diverged.
"""

from __future__ import annotations

from pathlib import Path

from agentlock.atep.clock import Hlc
from agentlock.atep.event import StreamId
from agentlock.atep.segment import SegmentReader
from agentlock.crypto.signing import VerifyingKey

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_atep_segments"
GOLDEN_SEGMENT = FIXTURE_DIR / "golden_v1.atep"
GOLDEN_PUB_PEM = FIXTURE_DIR / "golden_pub.pem"


def test_golden_segment_envelope_intact() -> None:
    reader = SegmentReader(GOLDEN_SEGMENT)
    assert reader.event_count == 3
    assert reader.first_hlc == Hlc(1000, 0, 0)
    assert reader.last_hlc == Hlc(1002, 0, 0)


def test_golden_segment_merkle_root() -> None:
    reader = SegmentReader(GOLDEN_SEGMENT)
    assert reader.verify_merkle_root()


def test_golden_segment_signatures_valid() -> None:
    vk = VerifyingKey.from_pem_file(GOLDEN_PUB_PEM)
    reader = SegmentReader(GOLDEN_SEGMENT)
    events = list(reader.iter_events())
    assert len(events) == 3
    for i, ev in enumerate(events):
        assert ev.verify(vk), f"event {i} failed verify"
        assert ev.header.stream == StreamId.IDENTITY
        assert ev.header.agent_id == "agent://acme/golden"
        assert ev.payload == {"i": i, "label": f"event-{i}"}


def test_golden_segment_chain_parents() -> None:
    reader = SegmentReader(GOLDEN_SEGMENT)
    events = list(reader.iter_events())
    assert events[0].header.parents == []
    assert events[1].header.parents == [events[0].causal_hash]
    assert events[2].header.parents == [events[1].causal_hash]
