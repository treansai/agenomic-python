# ATEP

ATEP is the **Attested Tamper-Evident Provenance** log format. It captures
the full history of an agent — identity, capabilities, knowledge, policy,
runtime, interactions, governance — as a chain of signed events.

`agenomic-python` produces ATEP segments that are bit-for-bit compatible
with `agenomic-cli` (Rust) and `agenomic-cloud`.

## Concepts

- **Event**: one signed unit. Carries a header, a payload, a 32-byte BLAKE3
  causal hash over (header || payload || sorted parents), and an ed25519
  signature over the causal hash.
- **Segment**: a binary file (`.atep`) containing N events plus a header
  with first/last HLC, event count, and a Merkle root over causal hashes.
- **Store**: a directory layout with one `manifest.json` and one or more
  segments per stream.

## Streams

| stream         | what goes in it                                           |
| -------------- | --------------------------------------------------------- |
| `identity`     | agent creation, ownership                                 |
| `capability`   | enabled tools, permissions                                |
| `knowledge`    | system prompts, knowledge bases attached                  |
| `policy`       | policy rules, guardrails                                  |
| `runtime`      | model fingerprint, runtime config                         |
| `interaction`  | one event per agent run (the SDK's main output)           |
| `governance`   | release decisions, approvals                              |

## Python API

```python
from pathlib import Path
from agenomic.atep import AtepStore, StreamId
from agenomic.crypto import SigningKey
from agenomic.exporters import AtepLocalExporter
from agenomic.trace.decorator import trace_agent_run

sk = SigningKey.generate()
store = AtepStore.open_or_init(Path("store"), "agent://acme/demo")
exporter = AtepLocalExporter(store, sk)

@trace_agent_run(agent_id="agent://acme/demo", exporter=exporter)
def handle(q: str) -> dict:
    return {"answer": q.upper()}

handle("hello")

report = store.verify_all(sk.verifying_key())
assert report.ok and report.events_checked == 1
root = store.compute_root_hash().hex()   # deterministic store root
```

Key pieces:

- **`AtepStore.open_or_init(root, agent_id)`** — create/open the
  directory layout (`manifest.json` + `streams/*.atep`). Opening a store
  that belongs to a different agent raises `AtepError`.
- **`store.append_event(event)`** / **`store.read_stream(stream)`** /
  **`store.list_segments(stream=None)`** — low-level event I/O. v0.1
  writes one event per segment.
- **`store.verify_all(verifying_key)`** — checks every segment's CRC and
  Merkle root plus every event's causal hash and signature; returns a
  `VerificationReport(ok, segments_checked, events_checked, failures)`.
- **`store.compute_root_hash()`** — 32-byte deterministic root over all
  segment names + Merkle roots; the value that goes into a
  `ReleaseAttestation.atep_root_hash`.
- **`AtepEvent.seal(header, payload, signing_key)`** — hash + sign one
  event; **`event.verify(verifying_key)`** — recompute the causal hash
  and check the signature.
- **`SegmentWriter`** / **`SegmentReader`** — file-level access; the
  reader validates magic bytes and CRC32 on open.

Events are timestamped with a **hybrid logical clock** (`Hlc`) — 16-byte
`physical_ms / logical / node_id` — so causal order survives clock skew
across processes.

From the shell: `agenomic-py atep verify <segment> --public-key <pem>` and
`agenomic-py atep inspect <segment>` (see [CLI](cli.md)).

## Wire format

See [`src/agenomic/atep/segment.py`](../src/agenomic/atep/segment.py)
for the canonical layout. CRC32 over everything before the trailing
`PETA` magic protects against truncation. The 32-byte Merkle root over
event causal hashes protects against per-event tampering.

## Cross-implementation compat

The fixture at
[`tests/fixtures/golden_atep_segments/golden_v1.atep`](../tests/fixtures/golden_atep_segments/)
is the wire-format anchor. Any implementation that wants to claim ATEP-v1
compatibility MUST be able to read it and verify its signatures.
