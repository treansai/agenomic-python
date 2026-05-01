# ATEP

ATEP is the **Attested Tamper-Evident Provenance** log format. It captures
the full history of an agent — identity, capabilities, knowledge, policy,
runtime, interactions, governance — as a chain of signed events.

`agentlock-python` produces ATEP segments that are bit-for-bit compatible
with `agentlock-cli` (Rust) and `agentlock-cloud`.

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

## Wire format

See [`src/agentlock/atep/segment.py`](../src/agentlock/atep/segment.py)
for the canonical layout. CRC32 over everything before the trailing
`PETA` magic protects against truncation. The 32-byte Merkle root over
event causal hashes protects against per-event tampering.

## Cross-implementation compat

The fixture at
[`tests/fixtures/golden_atep_segments/golden_v1.atep`](../tests/fixtures/golden_atep_segments/)
is the wire-format anchor. Any implementation that wants to claim ATEP-v1
compatibility MUST be able to read it and verify its signatures.
