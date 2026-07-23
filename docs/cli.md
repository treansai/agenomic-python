# `agenomic-py` CLI

The package installs a small utility CLI named `agenomic-py` for ATEP and
trace inspection. For full CLI features (bundle, sign, replay), use
`agenomic-cli` (Rust).

```bash
agenomic-py --version
```

## `atep verify`

Verify an ATEP segment's Merkle root and every event signature:

```bash
agenomic-py atep verify path/to/segment.atep --public-key signer.pem.pub
```

| exit code | meaning                                    |
| --------- | ------------------------------------------ |
| `0`       | all events verified (`ok: N events verified`) |
| `2`       | segment could not be read (format error)   |
| `3`       | Merkle root mismatch (tampering/truncation) |
| `4`       | one or more event signatures failed        |

## `atep inspect`

Print a JSON summary of a segment — version, event count, first/last HLC
clock, Merkle root, and per-type event counts:

```bash
agenomic-py atep inspect path/to/segment.atep
```

```json
{
  "version": 1,
  "event_count": 3,
  "first_hlc": {"physical_ms": 1720000000000, "logical": 0, "node_id": 0},
  "last_hlc":  {"physical_ms": 1720000000042, "logical": 1, "node_id": 0},
  "merkle_root": "9f3a…",
  "event_types": {"interaction.run_completed": 3}
}
```

## `traces summarize`

Summarize a JSONL file of `TraceEnvelope` lines:

```bash
agenomic-py traces summarize traces.jsonl
```

Reports envelope count, error count, per-agent counts, and average
`duration_ms`. Blank or invalid lines are skipped. Exit `2` if the file
does not exist.

## `keys generate`

Generate a new ed25519 signing keypair as PEM files:

```bash
agenomic-py keys generate signer.pem          # writes signer.pem + signer.pem.pub
agenomic-py keys generate signer.pem --force  # overwrite an existing file
```

Prints the private path, public path, and the key id. Exit `2` if the
output file exists and `--force` was not given.

Keep the private PEM at file mode `0600` — the SDK warns when a looser
mode is detected on load.
