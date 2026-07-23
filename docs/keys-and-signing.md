# Keys, hashing & signing

`agenomic.crypto` provides the primitives every attested artifact is
built on: ed25519 signatures, BLAKE3 hashing with domain separation, and
canonical CBOR encoding. All of it is byte-for-byte compatible with
`agenomic-cli` (Rust) and `agenomic-cloud`.

## Signing keys (ed25519)

```python
from pathlib import Path
from agenomic.crypto import SigningKey, VerifyingKey

sk = SigningKey.generate()                     # fresh keypair
sk.write_pem_file(Path("signer.pem"))          # PKCS#8 PEM, chmod 0600
sk.write_public_pem_file(Path("signer.pem.pub"))  # SPKI PEM

sk = SigningKey.from_pem_file(Path("signer.pem"))  # warns if mode > 0600
vk = VerifyingKey.from_pem_file(Path("signer.pem.pub"))

sig = sk.sign(b"message")        # 64 bytes
vk.verify(sig, b"message")       # True / False — never raises on bad sig
```

- `key_id` is the first 16 hex chars of BLAKE3 over the raw 32-byte
  public key; derived automatically, or pass your own.
- Private PEMs are written with mode `0600`; loading a key with looser
  permissions logs a warning.
- Keys are always unencrypted PKCS#8 — protect them with filesystem
  permissions or a secrets manager.
- `agenomic-py keys generate signer.pem` does the generate+write dance
  from the shell (see [CLI](cli.md)).

## Hashing (BLAKE3 + domains)

```python
from agenomic.crypto import (
    blake3_hex, blake3_bytes, hash_with_domain,
    LEAF_DOMAIN, NODE_DOMAIN, ATEP_DOMAIN, ATTESTATION_DOMAIN,
)

blake3_hex(b"hello")                       # '<64 hex chars>'
hash_with_domain(ATEP_DOMAIN, b"payload")  # 32 bytes
```

Domain separators (`AGENTLOCK-LEAF-v1\0`, `AGENTLOCK-NODE-v1\0`,
`ATEP-v1\0`, `AGENTLOCK-ATTESTATION-v1\0`) MUST match the other
implementations byte-for-byte — never change them.

## Canonical CBOR

```python
from agenomic.crypto import canonical_cbor, canonical_cbor_decode

canonical_cbor({"b": 1, "a": 2}) == canonical_cbor({"a": 2, "b": 1})  # True
```

RFC 8949 §4.2 canonical form (deterministic key order, datetimes as
timestamps) — the serialization under every ATEP hash and frame.

## Release attestations

`ReleaseAttestation` links a release to a bundle hash and an ATEP root
hash. The signing convention: sign the model's JSON with `signature_hex`
excluded, then store the hex signature back in the field:

```python
from agenomic.types import ReleaseAttestation

att = ReleaseAttestation(
    agent_id="agent://acme/demo",
    release_id="rel-2026-01-01",
    bundle_hash=bundle_hash,
    atep_root_hash=store.compute_root_hash().hex(),
    signer_key_id=sk.key_id,
    signature_hex="",
)
payload = att.model_dump_json(exclude={"signature_hex"}).encode()
att = att.model_copy(update={"signature_hex": sk.sign(payload).hex()})
```

`examples/07_offline_signed_release.py` is the runnable end-to-end
version — traced runs → ATEP store → signed attestation, fully offline.

## Which hash where

| surface                          | algorithm                              |
| -------------------------------- | -------------------------------------- |
| ATEP causal hashes, Merkle roots | BLAKE3 (+ domain separation)           |
| Canonical v0.3 event/content hashes | BLAKE3 over canonical JSON (`blake3:` prefix) |
| Redaction `HASH` mode            | BLAKE3 over sorted-key JSON, 16 hex    |
| Hugging Face lockfile entries    | SHA-256 over canonical JSON            |
