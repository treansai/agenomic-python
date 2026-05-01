"""ATEP signed event."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentlock.atep.clock import Hlc
from agentlock.crypto.canonical import canonical_cbor
from agentlock.crypto.hashing import ATEP_DOMAIN, hash_with_domain
from agentlock.crypto.signing import SigningKey, VerifyingKey


class StreamId(str, Enum):
    """Logical stream within an agent's ATEP history."""

    IDENTITY = "identity"
    CAPABILITY = "capability"
    KNOWLEDGE = "knowledge"
    POLICY = "policy"
    RUNTIME = "runtime"
    INTERACTION = "interaction"
    GOVERNANCE = "governance"


class EventHeader(BaseModel):
    """ATEP event header."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    event_id: bytes = Field(min_length=16, max_length=16)
    agent_id: str
    stream: StreamId
    stream_seq: int
    clock: Hlc
    parents: list[bytes] = Field(default_factory=list)
    event_type: str
    payload_schema_uri: str


class EventAttestation(BaseModel):
    """Signature over the event's causal_hash."""

    signer_key_id: str
    signature: bytes = Field(min_length=64, max_length=64)
    algo: str = "ed25519"


class AtepEvent(BaseModel):
    """A signed ATEP event with a BLAKE3 causal hash and ed25519 signature.

    Example:
        >>> from agentlock.crypto.signing import SigningKey
        >>> import ulid
        >>> sk = SigningKey.generate()
        >>> hdr = EventHeader(
        ...     event_id=ulid.new().bytes, agent_id="agent://a/b",
        ...     stream=StreamId.IDENTITY, stream_seq=0,
        ...     clock=Hlc(1, 0, 0), event_type="identity.created",
        ...     payload_schema_uri="atep://schemas/v1/identity",
        ... )
        >>> ev = AtepEvent.seal(hdr, {"name": "demo"}, sk)
        >>> ev.verify(sk.verifying_key())
        True
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    header: EventHeader
    payload: dict[str, Any]
    causal_hash: bytes = Field(min_length=32, max_length=32)
    attestation: EventAttestation

    @staticmethod
    def compute_causal_hash(header: EventHeader, payload: dict[str, Any]) -> bytes:
        """``BLAKE3(ATEP-v1\\0 || len(body) || body || len(parents) || sorted(parents))``.

        Body is canonical CBOR of ``{"header": header_dict_without_parents,
        "payload": payload}``. Parent hashes are sorted lexicographically and
        concatenated outside the body so reordering parents in memory does not
        change the hash.
        """
        header_dict: dict[str, Any] = header.model_dump(mode="python")
        # Convert HLC dataclass to its dict form for canonical CBOR
        clock = header_dict.get("clock")
        if isinstance(clock, Hlc):
            header_dict["clock"] = {
                "physical_ms": clock.physical_ms,
                "logical": clock.logical,
                "node_id": clock.node_id,
            }
        # Stream enum → its value
        stream = header_dict.get("stream")
        if isinstance(stream, StreamId):
            header_dict["stream"] = stream.value
        parents = header_dict.pop("parents", []) or []
        body = canonical_cbor({"header": header_dict, "payload": payload})
        sorted_parents = sorted(parents)
        parents_concat = b"".join(sorted_parents)
        return hash_with_domain(
            ATEP_DOMAIN,
            len(body).to_bytes(8, "little"),
            body,
            len(sorted_parents).to_bytes(4, "little"),
            parents_concat,
        )

    @classmethod
    def seal(
        cls,
        header: EventHeader,
        payload: dict[str, Any],
        signing_key: SigningKey,
    ) -> AtepEvent:
        """Build a signed ATEP event."""
        causal_hash = cls.compute_causal_hash(header, payload)
        signature = signing_key.sign(causal_hash)
        return cls(
            header=header,
            payload=payload,
            causal_hash=causal_hash,
            attestation=EventAttestation(
                signer_key_id=signing_key.key_id,
                signature=signature,
            ),
        )

    def verify(self, verifying_key: VerifyingKey) -> bool:
        """Verify causal_hash recomputes AND signature is valid."""
        recomputed = self.compute_causal_hash(self.header, self.payload)
        if recomputed != self.causal_hash:
            return False
        return verifying_key.verify(self.attestation.signature, self.causal_hash)
