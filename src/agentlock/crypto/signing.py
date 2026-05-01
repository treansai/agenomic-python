"""ed25519 signing and verifying keys with file-mode safety."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from agentlock.crypto.hashing import blake3_hex
from agentlock.exceptions import CryptoError

logger = logging.getLogger("agentlock.crypto.signing")


def _derive_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return blake3_hex(raw)[:16]


class SigningKey:
    """Wrapper around an ed25519 private key with file-mode warnings.

    Example:
        >>> k = SigningKey.generate()
        >>> sig = k.sign(b"msg")
        >>> len(sig)
        64
    """

    def __init__(self, private_key: Ed25519PrivateKey, key_id: str) -> None:
        self._key = private_key
        self.key_id = key_id

    @classmethod
    def from_pem_file(
        cls, path: Path, key_id: Optional[str] = None
    ) -> SigningKey:
        """Load a PKCS#8 PEM private key from disk.

        Warns if file mode is not 0600 on POSIX systems.
        """
        path = Path(path)
        if os.name == "posix":
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                logger.warning(
                    "signing key %s has insecure file mode %o (expected 0600)",
                    path,
                    mode,
                )
        pem = path.read_bytes()
        try:
            key = serialization.load_pem_private_key(pem, password=None)
        except Exception as e:
            raise CryptoError(f"failed to load PEM at {path}: {e}") from e
        if not isinstance(key, Ed25519PrivateKey):
            raise CryptoError(f"key at {path} is not an ed25519 private key")
        kid = key_id or _derive_key_id(key.public_key())
        return cls(key, kid)

    @classmethod
    def generate(cls, key_id: Optional[str] = None) -> SigningKey:
        """Generate a fresh ed25519 signing key.

        Example:
            >>> SigningKey.generate().key_id != ""
            True
        """
        key = Ed25519PrivateKey.generate()
        kid = key_id or _derive_key_id(key.public_key())
        return cls(key, kid)

    def sign(self, message: bytes) -> bytes:
        """Sign `message`. Returns 64-byte ed25519 signature."""
        return self._key.sign(message)

    def public_key(self) -> Ed25519PublicKey:
        """Return the raw cryptography public key object."""
        return self._key.public_key()

    def verifying_key(self) -> VerifyingKey:
        """Return a VerifyingKey wrapper around our public half."""
        return VerifyingKey(self._key.public_key(), self.key_id)

    def public_pem(self) -> str:
        """Return the SPKI PEM-encoded public key as a str."""
        pem = self._key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("ascii")

    def write_pem_file(self, path: Path) -> None:
        """Write PKCS#8 PEM with mode 0600 on POSIX."""
        pem = self._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        path = Path(path)
        path.write_bytes(pem)
        if os.name == "posix":
            os.chmod(path, 0o600)

    def write_public_pem_file(self, path: Path) -> None:
        """Write SubjectPublicKeyInfo PEM (world-readable)."""
        Path(path).write_text(self.public_pem(), encoding="ascii")


class VerifyingKey:
    """Wrapper around an ed25519 public key.

    Example:
        >>> sk = SigningKey.generate()
        >>> vk = sk.verifying_key()
        >>> vk.verify(sk.sign(b"hi"), b"hi")
        True
    """

    def __init__(self, public_key: Ed25519PublicKey, key_id: str) -> None:
        self._key = public_key
        self.key_id = key_id

    @classmethod
    def from_pem(cls, pem: str, key_id: Optional[str] = None) -> VerifyingKey:
        """Load a SubjectPublicKeyInfo PEM."""
        try:
            key = serialization.load_pem_public_key(pem.encode("ascii"))
        except Exception as e:
            raise CryptoError(f"failed to load public PEM: {e}") from e
        if not isinstance(key, Ed25519PublicKey):
            raise CryptoError("not an ed25519 public key")
        kid = key_id or _derive_key_id(key)
        return cls(key, kid)

    @classmethod
    def from_pem_file(
        cls, path: Path, key_id: Optional[str] = None
    ) -> VerifyingKey:
        return cls.from_pem(Path(path).read_text(encoding="ascii"), key_id)

    def verify(self, signature: bytes, message: bytes) -> bool:
        """Return True if `signature` is a valid ed25519 signature over `message`."""
        try:
            self._key.verify(signature, message)
        except InvalidSignature:
            return False
        return True
