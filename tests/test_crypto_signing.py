"""Tests for ed25519 signing/verifying."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from agenomic.crypto.signing import SigningKey, VerifyingKey
from agenomic.exceptions import CryptoError


def test_sign_verify_roundtrip() -> None:
    sk = SigningKey.generate()
    sig = sk.sign(b"hello")
    assert len(sig) == 64
    assert sk.verifying_key().verify(sig, b"hello")


def test_tampered_message_fails() -> None:
    sk = SigningKey.generate()
    sig = sk.sign(b"hello")
    assert not sk.verifying_key().verify(sig, b"hellp")


def test_wrong_key_fails() -> None:
    sk1 = SigningKey.generate()
    sk2 = SigningKey.generate()
    sig = sk1.sign(b"x")
    assert not sk2.verifying_key().verify(sig, b"x")


def test_pem_roundtrip(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    pem_path = tmp_path / "k.pem"
    sk.write_pem_file(pem_path)
    sk2 = SigningKey.from_pem_file(pem_path)
    assert sk.key_id == sk2.key_id
    sig = sk2.sign(b"x")
    assert sk.verifying_key().verify(sig, b"x")


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only file modes")
def test_pem_file_mode_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sk = SigningKey.generate()
    pem_path = tmp_path / "loose.pem"
    sk.write_pem_file(pem_path)
    os.chmod(pem_path, 0o644)
    with caplog.at_level(logging.WARNING, logger="agenomic.crypto.signing"):
        SigningKey.from_pem_file(pem_path)
    assert any("insecure file mode" in r.message for r in caplog.records)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only file modes")
def test_write_pem_sets_0600(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    pem_path = tmp_path / "tight.pem"
    sk.write_pem_file(pem_path)
    assert pem_path.stat().st_mode & 0o777 == 0o600


def test_verifying_key_from_pem() -> None:
    sk = SigningKey.generate()
    vk = VerifyingKey.from_pem(sk.public_pem())
    assert vk.key_id == sk.key_id
    assert vk.verify(sk.sign(b"hi"), b"hi")


def test_load_non_ed25519_raises(tmp_path: Path) -> None:
    pem_path = tmp_path / "garbage.pem"
    pem_path.write_text("not a real PEM")
    with pytest.raises(CryptoError):
        SigningKey.from_pem_file(pem_path)


def test_key_id_is_short_hex() -> None:
    sk = SigningKey.generate()
    assert len(sk.key_id) == 16
    int(sk.key_id, 16)  # parses as hex


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
def test_write_public_pem(tmp_path: Path) -> None:
    sk = SigningKey.generate()
    out = tmp_path / "pub.pem"
    sk.write_public_pem_file(out)
    text = out.read_text()
    assert "BEGIN PUBLIC KEY" in text
