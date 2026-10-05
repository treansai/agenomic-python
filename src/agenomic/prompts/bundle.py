from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast

import blake3
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from agenomic.crypto.signing import SigningKey
from agenomic.prompts.digest import (
    MANIFEST_SCHEMA,
    artifact_set_digest,
    canonical_json_v1,
    closure_gaps,
    content_digest,
    ensure_ajs,
    manifest_digest,
    sorted_keys,
)
from agenomic.prompts.errors import AjsError, binding_error, integrity_error
from agenomic.prompts.models import ManagedPromptVersion, PromptVersionRecord, ResolvedClosure
from agenomic.prompts.pinned import PinnedPromptSet

BUNDLE_SCHEMA = "agenomic.prompt_bundle/v1"
_REQUIRED_MEMBERS: tuple[tuple[str, type], ...] = (
    ("workspace_id", str),
    ("agent_id", str),
    ("source", dict),
    ("release", dict),
    ("prompt_manifest_digest", str),
    ("manifest", dict),
    ("children", dict),
    ("prompts", dict),
    ("prompt_bundle_digest", str),
)


def _public_key(pem: Union[str, bytes]) -> Ed25519PublicKey:
    data = pem.encode("ascii") if isinstance(pem, str) else pem
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("bundle trust keys must be ed25519 public keys")
    return key


@dataclass(frozen=True)
class BundleTrust:
    keys: Mapping[str, Ed25519PublicKey]

    @classmethod
    def from_pems(cls, pems: Mapping[str, Union[str, bytes]]) -> BundleTrust:
        return cls({key_id: _public_key(pem) for key_id, pem in pems.items()})

    @classmethod
    def from_pem_files(cls, *paths: os.PathLike[str]) -> BundleTrust:
        return cls({Path(path).stem: _public_key(Path(path).read_bytes()) for path in paths})


def signing_digest(document: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in document.items() if key != "signature"}
    return blake3.blake3(canonical_json_v1(unsigned).encode("utf-8")).digest()


def build_bundle(document: Mapping[str, Any], *, signer: SigningKey) -> dict[str, Any]:
    signed = {key: value for key, value in document.items() if key != "signature"}
    signed["issuer"] = {"key_id": signer.key_id, "algorithm": "ed25519"}
    signature = signer.sign(signing_digest(signed))
    signed["signature"] = {
        "algorithm": "ed25519",
        "value": base64.b64encode(signature).decode("ascii"),
        "public_key_pem": signer.public_pem(),
    }
    return signed


def _incomplete(message: str, **details: Any) -> Exception:
    return integrity_error("bundle_incomplete", message, **details)


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise _incomplete("expires_at is not a timestamp", reason="invalid_field_type")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _incomplete("expires_at is not a timestamp", reason="invalid_field_type") from error
    if parsed.tzinfo is None:
        raise _incomplete("expires_at has no time zone", reason="invalid_field_type")
    return parsed


def _read(source: Union[str, os.PathLike[str], Mapping[str, Any]]) -> Any:
    if isinstance(source, Mapping):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _shape(raw: Any) -> dict[str, Any]:
    try:
        document = ensure_ajs(raw)
    except AjsError as error:
        raise _incomplete(
            "the bundle is outside the JSON subset",
            reason=error.reason,
            value_path=error.value_path,
        ) from error
    if not isinstance(document, dict):
        raise _incomplete("the bundle is not an object", reason="invalid_field_type")
    if document.get("schema") != BUNDLE_SCHEMA:
        raise _incomplete("unsupported bundle schema", reason="unsupported_schema")
    for member, kind in _REQUIRED_MEMBERS:
        if member not in document:
            raise _incomplete(f"missing member {member}", reason="missing_field", path="/" + member)
        if not isinstance(document[member], kind):
            raise _incomplete(
                f"member {member} has the wrong type",
                reason="invalid_field_type",
                path="/" + member,
            )
    return cast(dict[str, Any], document)


def _verify_signature(document: Mapping[str, Any], trust: BundleTrust) -> None:
    issuer = document.get("issuer")
    signature = document.get("signature")
    if (
        not isinstance(issuer, dict)
        or not isinstance(signature, dict)
        or issuer.get("algorithm") != "ed25519"
        or signature.get("algorithm") != "ed25519"
        or not isinstance(signature.get("value"), str)
    ):
        raise integrity_error("bundle_signature_invalid", "the bundle signature block is malformed")
    key = trust.keys.get(str(issuer.get("key_id")))
    if key is None:
        raise integrity_error(
            "bundle_untrusted_key",
            "the bundle issuer key is not in the trust store",
            key_id=issuer.get("key_id"),
        )
    try:
        raw = base64.b64decode(signature["value"], validate=True)
        key.verify(raw, signing_digest(document))
    except (binascii.Error, InvalidSignature) as error:
        raise integrity_error(
            "bundle_signature_invalid", "the bundle signature does not verify"
        ) from error


def _check_prompts(document: Mapping[str, Any]) -> None:
    prompts = document["prompts"]
    for ref in sorted_keys(prompts):
        entry = prompts[ref]
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), dict):
            raise _incomplete(f"prompt entry {ref} is malformed", reason="invalid_field_type")
        actual = content_digest(entry["content"])
        if actual != entry.get("content_digest"):
            raise integrity_error(
                "prompt_digest_mismatch",
                f"content digest of {ref} does not match",
                ref=ref,
                expected=entry.get("content_digest"),
                actual=actual,
            )


def _artifact_set(document: Mapping[str, Any], expected: Optional[str]) -> str:
    actual = artifact_set_digest(document)
    for pinned in (document["prompt_bundle_digest"], expected):
        if pinned is not None and actual != pinned:
            raise integrity_error(
                "prompt_digest_mismatch",
                "the prompt artifact set digest does not match",
                document="artifact_set",
                expected=pinned,
                actual=actual,
            )
    return actual


def _manifest_ok(manifest: Any) -> bool:
    return (
        isinstance(manifest, dict)
        and manifest.get("schema") == MANIFEST_SCHEMA
        and isinstance(manifest.get("slots"), dict)
        and isinstance(manifest.get("children"), dict)
    )


def _check_manifests(document: Mapping[str, Any], expected: Optional[str]) -> None:
    manifests: list[tuple[Optional[str], Any, Any]] = [
        (None, document["manifest"], document["prompt_manifest_digest"])
    ]
    for child_id in sorted_keys(document["children"]):
        child = document["children"][child_id]
        if not isinstance(child, dict):
            raise _incomplete(f"child {child_id} is malformed", reason="invalid_field_type")
        manifests.append((child_id, child.get("manifest"), child.get("prompt_manifest_digest")))
    for owner, manifest, pinned in manifests:
        if not _manifest_ok(manifest):
            raise _incomplete("a manifest is malformed", reason="invalid_field_type")
        actual = manifest_digest(manifest)
        if actual != pinned:
            raise integrity_error(
                "manifest_digest_mismatch",
                "a manifest digest does not match",
                child_agent_id=owner,
                expected=pinned,
                actual=actual,
            )
    if expected is not None and expected != document["prompt_manifest_digest"]:
        raise integrity_error(
            "manifest_digest_mismatch",
            "the manifest digest differs from the expected digest",
            expected=expected,
            actual=document["prompt_manifest_digest"],
        )


def _check_closure(document: Mapping[str, Any]) -> None:
    prompts: dict[str, Any] = document["prompts"]
    children: dict[str, Any] = document["children"]
    wanted: set[str] = set()
    missing: set[str] = set()
    reached: set[str] = set()
    pending: list[Mapping[str, Any]] = [document["manifest"]]
    while pending:
        manifest = pending.pop()
        refs, gaps = closure_gaps(manifest, prompts)
        wanted |= refs
        missing |= gaps
        for child_id, pin in manifest["children"].items():
            if child_id in reached:
                continue
            reached.add(child_id)
            child = children.get(child_id)
            if (
                not isinstance(pin, dict)
                or child is None
                or child.get("release_id") != pin.get("release_id")
                or child.get("genome_version") != pin.get("genome_version")
                or child["manifest"].get("agent_id") != child_id
            ):
                missing.add(child_id)
                continue
            pending.append(child["manifest"])
    extra = {
        ref
        for ref, entry in prompts.items()
        if ref not in wanted or ref != f"{entry.get('prompt_id')}:{entry.get('version')}"
    }
    extra |= set(children) - reached
    if missing or extra:
        raise _incomplete(
            "the bundle closure is not exact",
            missing=sorted(missing),
            extra=sorted(extra),
        )


def _check_scope(document: Mapping[str, Any], workspace_id: str, agent_id: str) -> None:
    if (
        document["workspace_id"] != workspace_id
        or document["agent_id"] != agent_id
        or document["manifest"].get("agent_id") != agent_id
    ):
        raise integrity_error(
            "bundle_scope_mismatch", "the bundle belongs to another workspace or agent"
        )


class PromptBundle:
    __slots__ = ("_document", "_signed", "_versions")

    def __init__(self, document: dict[str, Any], *, signed: bool) -> None:
        self._document = document
        self._signed = signed
        self._versions: dict[tuple[str, str], ManagedPromptVersion] = {}

    @classmethod
    def load(
        cls,
        source: Union[str, os.PathLike[str], Mapping[str, Any]],
        *,
        expected_workspace_id: str,
        expected_agent_id: str,
        trust: Optional[BundleTrust] = None,
        expected_bundle_digest: Optional[str] = None,
        expected_manifest_digest: Optional[str] = None,
        allow_ungoverned_bundle: bool = False,
        now: Optional[datetime] = None,
    ) -> PromptBundle:
        document = _shape(_read(source))
        signed = "signature" in document
        if trust is not None and signed:
            _verify_signature(document, trust)
        elif expected_bundle_digest is None:
            raise integrity_error(
                "bundle_untrusted_key",
                "the bundle is neither signed by a trusted key nor pinned by digest",
            )
        cls._verify(
            document,
            expected_workspace_id,
            expected_agent_id,
            expected_bundle_digest,
            expected_manifest_digest,
            now,
        )
        if expected_bundle_digest is None and not allow_ungoverned_bundle:
            governance = document.get("governance")
            if not isinstance(governance, dict) or governance.get("approved") is not True:
                raise integrity_error(
                    "bundle_ungoverned",
                    "the signed bundle pins a release that is not approved",
                    release_status=(
                        governance.get("release_status") if isinstance(governance, dict) else None
                    ),
                )
        return cls(document, signed=signed)

    @classmethod
    def from_online_response(
        cls,
        document: Mapping[str, Any],
        *,
        expected_workspace_id: str,
        expected_agent_id: str,
        expected_manifest_digest: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> PromptBundle:
        shaped = _shape(document)
        if "signature" in shaped:
            raise integrity_error(
                "bundle_scope_mismatch",
                "a signed bundle is an offline artifact; load it with PromptBundle.load",
            )
        cls._verify(
            shaped, expected_workspace_id, expected_agent_id, None, expected_manifest_digest, now
        )
        return cls(shaped, signed=False)

    @staticmethod
    def _verify(
        document: dict[str, Any],
        workspace_id: str,
        agent_id: str,
        expected_bundle_digest: Optional[str],
        expected_manifest_digest: Optional[str],
        now: Optional[datetime],
    ) -> None:
        expires_at = document.get("expires_at")
        if expires_at is not None:
            moment = now or datetime.now(timezone.utc)
            if _parse_time(expires_at) <= moment:
                raise integrity_error(
                    "bundle_expired", "the bundle has expired", expires_at=expires_at
                )
        _check_prompts(document)
        _artifact_set(document, expected_bundle_digest)
        _check_manifests(document, expected_manifest_digest)
        _check_closure(document)
        _check_scope(document, workspace_id, agent_id)

    @property
    def document(self) -> dict[str, Any]:
        return self._document

    @property
    def signed(self) -> bool:
        return self._signed

    @property
    def workspace_id(self) -> str:
        return cast(str, self._document["workspace_id"])

    @property
    def agent_id(self) -> str:
        return cast(str, self._document["agent_id"])

    @property
    def release(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._document["release"])

    @property
    def release_id(self) -> str:
        return str(self.release.get("release_id"))

    @property
    def source(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._document["source"])

    @property
    def governance(self) -> Optional[dict[str, Any]]:
        value = self._document.get("governance")
        return value if isinstance(value, dict) else None

    @property
    def prompt_manifest_digest(self) -> str:
        return cast(str, self._document["prompt_manifest_digest"])

    @property
    def prompt_bundle_digest(self) -> str:
        return cast(str, self._document["prompt_bundle_digest"])

    @property
    def prompt_refs(self) -> list[str]:
        return sorted_keys(self._document["prompts"])

    @property
    def child_agent_ids(self) -> list[str]:
        return sorted_keys(self._document["children"])

    @property
    def child_manifest_digests(self) -> dict[str, str]:
        return {
            child_id: child["prompt_manifest_digest"]
            for child_id, child in self._document["children"].items()
        }

    def manifest(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        if agent_id is None or agent_id == self.agent_id:
            return cast(dict[str, Any], self._document["manifest"])
        child = self._document["children"].get(agent_id)
        if child is None:
            raise binding_error(
                "child_agent_not_pinned",
                "the agent is not pinned by this binding",
                child_agent_id=agent_id,
            )
        return cast(dict[str, Any], child["manifest"])

    def slots(self, agent_id: Optional[str] = None) -> list[str]:
        return sorted_keys(self.manifest(agent_id)["slots"])

    def _record(self, prompt_id: str, version: int) -> Optional[PromptVersionRecord]:
        entry = self._document["prompts"].get(f"{prompt_id}:{version}")
        if entry is None:
            return None
        return PromptVersionRecord(
            prompt_id=entry["prompt_id"],
            version=entry["version"],
            prompt_kind=entry.get("prompt_kind"),
            content_digest=entry["content_digest"],
            content=entry["content"],
            workspace_id=self.workspace_id,
        )

    def version(self, slot_path: str, *, agent_id: Optional[str] = None) -> ManagedPromptVersion:
        manifest = self.manifest(agent_id)
        key = (manifest["agent_id"], slot_path)
        cached = self._versions.get(key)
        if cached is not None:
            return cached
        pin = manifest["slots"].get(slot_path)
        record = None if pin is None else self._record(pin["prompt_id"], pin["version"])
        if record is None:
            raise binding_error(
                "slot_not_in_manifest",
                f"slot {slot_path} is not in the pinned manifest",
                slot_path=slot_path,
                agent_id=manifest["agent_id"],
            )
        version = ManagedPromptVersion.from_record(
            record, workspace_id=self.workspace_id, lookup=self._record
        )
        self._versions[key] = version
        return version

    def pinned_set(
        self, *, binding_id: str, node_children: Optional[Mapping[str, str]] = None
    ) -> PinnedPromptSet:
        return PinnedPromptSet(self, binding_id=binding_id, node_children=node_children)

    def closure(self) -> ResolvedClosure:
        return ResolvedClosure(
            prompt_manifest_digest=self.prompt_manifest_digest,
            manifest=self._document["manifest"],
            children=self._document["children"],
            prompts=self._document["prompts"],
            prompt_bundle_digest=self.prompt_bundle_digest,
        )
