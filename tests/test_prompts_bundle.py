from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from prompt_fakes import (
    AGENT,
    CHILD,
    NOW,
    OTHER_AGENT,
    OTHER_WORKSPACE,
    WORKSPACE,
    make_signed_bundle,
    release_with_child,
    seeded_engine,
)

from agenomic.crypto.signing import SigningKey
from agenomic.prompts import (
    BundleTrust,
    PromptBindingError,
    PromptBundle,
    PromptIntegrityError,
    build_bundle,
    content_digest,
    manifest_digest,
)
from agenomic.prompts.digest import artifact_set_digest

SPEC_SIGNED = Path(__file__).parent / "fixtures" / "prompt_bundles" / "spec-signed.json"
SPEC_AGENT = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"


def trust_for(key: SigningKey) -> BundleTrust:
    return BundleTrust.from_pems({key.key_id: key.public_pem()})


def load(bundle: Any, **kwargs: Any) -> PromptBundle:
    options: dict[str, Any] = {
        "expected_workspace_id": WORKSPACE,
        "expected_agent_id": AGENT,
        "now": NOW,
    }
    options.update(kwargs)
    return PromptBundle.load(bundle, **options)


def code_of(bundle: Any, **kwargs: Any) -> str:
    with pytest.raises(PromptIntegrityError) as raised:
        load(bundle, **kwargs)
    return raised.value.code


def test_signed_bundle_trusted_key(tmp_path: Path) -> None:
    bundle, key = make_signed_bundle()
    path = tmp_path / f"{key.key_id}.pem"
    key.write_public_pem_file(path)
    loaded = load(bundle, trust=BundleTrust.from_pem_files(path))
    assert loaded.signed
    assert loaded.governance == bundle["governance"]
    assert loaded.release_id == bundle["release"]["release_id"]
    assert loaded.source == {"release_id": loaded.release_id}
    assert loaded.child_agent_ids == [CHILD]
    assert loaded.slots() == ["planner.instructions"]
    assert loaded.slots(CHILD) == ["writer.response"]
    planner = loaded.version("planner.instructions")
    assert planner.workspace_id == WORKSPACE
    assert planner.render_messages({"customer": "Acme", "question": "Why?"})[0].content == (
        "Plan for Acme. Never share internal notes."
    )
    assert loaded.version("planner.instructions") is planner
    writer = loaded.version("writer.response", agent_id=CHILD)
    assert writer.render_text({"topic": "tea"}) == "Write about tea."
    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_text(json.dumps(bundle), encoding="utf-8")
    assert (
        load(bundle_file, trust=trust_for(key)).prompt_bundle_digest
        == bundle["prompt_bundle_digest"]
    )


def test_spec_fixture_signature_verifies_and_embedded_pem_is_ignored() -> None:
    document = json.loads(SPEC_SIGNED.read_text(encoding="utf-8"))
    embedded = BundleTrust.from_pems({"orgkey_2026_09": document["signature"]["public_key_pem"]})
    loaded = PromptBundle.load(
        document,
        expected_workspace_id=document["workspace_id"],
        expected_agent_id=SPEC_AGENT,
        trust=embedded,
        now=datetime(2026, 10, 10, tzinfo=timezone.utc),
    )
    assert loaded.signed
    with pytest.raises(PromptIntegrityError) as untrusted:
        PromptBundle.load(
            document,
            expected_workspace_id=document["workspace_id"],
            expected_agent_id=SPEC_AGENT,
            now=datetime(2026, 10, 10, tzinfo=timezone.utc),
        )
    assert untrusted.value.code == "bundle_untrusted_key"


def test_embedded_pem_never_trusted() -> None:
    org = SigningKey.generate("orgkey_test")
    attacker = SigningKey.generate("orgkey_test")
    forged, _ = make_signed_bundle(signer=attacker)
    assert forged["signature"]["public_key_pem"] == attacker.public_pem()
    assert code_of(forged, trust=trust_for(org)) == "bundle_signature_invalid"
    assert code_of(forged, trust=BundleTrust({})) == "bundle_untrusted_key"
    assert code_of(forged) == "bundle_untrusted_key"


def test_tampered_signed_bundle_fails_signature() -> None:
    bundle, key = make_signed_bundle()
    tampered = copy.deepcopy(bundle)
    tampered["exported_at"] = "2026-10-05T12:00:01Z"
    assert code_of(tampered, trust=trust_for(key)) == "bundle_signature_invalid"
    malformed = copy.deepcopy(bundle)
    malformed["signature"]["value"] = "not base64!"
    assert code_of(malformed, trust=trust_for(key)) == "bundle_signature_invalid"
    del malformed["issuer"]
    assert code_of(malformed, trust=trust_for(key)) == "bundle_signature_invalid"


def test_unsigned_bundle_requires_expected_digest() -> None:
    engine = seeded_engine()
    release, _ = release_with_child(engine)
    document = engine.resolve(AGENT, release_id=release)
    assert code_of(document) == "bundle_untrusted_key"
    loaded = load(document, expected_bundle_digest=document["prompt_bundle_digest"])
    assert not loaded.signed
    assert loaded.prompt_refs == ["prm_planner:1", "prm_safety:1", "prm_writer:1"]


def test_unpinned_bundle_refused() -> None:
    bundle, key = make_signed_bundle()
    unsigned = {k: v for k, v in bundle.items() if k not in ("signature", "issuer")}
    assert code_of(unsigned, trust=trust_for(key)) == "bundle_untrusted_key"


def test_other_agent_bundle_same_org_key_refused() -> None:
    key = SigningKey.generate("orgkey_test")
    other, _ = make_signed_bundle(signer=key, agent_id=OTHER_AGENT)
    assert code_of(other, trust=trust_for(key)) == "bundle_scope_mismatch"
    foreign, _ = make_signed_bundle(signer=key, workspace_id=OTHER_WORKSPACE)
    assert code_of(foreign, trust=trust_for(key)) == "bundle_scope_mismatch"


def test_rejected_release_bundle_refused_offline() -> None:
    bundle, key = make_signed_bundle(status="awaiting_approval")
    assert bundle["governance"]["approved"] is False
    with pytest.raises(PromptIntegrityError) as raised:
        load(bundle, trust=trust_for(key))
    assert raised.value.code == "bundle_ungoverned"
    assert raised.value.details["release_status"] == "awaiting_approval"
    assert load(bundle, trust=trust_for(key), allow_ungoverned_bundle=True).signed
    pinned = load(bundle, expected_bundle_digest=bundle["prompt_bundle_digest"])
    assert pinned.governance is not None
    stripped = {k: v for k, v in bundle.items() if k != "governance"}
    resigned = build_bundle(stripped, signer=key)
    assert code_of(resigned, trust=trust_for(key)) == "bundle_ungoverned"


def test_child_prompt_swapped_fails_bundle_pin() -> None:
    bundle, key = make_signed_bundle()
    pin = bundle["prompt_bundle_digest"]
    swapped = copy.deepcopy(bundle)
    entry = swapped["prompts"]["prm_writer:1"]
    entry["content"]["body"] = "Write anything about {topic}."
    entry["content_digest"] = content_digest(entry["content"])
    child = swapped["children"][CHILD]
    child["manifest"]["slots"]["writer.response"]["content_digest"] = entry["content_digest"]
    child["prompt_manifest_digest"] = manifest_digest(child["manifest"])
    swapped["prompt_bundle_digest"] = artifact_set_digest(swapped)
    with pytest.raises(PromptIntegrityError) as raised:
        load(swapped, expected_bundle_digest=pin)
    assert raised.value.code == "prompt_digest_mismatch"
    assert raised.value.details["document"] == "artifact_set"
    assert raised.value.details["expected"] == pin


def test_missing_fragment_incomplete() -> None:
    bundle, _ = make_signed_bundle()
    broken = copy.deepcopy(bundle)
    del broken["prompts"]["prm_safety:1"]
    broken["prompt_bundle_digest"] = artifact_set_digest(broken)
    with pytest.raises(PromptIntegrityError) as raised:
        load(broken, expected_bundle_digest=broken["prompt_bundle_digest"])
    assert raised.value.code == "bundle_incomplete"
    assert raised.value.details["missing"] == ["prm_safety:1"]
    extra = copy.deepcopy(bundle)
    extra["prompts"]["prm_extra:1"] = copy.deepcopy(extra["prompts"]["prm_writer:1"])
    extra["prompt_bundle_digest"] = artifact_set_digest(extra)
    with pytest.raises(PromptIntegrityError) as surplus:
        load(extra, expected_bundle_digest=extra["prompt_bundle_digest"])
    assert surplus.value.details["extra"] == ["prm_extra:1"]
    orphan = copy.deepcopy(bundle)
    del orphan["children"][CHILD]
    del orphan["prompts"]["prm_writer:1"]
    orphan["prompt_bundle_digest"] = artifact_set_digest(orphan)
    with pytest.raises(PromptIntegrityError) as child:
        load(orphan, expected_bundle_digest=orphan["prompt_bundle_digest"])
    assert child.value.details["missing"] == [CHILD]


def test_expired_bundle() -> None:
    bundle, key = make_signed_bundle()
    later = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert code_of(bundle, trust=trust_for(key), now=later) == "bundle_expired"


def test_integrity_failures() -> None:
    bundle, key = make_signed_bundle()
    pin = bundle["prompt_bundle_digest"]
    entry_changed = copy.deepcopy(bundle)
    entry_changed["prompts"]["prm_writer:1"]["content"]["body"] = "changed {topic}"
    assert code_of(entry_changed, expected_bundle_digest=pin) == "prompt_digest_mismatch"
    in_file = copy.deepcopy(bundle)
    in_file["prompt_bundle_digest"] = "sha256:" + "0" * 64
    assert code_of(in_file, expected_bundle_digest=pin) == "prompt_digest_mismatch"
    manifest_changed = copy.deepcopy(bundle)
    manifest_changed["prompt_manifest_digest"] = "sha256:" + "1" * 64
    manifest_changed["prompt_bundle_digest"] = artifact_set_digest(manifest_changed)
    assert (
        code_of(manifest_changed, expected_bundle_digest=manifest_changed["prompt_bundle_digest"])
        == "manifest_digest_mismatch"
    )
    assert (
        code_of(bundle, trust=trust_for(key), expected_manifest_digest="sha256:" + "2" * 64)
        == "manifest_digest_mismatch"
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda b: b.update({"schema": "agenomic.prompt_bundle/v2"}), "unsupported_schema"),
        (lambda b: b.pop("manifest"), "missing_field"),
        (lambda b: b.update({"children": []}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": 5}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "tomorrow"}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "2026-11-03T12:00:00"}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "2026-11-03T12:00:00.12Z"}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "2026-11-03T12:00:00+00:00"}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "2026-11-03 12:00:00Z"}), "invalid_field_type"),
        (lambda b: b.update({"expires_at": "2026-02-30T12:00:00Z"}), "invalid_field_type"),
        (lambda b: b["prompts"].update({"prm_x:1": "bad"}), "invalid_field_type"),
        (lambda b: b["children"].update({CHILD: "bad"}), "invalid_field_type"),
        (lambda b: b["manifest"].update({"schema": "x"}), "invalid_field_type"),
        (lambda b: b.update({"note": 0.5}), "float_not_allowed"),
    ],
)
def test_malformed_bundles_are_incomplete(mutate: Any, reason: str) -> None:
    bundle, _ = make_signed_bundle()
    broken = copy.deepcopy(bundle)
    mutate(broken)
    if reason == "invalid_field_type" and isinstance(broken["children"], dict):
        broken["prompt_bundle_digest"] = artifact_set_digest(broken)
    with pytest.raises(PromptIntegrityError) as raised:
        load(broken, expected_bundle_digest=broken["prompt_bundle_digest"])
    assert raised.value.code == "bundle_incomplete"
    assert raised.value.details["reason"] == reason


def test_non_object_bundle_file(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(PromptIntegrityError) as raised:
        load(path, expected_bundle_digest="sha256:" + "0" * 64)
    assert raised.value.details["reason"] == "invalid_field_type"


def test_from_online_response() -> None:
    engine = seeded_engine()
    release, _ = release_with_child(engine)
    document = engine.resolve(AGENT, release_id=release)
    loaded = PromptBundle.from_online_response(
        document,
        expected_workspace_id=WORKSPACE,
        expected_agent_id=AGENT,
        expected_manifest_digest=document["prompt_manifest_digest"],
    )
    assert loaded.closure().artifact_set()["prompts"] == document["prompts"]
    signed, _ = make_signed_bundle()
    with pytest.raises(PromptIntegrityError) as raised:
        PromptBundle.from_online_response(
            signed, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT
        )
    assert raised.value.code == "bundle_scope_mismatch"
    with pytest.raises(PromptIntegrityError) as scope:
        PromptBundle.from_online_response(
            document, expected_workspace_id=WORKSPACE, expected_agent_id=OTHER_AGENT
        )
    assert scope.value.code == "bundle_scope_mismatch"


def test_slots_and_children_fail_closed() -> None:
    bundle, key = make_signed_bundle()
    loaded = load(bundle, trust=trust_for(key))
    with pytest.raises(PromptBindingError) as slot:
        loaded.version("planner.unknown")
    assert slot.value.code == "slot_not_in_manifest"
    with pytest.raises(PromptBindingError) as child:
        loaded.version("writer.response", agent_id=OTHER_AGENT)
    assert child.value.code == "child_agent_not_pinned"
    pinned = loaded.pinned_set(binding_id="bnd_test", node_children={"writer": CHILD})
    assert pinned.binding_id == "bnd_test"
    assert pinned.workspace_id == WORKSPACE
    assert pinned.agent_id == AGENT
    assert pinned.prompt_manifest_digest == loaded.prompt_manifest_digest
    assert pinned.children == {CHILD: bundle["children"][CHILD]["prompt_manifest_digest"]}
    assert pinned.agent_for_node("writer") == CHILD
    assert pinned.agent_for_node("planner") == AGENT
    assert pinned.version("writer.response", agent_id=CHILD).ref.prompt_id == "prm_writer"
    with pytest.raises(PromptBindingError):
        loaded.pinned_set(binding_id="bnd_test", node_children={"x": OTHER_AGENT})


def test_trust_store_refuses_non_ed25519_keys(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises(ValueError):
        BundleTrust.from_pems({"k": pem})
