from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from prompt_fakes import (
    AGENT,
    CHILD,
    NOW,
    OTHER_WORKSPACE,
    WORKSPACE,
    chat_content,
    release_with_child,
    required,
    seeded_engine,
    text_content,
)

from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts import (
    BundleTrust,
    LocalPromptEngine,
    PromptBindingError,
    PromptBundle,
    PromptConflictError,
    PromptRefError,
    PromptTemplateError,
    PromptUri,
    PromptVersionRef,
    ResolvedFrom,
    prompt_digest,
)


def raises(code: str, call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as raised:
        call()
    assert raised.value.code == code, raised.value
    return raised.value


def test_registry_publish_and_get() -> None:
    engine = seeded_engine()
    assert engine.prompt("prm_writer")["latest_version"] == 1
    content = text_content("Write well about {topic}.", {"topic": required()})
    second = engine.publish("prm_writer", content, parent_version=1, change_message="better")
    assert second.ref == PromptVersionRef("prm_writer", 2)
    assert engine.publish("prm_writer", content, parent_version=1, change_message="again") == second
    conflict = raises(
        "prompt_version_conflict",
        lambda: engine.publish(
            "prm_writer", text_content("x"), parent_version=1, change_message=""
        ),
    )
    assert isinstance(conflict, PromptConflictError)
    assert conflict.details["current"] == 2
    raises(
        "prompt_version_unchanged",
        lambda: engine.publish(
            "prm_writer",
            {**content, "variables": {"topic": required()}},
            parent_version=2,
            change_message="",
        ),
    )
    raises(
        "validation_error",
        lambda: engine.publish(
            "prm_writer",
            text_content("y"),
            parent_version=2,
            change_message="",
            variable_descriptions={"z": "?"},
        ),
    )
    invalid = raises(
        "prompt_template_invalid",
        lambda: engine.publish(
            "prm_writer", text_content("{x"), parent_version=2, change_message=""
        ),
    )
    assert invalid.status == 400
    raises(
        "prompt_secret_detected",
        lambda: engine.publish(
            "prm_writer", text_content("sk-" + "a" * 30), parent_version=2, change_message=""
        ),
    )
    raises(
        "prompt_kind_mismatch",
        lambda: engine.publish(
            "prm_planner", text_content("x"), parent_version=1, change_message=""
        ),
    )
    assert engine.get("prm_writer:2") == second
    uri = PromptUri(engine.workspace_id, "prm_writer", 2)
    assert engine.get(str(uri)) == second
    assert engine.get(uri) == second
    assert engine.get(PromptVersionRef("prm_writer", 1)).ref.version == 1
    raises(
        "prompt_ref_cross_workspace",
        lambda: engine.get(PromptUri(OTHER_WORKSPACE, "prm_writer", 2)),
    )
    raises("prompt_ref_unversioned", lambda: engine.get("prm_writer"))
    raises("prompt_not_found", lambda: engine.get("prm_missing:1"))
    raises("prompt_version_not_found", lambda: engine.get("prm_writer:9"))


def test_create_prompt_refusals() -> None:
    engine = seeded_engine()
    raises("prompt_ref_invalid", lambda: engine.create_prompt("planner", kind="text", name="x"))
    raises("validation_error", lambda: engine.create_prompt("prm_y", kind="audio", name="x"))
    raises("prompt_id_taken", lambda: engine.create_prompt("prm_writer", kind="text", name="x"))
    with pytest.raises(ValueError):
        LocalPromptEngine("NOT-A-UUID")


def test_drafts_with_revisions() -> None:
    engine = seeded_engine()
    raises("prompt_draft_not_found", lambda: engine.get_draft("prm_writer"))
    draft = engine.save_draft(
        "prm_writer", text_content("{bad"), base_version=1, expected_revision=0
    )
    assert draft["revision"] == 1
    assert draft["validation"]["ok"] is False
    conflict = raises(
        "prompt_draft_conflict",
        lambda: engine.save_draft(
            "prm_writer", text_content("x"), base_version=1, expected_revision=0
        ),
    )
    assert conflict.details["current"] == 1
    raises(
        "prompt_secret_detected",
        lambda: engine.save_draft(
            "prm_writer", text_content("AKIA" + "A" * 16), base_version=1, expected_revision=1
        ),
    )
    raises(
        "prompt_version_not_found",
        lambda: engine.save_draft(
            "prm_writer", text_content("x"), base_version=7, expected_revision=1
        ),
    )
    assert engine.get_draft("prm_writer")["revision"] == 1


def test_aliases_resolve_and_record_the_generation() -> None:
    engine = seeded_engine()
    raises("prompt_alias_not_found", lambda: engine.get("prm_writer@staging"))
    moved = engine.move_alias("prm_writer", "staging", version=1, expected_generation=0)
    assert moved["generation"] == 1
    resolved = engine.get("prm_writer@staging")
    assert resolved.resolved_from == ResolvedFrom("staging", 1)
    raises(
        "prompt_alias_conflict",
        lambda: engine.move_alias("prm_writer", "staging", version=1, expected_generation=0),
    )
    raises(
        "prompt_ref_invalid",
        lambda: engine.move_alias("prm_writer", "Bad", version=1, expected_generation=0),
    )
    raises(
        "prompt_version_not_found",
        lambda: engine.move_alias("prm_writer", "next", version=5, expected_generation=0),
    )
    for index in range(31):
        engine.move_alias("prm_writer", f"a{index}", version=1, expected_generation=0)
    raises(
        "prompt_alias_limit_reached",
        lambda: engine.move_alias("prm_writer", "full", version=1, expected_generation=0),
    )


def test_releases_and_channels() -> None:
    engine = seeded_engine()
    root, child = release_with_child(engine, status="awaiting_approval")
    assert engine.get_release(root)["name"] == "av_0001"
    assert engine.create_release(AGENT, {}, name="custom") != root
    assert engine.create_release(AGENT, {})
    assert {engine.get_release(r)["name"] for r in engine._state["releases"]} >= {"av_0002"}
    raises("candidate_name_taken", lambda: engine.create_release(AGENT, {}, name="custom"))
    raises("validation_error", lambda: engine.create_release("agent", {}))
    raises("validation_error", lambda: engine.create_release(AGENT, {}, status="live"))
    raises(
        "agent_prompt_slot_invalid", lambda: engine.create_release(AGENT, {"Bad": "prm_writer:1"})
    )
    raises("validation_error", lambda: engine.create_release(AGENT, {"a.b": "prm_writer@x"}))
    raises("prompt_kind_mismatch", lambda: engine.create_release(AGENT, {"a.b": "prm_safety:1"}))
    raises("validation_error", lambda: engine.create_release(CHILD, {}, children={CHILD: child}))
    uri = str(PromptUri(engine.workspace_id, "prm_writer", 1))
    assert engine.create_release(AGENT, {"a.b": uri})
    raises("not_found", lambda: engine.get_release("missing"))
    raises("channel_not_found", lambda: engine.get_channel(AGENT, "staging"))
    assert engine.get_channel(AGENT, "production")["materialized"] is False
    raises(
        "release_not_promotable",
        lambda: engine.move_channel(AGENT, "production", root, expected_generation=0),
    )
    staging = engine.move_channel(AGENT, "staging", root, expected_generation=0)
    assert staging["generation"] == 1
    assert staging["protected"] is False
    raises(
        "channel_name_invalid",
        lambda: engine.move_channel(AGENT, "Prod", root, expected_generation=0),
    )
    engine.set_release_status(root, "approved")
    raises("validation_error", lambda: engine.set_release_status(root, "live"))
    production = engine.move_channel(AGENT, "production", root, expected_generation=0)
    assert production["release_id"] == root
    assert engine.get_release(root)["status"] == "production"
    second = engine.create_release(AGENT, {"planner.instructions": "prm_planner:1"})
    conflict = raises(
        "channel_conflict",
        lambda: engine.move_channel(AGENT, "production", second, expected_generation=0),
    )
    assert conflict.details["current"] == 1
    engine.move_channel(AGENT, "production", second, expected_generation=1)
    assert engine.get_release(root)["status"] == "approved"
    assert len(engine.get_channel(AGENT, "production")["history"]) == 2


def test_bindings_first_writer_wins() -> None:
    engine = seeded_engine()
    root, child = release_with_child(engine)
    raises(
        "channel_unassigned",
        lambda: engine.create_binding(AGENT, thread_key="t", scope="thread", channel="production"),
    )
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    binding, artifacts, created = engine.create_binding(
        AGENT, thread_key="t", scope="thread", channel="production"
    )
    assert created
    assert binding["resolved_from"] == {"channel": "production", "generation": 1}
    assert binding["children"][CHILD]["source"] == "manifest"
    assert artifacts["source"] == {"channel": "production", "channel_generation": 1}
    second = engine.create_release(AGENT, {"planner.instructions": "prm_planner:1"})
    engine.move_channel(AGENT, "production", second, expected_generation=1)
    again, _, created_again = engine.create_binding(
        AGENT, thread_key="t", scope="thread", channel="production"
    )
    assert not created_again
    assert again["release_id"] == root
    fresh, _, _ = engine.create_binding(AGENT, thread_key="u", scope="thread", channel="production")
    assert fresh["release_id"] == second
    error = raises(
        "execution_binding_conflict",
        lambda: engine.create_binding(
            AGENT, thread_key="t", scope="execution", channel="production"
        ),
    )
    assert isinstance(error, PromptBindingError)
    raises(
        "execution_binding_conflict",
        lambda: engine.create_binding(
            AGENT,
            thread_key="v",
            scope="thread",
            release_id=root,
            expect_manifest_digest="sha256:" + "0" * 64,
        ),
    )
    assert engine.get_binding(AGENT, binding["binding_id"])[0] == binding
    raises("execution_binding_not_found", lambda: engine.get_binding(AGENT, "bnd_missing"))
    for key, reason in (
        (" t", "grammar"),
        ("a" * 257, "grammar"),
        ("a\u0085", "grammar"),
        ("exp:1", "reserved_prefix"),
    ):
        assert (
            raises(
                "thread_key_invalid",
                lambda key=key: engine.create_binding(
                    AGENT, thread_key=key, scope="thread", release_id=root
                ),
            ).details["reason"]
            == reason
        )
    raises(
        "validation_error",
        lambda: engine.create_binding(AGENT, thread_key="w", scope="run", release_id=root),
    )
    raises(
        "agent_selector_required",
        lambda: engine.create_binding(AGENT, thread_key="w", scope="thread"),
    )
    engine.set_release_status(child, "rejected")
    child_error = raises(
        "release_not_bindable",
        lambda: engine.create_binding(AGENT, thread_key="x", scope="thread", release_id=root),
    )
    assert child_error.details["child_agent_id"] == CHILD
    engine.set_release_status(second, "rolled_back")
    raises("release_not_bindable", lambda: engine.resolve(AGENT, release_id=second))


def test_export_bundle_and_offline_load(tmp_path: Path) -> None:
    engine = seeded_engine()
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    key = SigningKey.generate("orgkey_local")
    bundle = engine.export_bundle(
        AGENT, signer=key, channel="production", now=NOW, expires_in_days=1
    )
    assert bundle["governance"] == {
        "release_status": "production",
        "channel": "production",
        "channel_protected": True,
        "approved": True,
    }
    assert bundle["expires_at"] == "2026-10-06T12:00:00Z"
    loaded = PromptBundle.load(
        bundle,
        expected_workspace_id=WORKSPACE,
        expected_agent_id=AGENT,
        trust=BundleTrust.from_pems({key.key_id: key.public_pem()}),
        now=NOW,
    )
    assert loaded.source == {"channel": "production", "channel_generation": 1}
    raises(
        "validation_error",
        lambda: engine.export_bundle(AGENT, signer=key, release_id=root, expires_in_days=0),
    )
    raises("agent_selector_required", lambda: engine.export_bundle(AGENT, signer=key))
    engine.create_prompt("prm_leaky", kind="text", name="Leaky")
    engine.publish("prm_leaky", text_content("clean"), parent_version=None, change_message="")
    leaky = engine.create_release(AGENT, {"leak.text": "prm_leaky:1"})
    leaky_content = text_content("token xoxb-" + "1" * 12)
    engine._state["prompts"]["prm_leaky"]["versions"]["1"].update(
        content=leaky_content, content_digest=prompt_digest(leaky_content)
    )
    blocked = raises(
        "prompt_bundle_export_blocked",
        lambda: engine.export_bundle(AGENT, signer=key, release_id=leaky),
    )
    assert blocked.details == {"ref": "prm_leaky:1", "pattern": "slack_token"}


def test_state_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "prompts.json"
    engine = LocalPromptEngine(WORKSPACE, state_path=path)
    engine.create_prompt("prm_writer", kind="text", name="Writer")
    engine.publish(
        "prm_writer",
        text_content("Hi {who}", {"who": required()}),
        parent_version=None,
        change_message="",
    )
    reopened = LocalPromptEngine(state_path=path)
    assert reopened.workspace_id == WORKSPACE
    assert reopened.get("prm_writer:1").render_text({"who": "Ada"}) == "Hi Ada"
    with pytest.raises(PromptRefError):
        LocalPromptEngine(OTHER_WORKSPACE, state_path=path)
    path.write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(ValueError):
        LocalPromptEngine(state_path=path)


def test_chat_template_refusal_in_engine() -> None:
    engine = LocalPromptEngine(WORKSPACE)
    engine.create_prompt("prm_chat", kind="chat", name="Chat")
    with pytest.raises(PromptTemplateError):
        engine.publish("prm_chat", chat_content([]), parent_version=None, change_message="")
