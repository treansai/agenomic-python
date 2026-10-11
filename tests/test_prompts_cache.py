from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest
from prompt_fakes import (
    AGENT,
    OTHER_WORKSPACE,
    WORKSPACE,
    release_with_child,
    seeded_engine,
    text_content,
)

from agenomic.prompts import (
    ExecutionBinding,
    ManagedPromptVersion,
    PromptBundle,
    PromptCache,
    PromptIntegrityError,
    PromptVersionRecord,
    prompt_digest,
    thread_key,
)


def _versions() -> tuple:
    first = seeded_engine(WORKSPACE)
    second = seeded_engine(OTHER_WORKSPACE)
    second.publish(
        "prm_writer",
        text_content("Other text about {topic}.", {"topic": {"type": "string", "required": True}}),
        parent_version=1,
        change_message="second",
    )
    return first, second


def test_two_workspaces_same_ref_never_share_cache(tmp_path: Path) -> None:
    first, second = _versions()
    for directory in (None, tmp_path):
        cache = PromptCache(directory)
        mine = first.get_version("prm_planner", 1)
        cache.put_version(WORKSPACE, mine)
        assert cache.get_version(OTHER_WORKSPACE, "prm_planner", 1) is None
        theirs = second.get_version("prm_planner", 1)
        cache.put_version(OTHER_WORKSPACE, theirs)
        assert cache.get_version(WORKSPACE, "prm_planner", 1) is mine
        assert cache.get_version(OTHER_WORKSPACE, "prm_planner", 1) is theirs
        with pytest.raises(PromptIntegrityError) as raised:
            cache.put_version(OTHER_WORKSPACE, mine)
        assert raised.value.code == "cache_conflict"
    reopened = PromptCache(tmp_path)
    loaded = reopened.get_version(WORKSPACE, "prm_planner", 1)
    assert loaded is not None
    assert loaded.content_digest == first.get_version("prm_planner", 1).content_digest
    assert loaded.render_messages({"customer": "a", "question": "b"})[0].content.endswith(
        "Never share internal notes."
    )


def test_two_digests_under_one_version_conflict(tmp_path: Path) -> None:
    first, second = _versions()
    other = second.get_version("prm_writer", 2)
    mine = first.get_version("prm_writer", 1)
    cache = PromptCache(tmp_path)
    cache.put_version(WORKSPACE, mine)
    forged = other.model_copy(update={"workspace_id": WORKSPACE, "ref": mine.ref})
    with pytest.raises(PromptIntegrityError):
        cache.put_version(WORKSPACE, forged)
    fresh = PromptCache(tmp_path)
    directory = tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1"
    (directory / ("0" * 64 + ".json")).write_text("{}", encoding="utf-8")
    with pytest.raises(PromptIntegrityError) as raised:
        fresh.get_version(WORKSPACE, "prm_writer", 1)
    assert raised.value.code == "cache_conflict"
    with pytest.raises(PromptIntegrityError):
        fresh.put_version(WORKSPACE, mine)


def test_disk_tamper_is_cache_conflict(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    engine = seeded_engine()
    version = engine.get_version("prm_writer", 1)
    PromptCache(tmp_path).put_version(WORKSPACE, version)
    directory = tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1"
    (path,) = list(directory.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["closure"][0]["content"]["body"] = "Write about {topic}!"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptIntegrityError) as raised:
        PromptCache(tmp_path).get_version(WORKSPACE, "prm_writer", 1)
    assert raised.value.code == "cache_conflict"
    assert "prompt cache conflict" in caplog.text
    for broken in ("not json", json.dumps({"schema": "x"}), json.dumps({**payload, "closure": 3})):
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(PromptIntegrityError):
            PromptCache(tmp_path).get_version(WORKSPACE, "prm_writer", 1)
    renamed = directory / ("f" * 64 + ".json")
    path.write_text(json.dumps({**payload, "closure": payload["closure"]}), encoding="utf-8")
    PromptCache(tmp_path).put_version(WORKSPACE, version)
    path.rename(renamed)
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_version(WORKSPACE, "prm_writer", 1)


def test_disk_paths_reject_traversal(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path)
    for workspace, prompt_id, number in (
        ("../" + WORKSPACE, "prm_x", 1),
        (WORKSPACE, "../prm_x", 1),
        (WORKSPACE, "prm_x", 0),
        (WORKSPACE, "prm_x", True),
    ):
        with pytest.raises(ValueError):
            cache.get_version(workspace, prompt_id, number)
    with pytest.raises(ValueError):
        cache.get_closure(WORKSPACE, "sha256:../../etc")
    with pytest.raises(ValueError):
        cache.get_binding(WORKSPACE, "../agent", "thread")
    with pytest.raises(ValueError):
        cache.get_binding(WORKSPACE, AGENT, "")
    assert not any(tmp_path.iterdir())


def test_put_version_rejects_traversal(tmp_path: Path) -> None:
    content = text_content("hello")
    for prompt_id, number in (("../../../../escape", 1), ("prm_x", 0)):
        record = PromptVersionRecord(
            prompt_id=prompt_id,
            version=number,
            prompt_kind="text",
            content_digest=prompt_digest(content),
            content=content,
        )
        version = ManagedPromptVersion.from_record(
            record, workspace_id=WORKSPACE, lookup=lambda pid, v: None
        )
        for cache in (PromptCache(), PromptCache(tmp_path / "a" / "cache")):
            with pytest.raises(ValueError):
                cache.put_version(WORKSPACE, version)
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_disk_modes(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path)
    cache.put_version(WORKSPACE, seeded_engine().get_version("prm_writer", 1))
    for path in (tmp_path / "v1", *(tmp_path / "v1").rglob("*")):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path


def test_closures_and_bindings(tmp_path: Path) -> None:
    engine = seeded_engine()
    release, _ = release_with_child(engine)
    key = thread_key(WORKSPACE, "conversation-1")
    binding_doc, artifacts, _ = engine.create_binding(
        AGENT, thread_key=key, scope="thread", release_id=release
    )
    binding = ExecutionBinding.model_validate(binding_doc)
    closure = PromptBundle.from_online_response(
        artifacts, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT
    ).closure()
    cache = PromptCache(tmp_path)
    cache.put_closure(WORKSPACE, closure)
    cache.put_binding(WORKSPACE, binding)
    assert cache.get_closure(WORKSPACE, closure.prompt_manifest_digest) is closure
    assert cache.get_binding(WORKSPACE, AGENT, key) is binding
    reopened = PromptCache(tmp_path)
    assert reopened.get_closure(WORKSPACE, closure.prompt_manifest_digest) == closure
    assert reopened.get_binding(WORKSPACE, AGENT, key) == binding
    assert reopened.get_closure(OTHER_WORKSPACE, closure.prompt_manifest_digest) is None
    assert reopened.get_binding(WORKSPACE, AGENT, "other") is None
    with pytest.raises(PromptIntegrityError):
        cache.put_binding(OTHER_WORKSPACE, binding)
    cache.evict_binding(WORKSPACE, AGENT, key)
    assert PromptCache(tmp_path).get_binding(WORKSPACE, AGENT, key) is None
    cache.evict_binding(WORKSPACE, AGENT, key)
    with pytest.raises(PromptIntegrityError):
        cache.put_closure(
            WORKSPACE, closure.model_copy(update={"prompt_bundle_digest": "sha256:" + "0" * 64})
        )
    path = next((tmp_path / "v1" / WORKSPACE / "closures").glob("*.json"))
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["prompts"]["prm_writer:1"]["content"]["body"] = "x"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_closure(WORKSPACE, closure.prompt_manifest_digest)
    path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_closure(WORKSPACE, closure.prompt_manifest_digest)
    del tampered["prompts"]["prm_writer:1"]["content"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_closure(WORKSPACE, closure.prompt_manifest_digest)
    cache.put_binding(WORKSPACE, binding)
    binding_path = next((tmp_path / "v1" / WORKSPACE / "bindings" / AGENT).glob("*.json"))
    binding_path.write_text(json.dumps({**binding_doc, "thread_key": "other"}), encoding="utf-8")
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_binding(WORKSPACE, AGENT, key)
    binding_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(PromptIntegrityError):
        PromptCache(tmp_path).get_binding(WORKSPACE, AGENT, key)


def test_memory_lru_eviction() -> None:
    engine = seeded_engine()
    cache = PromptCache(max_memory_entries=1, max_bindings=1)
    writer = engine.get_version("prm_writer", 1)
    planner = engine.get_version("prm_planner", 1)
    cache.put_version(WORKSPACE, writer)
    cache.put_version(WORKSPACE, planner)
    assert cache.get_version(WORKSPACE, "prm_writer", 1) is None
    assert cache.get_version(WORKSPACE, "prm_planner", 1) is planner
    release, _ = release_with_child(engine)
    bindings = []
    for name in ("a", "b"):
        doc, _, _ = engine.create_binding(
            AGENT, thread_key=name, scope="thread", release_id=release
        )
        bindings.append(ExecutionBinding.model_validate(doc))
        cache.put_binding(WORKSPACE, bindings[-1])
    assert cache.get_binding(WORKSPACE, AGENT, "a") is None
    assert cache.get_binding(WORKSPACE, AGENT, "b") is bindings[1]


def test_async_twins(tmp_path: Path) -> None:
    engine = seeded_engine()
    release, _ = release_with_child(engine)
    doc, artifacts, _ = engine.create_binding(
        AGENT, thread_key="t", scope="thread", release_id=release
    )
    binding = ExecutionBinding.model_validate(doc)
    closure = PromptBundle.from_online_response(
        artifacts, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT
    ).closure()
    version = engine.get_version("prm_writer", 1)

    async def scenario(cache: PromptCache) -> None:
        await cache.aput_version(WORKSPACE, version)
        assert await cache.aget_version(WORKSPACE, "prm_writer", 1) is not None
        await cache.aput_closure(WORKSPACE, closure)
        assert await cache.aget_closure(WORKSPACE, closure.prompt_manifest_digest) is not None
        await cache.aput_binding(WORKSPACE, binding)
        assert await cache.aget_binding(WORKSPACE, AGENT, "t") is not None
        await cache.aevict_binding(WORKSPACE, AGENT, "t")
        assert await cache.aget_binding(WORKSPACE, AGENT, "t") is None

    asyncio.run(scenario(PromptCache()))
    asyncio.run(scenario(PromptCache(tmp_path)))
