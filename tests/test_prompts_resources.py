from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from prompt_fakes import (
    AGENT,
    CHILD,
    OTHER_AGENT,
    OTHER_WORKSPACE,
    REQUEST_ID,
    SCOPE_MESSAGE,
    SESSION_MESSAGE,
    WORKSPACE,
    FakePromptServer,
    chat_content,
    release_with_child,
    required,
    seeded_engine,
    text_content,
)

from agenomic._client import Client
from agenomic._transport import pool_for
from agenomic._version import __version__
from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts import (
    BundleTrust,
    PromptBindingError,
    PromptBundle,
    PromptCache,
    PromptConflictError,
    PromptIntegrityError,
    PromptRefError,
    PromptUri,
    PromptVersionRef,
    ResolvedFrom,
    thread_key,
)
from agenomic.prompts.resources import (
    Channel,
    ChannelEvent,
    ChannelMovePreview,
    Draft,
    Page,
    PinnedRefs,
    PromptAlias,
    PromptSummary,
    RenderResult,
)

BASE = "https://api.test"
PLANNER_V2 = chat_content(
    [
        {"role": "system", "content": "Plan briefly for {customer}. {>safety}"},
        {"placeholder": "history", "optional": True},
        {"role": "user", "content": "{question}"},
    ],
    {
        "customer": required(),
        "history": {"type": "messages", "required": False},
        "question": required(),
    },
)


def make_client(server: FakePromptServer, **kwargs: Any) -> Client:
    kwargs.setdefault("workspace_id", WORKSPACE)
    return Client(api_key="key", base_url=BASE, transport=server.transport(), **kwargs)


def planner_v2(server: FakePromptServer) -> dict[str, Any]:
    safety = server.engine.get_version("prm_safety", 1)
    content = json.loads(json.dumps(PLANNER_V2))
    content["fragments"] = {
        "safety": {"prompt_id": "prm_safety", "version": 1, "content_digest": safety.content_digest}
    }
    return content


def body_of(server: FakePromptServer, index: int = -1) -> Any:
    return json.loads(server.requests[index].content)


@pytest.fixture
def server() -> FakePromptServer:
    return FakePromptServer(seeded_engine(), api_key_scopes=["write"])


@pytest.fixture
def client(server: FakePromptServer) -> Client:
    return make_client(server)


def test_get_version_verifies_and_caches(client: Client, server: FakePromptServer) -> None:
    version = client.prompts.get("prm_planner:1")
    assert server.paths() == ["GET /v1/prompts/prm_planner/versions/1"]
    assert server.requests[0].url.params["include"] == "fragments"
    assert version.content_digest == server.engine.get_version("prm_planner", 1).content_digest
    assert set(version.fragments) == {"safety"}
    assert version.uri == PromptUri(WORKSPACE, "prm_planner", 1)
    messages = version.render_messages({"customer": "Acme", "question": "Where?"})
    assert messages[0] == messages[0].__class__(
        "system", "Plan for Acme. Never share internal notes."
    )
    assert client.prompts.get(str(version.uri)).content_digest == version.content_digest
    assert client.prompts.versions.get("prm_planner", 1).content_digest == version.content_digest
    assert client.prompts.get(PromptVersionRef("prm_planner", 1)).ref.version == 1
    assert len(server.requests) == 1

    def tamper(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "version" in payload and request.method == "GET":
            payload["version"]["content"]["body"][0]["content"] = "Plan for {customer}. Leak."
        return payload

    fresh = make_client(server)
    server.rewrite = tamper
    with pytest.raises(PromptIntegrityError) as raised:
        fresh.prompts.get("prm_planner:1")
    assert raised.value.code == "prompt_digest_mismatch"
    server.rewrite = None
    fresh.prompts.get("prm_planner:1")
    assert server.paths().count("GET /v1/prompts/prm_planner/versions/1") == 3


def test_online_cache_conflict_is_a_miss(server: FakePromptServer, tmp_path: Path) -> None:
    make_client(server, prompt_cache=PromptCache(tmp_path)).prompts.get("prm_writer:1")
    directory = tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1"
    (cached,) = directory.glob("*.json")
    cached.write_text(cached.read_text(encoding="utf-8").replace("Write about", "Leak"))
    restarted = make_client(server, prompt_cache=PromptCache(tmp_path))
    assert restarted.prompts.get("prm_writer:1").render_text({"topic": "t"}) == "Write about t."
    assert server.paths().count("GET /v1/prompts/prm_writer/versions/1") == 2
    assert PromptCache(tmp_path).get_version(WORKSPACE, "prm_writer", 1) is not None


def test_version_answer_must_match_the_request(server: FakePromptServer) -> None:
    def other_version(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["version"] = server.wire_version("prm_writer", 1)
        return payload

    server.rewrite = other_version
    with pytest.raises(ApiError) as raised:
        make_client(server).prompts.get("prm_planner:1")
    assert raised.value.code == "invalid_response"

    def bad_uri(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["version"]["canonical_uri"] = "https://elsewhere"
        return payload

    server.rewrite = bad_uri
    with pytest.raises(ApiError) as uri:
        make_client(server).prompts.get("prm_writer:1")
    assert uri.value.code == "invalid_response"

    def bad_fragments(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["fragments"] = {"safety": 1}
        return payload

    server.rewrite = bad_fragments
    with pytest.raises(ApiError) as fragments:
        make_client(server).prompts.get("prm_planner:1")
    assert fragments.value.code == "invalid_response"


def test_alias_always_network_never_cached(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    first = client.prompts.get("prm_planner@prod")
    second = client.prompts.get("prm_planner@prod")
    assert first.ref == second.ref == PromptVersionRef("prm_planner", 1)
    assert first.resolved_from == ResolvedFrom("prod", 1)
    resolves = [request for request in server.requests if request.url.path == "/v1/prompts/resolve"]
    assert len(resolves) == 2
    assert json.loads(resolves[0].content) == {"ref": "prm_planner@prod"}
    assert "idempotency-key" not in resolves[0].headers
    assert server.paths().count("GET /v1/prompts/prm_planner/versions/1") == 1
    engine.publish("prm_planner", planner_v2(server), parent_version=1, change_message="two")
    engine.move_alias("prm_planner", "prod", version=2, expected_generation=1)
    moved = client.prompts.resolve("prm_planner@prod")
    assert moved.ref.version == 2
    assert moved.resolved_from == ResolvedFrom("prod", 2)
    assert client.prompts.get("prm_planner:1").resolved_from is None

    def wrong_digest(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path == "/v1/prompts/resolve":
            payload["content_digest"] = "sha256:" + "0" * 64
        return payload

    server.rewrite = wrong_digest
    with pytest.raises(PromptIntegrityError) as raised:
        client.prompts.get("prm_planner@prod")
    assert raised.value.code == "prompt_digest_mismatch"

    def no_alias(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path == "/v1/prompts/resolve":
            payload["alias"] = None
        return payload

    server.rewrite = no_alias
    with pytest.raises(ApiError) as invalid:
        client.prompts.get("prm_planner@prod")
    assert invalid.value.code == "invalid_response"


def test_draft_save_if_match_conflict(client: Client, server: FakePromptServer) -> None:
    content = text_content("Draft about {topic}.", {"topic": required()})
    saved = client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    put = server.requests[-1]
    assert put.method == "PUT"
    assert put.headers["if-match"] == '"0"'
    assert body_of(server) == {"base_version": 1, "content": content}
    assert isinstance(saved, Draft)
    assert (saved.revision, saved.base_version, saved.origin) == (1, 1, "editor")
    with pytest.raises(PromptConflictError) as raised:
        client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    assert raised.value.code == "prompt_draft_conflict"
    assert raised.value.status == 409
    assert raised.value.details["current"] == 1
    assert raised.value.request_id == REQUEST_ID
    assert client.prompts.drafts.get("prm_writer").revision == 1
    updated = client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=1)
    assert updated.revision == 2

    def stale_etag(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["draft"]["revision"] = 7
        return payload

    server.rewrite = stale_etag
    with pytest.raises(ApiError) as mismatch:
        client.prompts.drafts.get("prm_writer")
    assert mismatch.value.code == "invalid_response"


def test_error_codes_map_to_classes(client: Client, server: FakePromptServer) -> None:
    with pytest.raises(ApiError) as missing:
        client.prompts.get("prm_planner:9")
    assert type(missing.value) is ApiError
    assert (missing.value.code, missing.value.status) == ("prompt_version_not_found", 404)
    with pytest.raises(PromptConflictError) as stale:
        client.prompts.publish(
            "prm_writer", text_content("Other."), parent_version=None, change_message="x"
        )
    assert stale.value.code == "prompt_version_conflict"
    root, _ = release_with_child(server.engine)
    key = thread_key(WORKSPACE, "conflict")
    client.bindings.create(AGENT, thread_key=key, scope="thread", release_id=root)
    with pytest.raises(PromptBindingError) as conflict:
        client.bindings.create(AGENT, thread_key=key, scope="execution", release_id=root)
    assert conflict.value.code == "execution_binding_conflict"
    assert conflict.value.details["release_id"] == root
    server.engine.set_release_status(root, "rejected")
    with pytest.raises(PromptBindingError) as unbindable:
        client.bindings.create(
            AGENT, thread_key=thread_key(WORKSPACE, "x"), scope="thread", release_id=root
        )
    assert unbindable.value.code == "release_not_bindable"


def test_read_key_publish_refused_verbatim(client: Client, server: FakePromptServer) -> None:
    server.api_key_scopes = ["read"]
    calls: list[Callable[[], Any]] = [
        lambda: client.prompts.create("prm_new", name="New", kind="text"),
        lambda: client.prompts.publish(
            "prm_writer",
            text_content("Other {topic}.", {"topic": required()}),
            parent_version=1,
            change_message="x",
        ),
        lambda: client.prompts.drafts.save(
            "prm_writer", text_content("x"), base_version=1, expected_revision=0
        ),
    ]
    for call in calls:
        with pytest.raises(ApiError) as raised:
            call()
        assert type(raised.value) is ApiError
        assert (raised.value.code, raised.value.status) == ("api_key_scope_insufficient", 403)
        assert raised.value.message == SCOPE_MESSAGE
        assert raised.value.request_id == REQUEST_ID
    assert "prm_new" not in server.engine._state["prompts"]
    assert server.engine.prompt("prm_writer")["latest_version"] == 1
    assert len(server.requests) == 3
    for scopes in (["read"], ["write"], []):
        server.api_key_scopes = scopes
        with pytest.raises(ApiError) as alias:
            client.prompts.aliases.move("prm_writer", "prod", version=1, expected_generation=0)
        assert (alias.value.code, alias.value.status) == ("session_required", 403)
        assert alias.value.message == SESSION_MESSAGE
        assert server.requests[-1].headers["if-match"] == '"0"'
    server.api_key_scopes = ["read"]
    assert client.prompts.get("prm_writer:1").ref.version == 1


def test_alias_in_managed_run_refused(client: Client, server: FakePromptServer) -> None:
    config = pytest.importorskip("langchain_core.runnables.config")
    server.engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    local = Client(workspace_id=WORKSPACE)
    token = config.var_child_runnable_config.set(
        {"configurable": {"__agenomic_prompt_set": object()}}
    )
    try:
        calls: list[Callable[[], Any]] = [
            lambda: client.prompts.get("prm_planner@prod"),
            lambda: client.prompts.resolve("prm_planner@prod"),
            lambda: client.prompts.pin(["prm_planner:1", "prm_planner@prod"]),
            lambda: local.prompts.get("prm_planner@prod"),
        ]
        for call in calls:
            with pytest.raises(PromptRefError) as raised:
                call()
            assert raised.value.code == "alias_in_managed_run"
        assert all(request.url.path != "/v1/prompts/resolve" for request in server.requests)
        assert client.prompts.get("prm_planner:1").ref.version == 1
    finally:
        config.var_child_runnable_config.reset(token)
    other = config.var_child_runnable_config.set({"configurable": {"thread_id": "t"}})
    try:
        assert client.prompts.get("prm_planner@prod").resolved_from == ResolvedFrom("prod", 1)
    finally:
        config.var_child_runnable_config.reset(other)


def test_pin_resolves_alias_once(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    pinned = client.prompts.pin(["prm_planner@prod", "prm_writer:1", "prm_planner@prod"])
    assert isinstance(pinned, PinnedRefs)
    assert list(pinned) == ["prm_planner@prod", "prm_writer:1"]
    assert len(pinned) == 2
    assert sum(request.url.path == "/v1/prompts/resolve" for request in server.requests) == 1
    engine.publish("prm_planner", planner_v2(server), parent_version=1, change_message="two")
    engine.move_alias("prm_planner", "prod", version=2, expected_generation=1)
    assert pinned["prm_planner@prod"].ref.version == 1
    assert pinned["prm_planner@prod"].resolved_from == ResolvedFrom("prod", 1)
    with pytest.raises(TypeError):
        client.prompts.pin("prm_planner@prod")


def test_cross_workspace_uri_never_reaches_the_server(
    client: Client, server: FakePromptServer
) -> None:
    foreign = PromptUri(OTHER_WORKSPACE, "prm_planner", 1)
    for ref in (str(foreign), foreign):
        with pytest.raises(PromptRefError) as raised:
            client.prompts.get(ref)
        assert raised.value.code == "prompt_ref_cross_workspace"
    assert server.requests == []


def test_workspace_is_learned_once_and_checked(server: FakePromptServer) -> None:
    learned = make_client(server, workspace_id=None)
    assert learned.workspace_id is None
    learned.prompts.get("prm_writer:1")
    learned.prompts.get("prm_planner:1")
    assert learned.workspace_id == WORKSPACE
    assert learned.whoami()["api_key_scopes"] == ["write"]
    assert server.paths().count("GET /v1/whoami") == 1
    mismatched = make_client(server, workspace_id=OTHER_WORKSPACE)
    with pytest.raises(PromptRefError) as canonical:
        mismatched.prompts.get("prm_writer:1")
    assert canonical.value.code == "workspace_mismatch"
    with pytest.raises(PromptRefError) as identity:
        mismatched.whoami()
    assert identity.value.code == "workspace_mismatch"
    for call in (lambda: mismatched.workspace_id, mismatched.whoami):
        with pytest.raises(PromptRefError):
            call()
    with pytest.raises(ValueError):
        Client(workspace_id="NOT-A-UUID")

    def no_org(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["org_id"] = "nope"
        return payload

    server.rewrite = no_org
    with pytest.raises(ApiError) as invalid:
        make_client(server, workspace_id=None).whoami()
    assert invalid.value.code == "invalid_response"


def test_create_publish_list_and_versions(client: Client, server: FakePromptServer) -> None:
    created = client.prompts.create(
        "prm_notes", name="Notes", kind="text", description="Notes", tags=["support"]
    )
    assert body_of(server) == {
        "prompt_id": "prm_notes",
        "kind": "text",
        "name": "Notes",
        "tags": ["support"],
        "description": "Notes",
    }
    assert isinstance(created, PromptSummary)
    assert (created.metadata_revision, created.latest_version) == (1, None)
    content = text_content("Note on {topic}.", {"topic": required()})
    first = client.prompts.publish(
        "prm_notes",
        content,
        parent_version=None,
        change_message="first",
        variable_descriptions={"topic": "What the note covers."},
    )
    assert body_of(server) == {
        "content": content,
        "parent_version": None,
        "change_message": "first",
        "variable_descriptions": {"topic": "What the note covers."},
    }
    assert server.requests[-1].headers.get("idempotency-key") is None
    assert first.ref == PromptVersionRef("prm_notes", 1)
    again = client.prompts.publish(
        "prm_notes", content, parent_version=None, change_message="first"
    )
    assert again.content_digest == first.content_digest
    assert "variable_descriptions" not in body_of(server)
    count = len(server.requests)
    assert client.prompts.get("prm_notes:1").ref.version == 1
    assert len(server.requests) == count
    second = client.prompts.publish(
        "prm_planner", planner_v2(server), parent_version=1, change_message="two"
    )
    assert second.ref.version == 2
    assert set(second.fragments) == {"safety"}
    assert server.paths()[-1] == "GET /v1/prompts/prm_planner/versions/2"
    page = client.prompts.list(query="prm", tags=["support"], kind="text", limit=1)
    assert dict(server.requests[-1].url.params) == {
        "status": "active",
        "limit": "1",
        "q": "prm",
        "kind": "text",
        "tags": "support",
    }
    assert isinstance(page, Page)
    assert [item.prompt_id for item in page.items] == ["prm_notes"]
    assert page.next_cursor is None
    first_page = client.prompts.list(limit=2)
    assert first_page.next_cursor == "2"
    rest = client.prompts.list(limit=2, cursor=first_page.next_cursor, status="all")
    assert [item.prompt_id for item in first_page.items + rest.items] == [
        "prm_notes",
        "prm_planner",
        "prm_safety",
        "prm_writer",
    ]
    versions = client.prompts.versions.list("prm_planner", limit=1)
    assert [item.version for item in versions.items] == [2]
    assert versions.next_cursor == "1"
    older = client.prompts.versions.list("prm_planner", cursor=versions.next_cursor)
    assert [item.version for item in older.items] == [1]


def test_render_returns_the_rendered_hash(client: Client) -> None:
    text = client.prompts.render("prm_writer:1", {"topic": "tides"})
    assert isinstance(text, RenderResult)
    assert (text.kind, text.text, text.messages) == ("text", "Write about tides.", None)
    assert text.ref == PromptVersionRef("prm_writer", 1)
    assert text.rendered_hash.startswith("sha256:")
    chat = client.prompts.render("prm_planner:1", {"customer": "Acme", "question": "Why?"})
    assert chat.kind == "chat"
    assert chat.messages is not None
    assert len(chat.messages) == 2


def test_bindings_create_get_and_counterfactual(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    root, child = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    key = thread_key(WORKSPACE, "thread-1")
    binding, bundle, created = client.bindings.create(
        AGENT, thread_key=key, scope="thread", channel="production"
    )
    assert body_of(server) == {
        "thread_key": key,
        "scope": "thread",
        "selector": {"channel": "production"},
        "runtime_client": {
            "sdk": "agenomic-python",
            "sdk_version": __version__,
            "adapter": None,
            "adapter_version": None,
        },
        "include": ["artifacts"],
    }
    assert "idempotency-key" not in server.requests[-1].headers
    assert created
    assert binding.resolved_from == {"channel": "production", "generation": 1}
    assert bundle.prompt_manifest_digest == binding.prompt_manifest_digest
    assert bundle.version("planner.instructions").ref == PromptVersionRef("prm_planner", 1)
    assert bundle.version("writer.response", agent_id=CHILD).ref.prompt_id == "prm_writer"
    again, _, created_again = client.bindings.create(
        AGENT,
        thread_key=key,
        scope="thread",
        channel="production",
        child_selectors={CHILD: {"channel": "production"}},
        expect_manifest_digest=binding.prompt_manifest_digest,
        runtime_client={"adapter": "langgraph", "adapter_version": "1.2.11"},
    )
    sent = body_of(server)
    assert sent["child_selectors"] == {CHILD: {"channel": "production"}}
    assert sent["expect"] == {"prompt_manifest_digest": binding.prompt_manifest_digest}
    assert sent["runtime_client"]["adapter"] == "langgraph"
    assert not created_again
    assert again.binding_id == binding.binding_id
    fetched, fetched_bundle = client.bindings.get(AGENT, binding.binding_id)
    assert server.requests[-1].url.params["include"] == "artifacts"
    assert fetched == binding
    assert fetched_bundle.prompt_bundle_digest == bundle.prompt_bundle_digest
    cf_key = "cf:sha256:" + "4" * 64
    counterfactual = client.bindings.counterfactual(
        AGENT, binding.binding_id, thread_key=cf_key, release_id=root
    )
    assert body_of(server) == {"thread_key": cf_key, "selector": {"release_id": root}}
    assert counterfactual.parent_binding_id == binding.binding_id
    for selector in ({}, {"channel": "production", "release_id": root}):
        with pytest.raises(ValueError):
            client.bindings.create(AGENT, thread_key=key, scope="thread", **selector)
    with pytest.raises(ValueError):
        client.bindings.create(
            AGENT, thread_key=key, scope="thread", channel="production", runtime_client={"x": "y"}
        )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p, root, child: p["binding"].update(agent_id=OTHER_AGENT), "binding_mismatch"),
        (lambda p, root, child: p["binding"].update(thread_key="other"), "binding_mismatch"),
        (lambda p, root, child: p["binding"].update(release_id=child), "binding_mismatch"),
        (
            lambda p, root, child: p["binding"]["children"][CHILD].update(
                prompt_manifest_digest="sha256:" + "0" * 64
            ),
            "manifest_digest_mismatch",
        ),
        (lambda p, root, child: p.pop("artifacts"), "invalid_response"),
        (lambda p, root, child: p.update(created="yes"), "invalid_response"),
        (lambda p, root, child: p["binding"].pop("binding_id"), "invalid_response"),
    ],
)
def test_binding_answers_are_checked(
    server: FakePromptServer, mutate: Callable[[dict[str, Any], str, str], Any], code: str
) -> None:
    root, child = release_with_child(server.engine)

    def rewrite(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        mutate(payload, root, child)
        return payload

    server.rewrite = rewrite
    with pytest.raises(ApiError) as raised:
        make_client(server).bindings.create(
            AGENT, thread_key=thread_key(WORKSPACE, "t"), scope="thread", release_id=root
        )
    assert raised.value.code == code


def test_channels_are_read_only(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    engine.move_channel(AGENT, "staging", root, expected_generation=0)
    channels = client.channels.list(AGENT)
    assert [channel.name for channel in channels] == ["production", "staging"]
    assert all(isinstance(channel, Channel) for channel in channels)
    production = client.channels.get(AGENT, "production")
    assert (production.generation, production.protected, production.release_id) == (1, True, root)
    assert production.release is not None
    assert production.release["status"] == "production"
    assert client.channels.get(OTHER_AGENT, "production").materialized is False
    history = client.channels.history(AGENT, "production")
    assert isinstance(history[0], ChannelEvent)
    assert (history[0].action, history[0].to_release_id) == ("promote", root)
    assert client.channels.history(AGENT, "production", after=1) == []
    assert server.requests[-1].url.params["after"] == "1"
    preview = client.channels.move_preview(AGENT, "production", release_id=root)
    assert isinstance(preview, ChannelMovePreview)
    assert preview.actions["promote"] is False
    assert dict(server.requests[-1].url.params) == {"action": "promote", "release_id": root}
    client.channels.move_preview(AGENT, "production", action="rollback", release_id=root)
    assert dict(server.requests[-1].url.params) == {"action": "rollback", "to_release_id": root}
    with pytest.raises(ValueError):
        client.channels.move_preview(AGENT, "production")
    with pytest.raises(ValueError):
        client.channels.move_preview(AGENT, "production", action="delete", release_id=root)
    assert not hasattr(client.channels, "promote")
    assert not hasattr(client.channels, "rollback")

    def stale(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "channel" in payload:
            payload["channel"]["generation"] = 9
        return payload

    server.rewrite = stale
    with pytest.raises(ApiError) as mismatch:
        client.channels.get(AGENT, "production")
    assert mismatch.value.code == "invalid_response"


def test_channel_history_follows_next_after(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    first, _ = release_with_child(engine)
    second = engine.create_release(AGENT, {"planner.instructions": "prm_planner:1"})
    for generation in range(120):
        target = first if generation % 2 == 0 else second
        engine.move_channel(AGENT, "staging", target, expected_generation=generation)
    history = client.channels.history(AGENT, "staging")
    assert [event.generation for event in history] == list(range(1, 121))
    pages = [request.url.params["after"] for request in server.requests]
    assert pages == ["0", "50", "100"]
    assert [
        event.generation for event in client.channels.history(AGENT, "staging", after=110)
    ] == list(range(111, 121))

    def stuck(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["next_after"] = int(request.url.params["after"])
        return payload

    server.rewrite = stuck
    with pytest.raises(ApiError) as raised:
        client.channels.history(AGENT, "staging")
    assert raised.value.code == "invalid_response"


def test_resolve_agent_and_export_bundle(
    client: Client, server: FakePromptServer, tmp_path: Path
) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    resolved = client.prompts.resolve_agent(AGENT, channel="production")
    assert dict(server.requests[-1].url.params) == {"channel": "production"}
    assert not resolved.signed
    assert resolved.release_id == root
    target = tmp_path / "bundle.json"
    exported = client.prompts.export_bundle(
        AGENT, channel="production", expires_in_days=7, path=target
    )
    assert server.paths()[-2:] == [
        f"GET /v1/agents/{AGENT}/prompt-bundle",
        f"GET /v1/signing-keys/{server.signer.key_id}",
    ]
    assert dict(server.requests[-2].url.params) == {"channel": "production", "expires_in_days": "7"}
    assert exported.signed
    assert exported.governance is not None
    assert exported.governance["approved"] is True
    trust = BundleTrust.from_pems({server.signer.key_id: server.signer.public_pem()})
    reloaded = PromptBundle.load(
        target, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT, trust=trust
    )
    assert reloaded.prompt_bundle_digest == exported.prompt_bundle_digest
    count = len(server.requests)
    pinned = client.prompts.export_bundle(AGENT, release_id=root, trust=trust)
    assert pinned.release_id == root
    assert len(server.requests) == count + 1
    stranger = SigningKey.generate("orgkey_other")
    with pytest.raises(PromptIntegrityError) as untrusted:
        client.prompts.export_bundle(
            AGENT,
            release_id=root,
            trust=BundleTrust.from_pems({stranger.key_id: stranger.public_pem()}),
        )
    assert untrusted.value.code == "bundle_untrusted_key"
    candidate = engine.create_release(
        AGENT, {"planner.instructions": "prm_planner:1"}, status="awaiting_approval"
    )
    engine.move_channel(AGENT, "staging", candidate, expected_generation=0)
    staged = client.prompts.export_bundle(AGENT, channel="staging")
    assert staged.governance is not None
    assert staged.governance["approved"] is False
    with pytest.raises(PromptBindingError) as ungoverned:
        client.prompts.resolve_agent(
            AGENT, release_id=engine.create_release(AGENT, {}, status="awaiting_approval")
        )
    assert ungoverned.value.code == "session_required"

    def unsigned(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.endswith("/prompt-bundle"):
            payload.pop("issuer")
        if request.url.path.startswith("/v1/signing-keys/"):
            payload["key_id"] = "orgkey_swapped"
        return payload

    server.rewrite = unsigned
    with pytest.raises(PromptIntegrityError) as missing:
        client.prompts.export_bundle(AGENT, channel="production")
    assert missing.value.code == "bundle_signature_invalid"

    def swapped(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.startswith("/v1/signing-keys/"):
            payload["key_id"] = "orgkey_swapped"
        return payload

    server.rewrite = swapped
    with pytest.raises(ApiError) as invalid:
        client.prompts.export_bundle(AGENT, channel="production")
    assert invalid.value.code == "invalid_response"

    def other_agent(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["agent_id"] = OTHER_AGENT
        return payload

    server.rewrite = other_agent
    with pytest.raises(ApiError) as foreign:
        client.prompts.resolve_agent(AGENT, channel="production")
    assert foreign.value.code == "invalid_response"


def test_local_mode_uses_the_engine(client: Client) -> None:
    local = Client(workspace_id=WORKSPACE)
    engine = local.prompts.local
    assert engine.workspace_id == local.workspace_id == WORKSPACE
    assert not local.is_cloud
    created = local.prompts.create("prm_writer", name="Writer", kind="text")
    assert isinstance(created, PromptSummary)
    content = text_content("Write about {topic}.", {"topic": required()})
    version = local.prompts.publish(
        "prm_writer", content, parent_version=None, change_message="first"
    )
    assert local.prompts.get("prm_writer:1").content_digest == version.content_digest
    assert local.prompts.versions.get("prm_writer", 1).ref.version == 1
    assert local.prompts.render("prm_writer:1", {"topic": "tides"}).text == "Write about tides."
    draft = local.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    assert local.prompts.drafts.get("prm_writer").revision == draft.revision == 1
    alias = local.prompts.aliases.move("prm_writer", "prod", version=1, expected_generation=0)
    assert isinstance(alias, PromptAlias)
    assert local.prompts.aliases.get("prm_writer", "prod").generation == 1
    assert local.prompts.get("prm_writer@prod").resolved_from == ResolvedFrom("prod", 1)
    release = engine.create_release(AGENT, {"writer.main": "prm_writer:1"})
    engine.move_channel(AGENT, "production", release, expected_generation=0)
    binding, bundle, created_binding = local.bindings.create(
        AGENT, thread_key=thread_key(WORKSPACE, "t"), scope="thread", channel="production"
    )
    assert created_binding
    assert bundle.version("writer.main").ref.prompt_id == "prm_writer"
    assert local.bindings.get(AGENT, binding.binding_id)[0] == binding
    assert local.channels.get(AGENT, "production").generation == 1
    assert local.channels.history(AGENT, "production")[0].action == "promote"
    assert local.channels.history(AGENT, "production", after=1) == []
    assert local.prompts.resolve_agent(AGENT, channel="production").release_id == release
    unsupported: list[Callable[[], Any]] = [
        local.prompts.list,
        lambda: local.prompts.versions.list("prm_writer"),
        lambda: local.prompts.export_bundle(AGENT, channel="production"),
        lambda: local.channels.list(AGENT),
        lambda: local.channels.move_preview(AGENT, "production", release_id=release),
        lambda: local.bindings.counterfactual(
            AGENT, binding.binding_id, thread_key="cf:x", release_id=release
        ),
        lambda: local.bindings.create(
            AGENT,
            thread_key="t2",
            scope="thread",
            channel="production",
            child_selectors={CHILD: {"channel": "production"}},
        ),
        local.whoami,
        lambda: client.prompts.local,
    ]
    for call in unsupported:
        with pytest.raises(ApiError) as raised:
            call()
        assert raised.value.code == "cloud_required"


def test_from_env_and_lifecycle(
    server: FakePromptServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENOMIC_ENDPOINT", BASE + "/")
    monkeypatch.setenv("AGENOMIC_API_KEY", "env-key")
    monkeypatch.setenv("AGENOMIC_WORKSPACE_ID", WORKSPACE)
    monkeypatch.setenv("AGENOMIC_PROMPT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENOMIC_TIMEOUT", "5")
    configured = Client.from_env(transport=server.transport())
    assert (configured.base_url, configured.api_key, configured.workspace_id) == (
        BASE,
        "env-key",
        WORKSPACE,
    )
    assert configured._timeout == 5.0
    configured.prompts.get("prm_writer:1")
    assert server.requests[-1].headers["authorization"] == "Bearer env-key"
    assert (tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1").is_dir()
    assert Client.from_env(api_key="explicit", transport=server.transport()).api_key == "explicit"
    monkeypatch.setenv("AGENOMIC_TIMEOUT", "soon")
    with pytest.raises(ValueError):
        Client.from_env()
    for variable in (
        "AGENOMIC_ENDPOINT",
        "AGENOMIC_API_KEY",
        "AGENOMIC_WORKSPACE_ID",
        "AGENOMIC_PROMPT_CACHE_DIR",
        "AGENOMIC_TIMEOUT",
    ):
        monkeypatch.delenv(variable)
    assert not Client.from_env().is_cloud
    with make_client(server) as managed:
        managed.prompts.get("prm_writer:1")
        http = pool_for(managed).sync()
    assert http.is_closed


async def test_async_twins(server: FakePromptServer, tmp_path: Path) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    async with make_client(server, workspace_id=None) as client:
        assert (await client.awhoami())["org_id"] == WORKSPACE
        version = await client.prompts.aget("prm_planner:1")
        assert (await client.prompts.aresolve("prm_planner@prod")).ref == version.ref
        assert len(await client.prompts.apin(["prm_planner@prod"])) == 1
        assert (
            await client.prompts.arender("prm_writer:1", {"topic": "t"})
        ).text == "Write about t."
        assert (await client.prompts.alist()).items
        assert (await client.prompts.versions.alist("prm_planner")).items[0].version == 1
        assert (await client.prompts.versions.aget("prm_writer", 1)).ref.version == 1
        created = await client.prompts.acreate("prm_async", name="Async", kind="text")
        assert created.prompt_id == "prm_async"
        published = await client.prompts.apublish(
            "prm_async", text_content("Hello."), parent_version=None, change_message="first"
        )
        assert published.ref.version == 1
        draft = await client.prompts.drafts.asave(
            "prm_async", text_content("Hi."), base_version=1, expected_revision=0
        )
        assert (await client.prompts.drafts.aget("prm_async")).revision == draft.revision
        assert (await client.prompts.aliases.aget("prm_planner", "prod")).generation == 1
        with pytest.raises(ApiError):
            await client.prompts.aliases.amove(
                "prm_planner", "prod", version=1, expected_generation=1
            )
        key = thread_key(WORKSPACE, "async")
        binding, _, created_binding = await client.bindings.acreate(
            AGENT, thread_key=key, scope="thread", channel="production"
        )
        assert created_binding
        assert (await client.bindings.aget(AGENT, binding.binding_id))[0] == binding
        child = await client.bindings.acounterfactual(
            AGENT, binding.binding_id, thread_key="cf:async", release_id=root
        )
        assert child.parent_binding_id == binding.binding_id
        assert (await client.prompts.aresolve_agent(AGENT, channel="production")).release_id == root
        exported = await client.prompts.aexport_bundle(
            AGENT, channel="production", path=tmp_path / "async.json"
        )
        assert exported.signed
        assert (tmp_path / "async.json").is_file()
        assert [channel.name for channel in await client.channels.alist(AGENT)] == ["production"]
        assert (await client.channels.aget(AGENT, "production")).generation == 1
        assert len(await client.channels.ahistory(AGENT, "production")) == 1
        preview = await client.channels.amove_preview(AGENT, "production", release_id=root)
        assert preview.agent_id == AGENT
