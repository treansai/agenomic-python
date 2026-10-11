from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pytest
from prompt_fakes import (
    AGENT,
    CHILD,
    WORKSPACE,
    FakePromptServer,
    release_with_child,
    seeded_engine,
)

from agenomic import _transport
from agenomic._client import Client
from agenomic.exceptions import ApiError
from agenomic.prompts import (
    PromptBindingError,
    PromptCache,
    PromptIntegrityError,
    PromptRefError,
    RegistryUnavailableError,
    thread_key,
)
from agenomic.prompts.authority import (
    OUTAGE_CACHED_BINDING,
    CloudBindingAuthority,
    LocalBindingAuthority,
    counters,
)

BASE = "https://api.test"
PRODUCTION = {"channel": "production"}


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(_transport, "_sleep", recorded.append)

    async def asleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(_transport, "_asleep", asleep)
    return recorded


@pytest.fixture
def server() -> FakePromptServer:
    server = FakePromptServer(seeded_engine())
    root, _ = release_with_child(server.engine)
    server.engine.move_channel(AGENT, "production", root, expected_generation=0)
    return server


def make_client(server: FakePromptServer, cache: Optional[PromptCache] = None) -> Client:
    return Client(
        api_key="key",
        base_url=BASE,
        transport=server.transport(),
        workspace_id=WORKSPACE,
        prompt_cache=cache,
    )


def authority_for(
    server: FakePromptServer, cache: Optional[PromptCache] = None
) -> CloudBindingAuthority:
    return CloudBindingAuthority(make_client(server, cache))


def production_release(server: FakePromptServer) -> str:
    return str(server.engine.get_channel(AGENT, "production")["release_id"])


def test_new_thread_outage_fails_closed(server: FakePromptServer) -> None:
    authority = authority_for(server)
    client = make_client(server)
    before = counters()[OUTAGE_CACHED_BINDING]
    for mode, status in (("transport_error", 0), ("http_503", 503)):
        server.outage = mode
        with pytest.raises(RegistryUnavailableError) as raised:
            authority.create_or_get(AGENT, thread_key(WORKSPACE, "new"), "thread", PRODUCTION)
        assert raised.value.status == status
        assert raised.value.code == "registry_unavailable"
    assert {request.url.path for request in server.requests} == {f"/v1/agents/{AGENT}/bindings"}
    for call in (
        lambda: client.prompts.get("prm_planner:1"),
        lambda: client.prompts.resolve_agent(AGENT, channel="production"),
        lambda: client.bindings.create(
            AGENT, thread_key=thread_key(WORKSPACE, "new"), scope="thread", channel="production"
        ),
    ):
        with pytest.raises(RegistryUnavailableError):
            call()
    assert counters()[OUTAGE_CACHED_BINDING] == before


def test_existing_thread_outage_uses_cached_binding_only(
    server: FakePromptServer, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    authority = authority_for(server, PromptCache(tmp_path))
    key = thread_key(WORKSPACE, "existing")
    binding, bundle, created = authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    assert created
    before = counters()[OUTAGE_CACHED_BINDING]
    server.outage = "transport_error"
    with caplog.at_level(logging.WARNING, logger="agenomic.prompts"):
        cached, cached_bundle, cached_created = authority.create_or_get(
            AGENT, key, "thread", PRODUCTION
        )
    assert cached.binding_id == binding.binding_id
    assert cached_created is False
    assert cached_bundle.prompt_bundle_digest == bundle.prompt_bundle_digest
    assert cached_bundle.version("writer.response", agent_id=CHILD).ref.prompt_id == "prm_writer"
    warnings = [
        record
        for record in caplog.records
        if record.name == "agenomic.prompts" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert binding.binding_id in warnings[0].getMessage()
    assert counters()[OUTAGE_CACHED_BINDING] == before + 1
    for thread, scope, selector in (
        (thread_key(WORKSPACE, "other"), "thread", PRODUCTION),
        (key, "execution", PRODUCTION),
        (key, "thread", {"release_id": production_release(server)}),
        (key, "thread", {"channel": "staging"}),
    ):
        with pytest.raises(RegistryUnavailableError):
            authority.create_or_get(AGENT, thread, scope, selector)
    server.outage = "http_503"
    restarted = authority_for(server, PromptCache(tmp_path))
    again, _, again_created = restarted.create_or_get(AGENT, key, "thread", PRODUCTION)
    assert (again.binding_id, again_created) == (binding.binding_id, False)
    assert counters()[OUTAGE_CACHED_BINDING] == before + 2


def test_403_evicts_and_never_uses_cache(server: FakePromptServer, tmp_path: Path) -> None:
    cache = PromptCache(tmp_path)
    authority = authority_for(server, cache)
    key = thread_key(WORKSPACE, "revoked")
    for status_mode, status in (("http_403", 403), ("http_404", 404)):
        server.outage = None
        authority.create_or_get(AGENT, key, "thread", PRODUCTION)
        assert cache.get_binding(WORKSPACE, AGENT, key) is not None
        server.outage = status_mode
        with pytest.raises(ApiError) as raised:
            authority.create_or_get(AGENT, key, "thread", PRODUCTION)
        assert type(raised.value) is ApiError
        assert raised.value.status == status
        assert cache.get_binding(WORKSPACE, AGENT, key) is None
        assert PromptCache(tmp_path).get_binding(WORKSPACE, AGENT, key) is None
        server.outage = "transport_error"
        with pytest.raises(RegistryUnavailableError):
            authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    server.outage = None
    authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    with pytest.raises(PromptBindingError) as conflict:
        authority.create_or_get(AGENT, key, "execution", PRODUCTION)
    assert conflict.value.code == "execution_binding_conflict"
    assert cache.get_binding(WORKSPACE, AGENT, key) is not None


@pytest.mark.parametrize(("mode", "status"), [("http_403", 403), ("http_404", 404)])
def test_resolution_refusal_evicts_the_cached_binding(
    server: FakePromptServer, mode: str, status: int
) -> None:
    key = thread_key(WORKSPACE, "revoked-resolution")
    binding, bundle, _ = authority_for(server).create_or_get(AGENT, key, "thread", PRODUCTION)
    cache = PromptCache()
    cache.put_binding(WORKSPACE, binding)
    authority = authority_for(server, cache)
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError):
        authority.resolution(binding)
    assert cache.get_binding(WORKSPACE, AGENT, key) is not None
    server.outage = mode
    with pytest.raises(ApiError) as raised:
        authority.resolution(binding)
    assert raised.value.status == status
    assert cache.get_binding(WORKSPACE, AGENT, key) is None
    cache.put_closure(WORKSPACE, bundle.closure())
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError):
        authority.create_or_get(AGENT, key, "thread", PRODUCTION)


def test_alias_outage_fails(server: FakePromptServer) -> None:
    client = make_client(server)
    server.engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    assert client.prompts.get("prm_planner@prod").ref.version == 1
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError):
        client.prompts.get("prm_planner@prod")
    with pytest.raises(RegistryUnavailableError):
        client.prompts.pin(["prm_planner@prod"])
    assert client.prompts.get("prm_planner:1").ref.version == 1
    with pytest.raises(RegistryUnavailableError):
        client.prompts.get("prm_writer:1")


def test_no_inline_string_fallback(server: FakePromptServer) -> None:
    client = make_client(server)
    for text, code in (
        ("Plan for {customer}", "prompt_ref_invalid"),
        ("prm_planner", "prompt_ref_unversioned"),
        ("", "prompt_ref_invalid"),
    ):
        with pytest.raises(PromptRefError) as raised:
            client.prompts.get(text)
        assert raised.value.code == code
    assert server.requests == []
    authority = CloudBindingAuthority(client)
    key = thread_key(WORKSPACE, "slots")
    authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    server.outage = "http_503"
    _, bundle, _ = authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    with pytest.raises(PromptBindingError) as missing:
        bundle.version("planner.unknown")
    assert missing.value.code == "slot_not_in_manifest"


def test_cache_conflict_during_outage_fails_closed(
    server: FakePromptServer, tmp_path: Path
) -> None:
    key = thread_key(WORKSPACE, "tampered")
    binding, _, _ = authority_for(server, PromptCache(tmp_path)).create_or_get(
        AGENT, key, "thread", PRODUCTION
    )
    digest = binding.prompt_manifest_digest.removeprefix("sha256:")
    path = tmp_path / "v1" / WORKSPACE / "closures" / f"{digest}.json"
    closure: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    entry = closure["prompts"]["prm_writer:1"]
    entry["content"]["body"] = "Write anything."
    path.write_text(json.dumps(closure), encoding="utf-8")
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError) as raised:
        authority_for(server, PromptCache(tmp_path)).create_or_get(AGENT, key, "thread", PRODUCTION)
    assert isinstance(raised.value.__cause__, PromptIntegrityError)


def test_resolution_reads_the_cache_then_the_registry(server: FakePromptServer) -> None:
    authority = authority_for(server)
    binding, bundle, _ = authority.create_or_get(
        AGENT, thread_key(WORKSPACE, "resolution"), "thread", PRODUCTION
    )
    count = len(server.requests)
    assert authority.resolution(binding).prompt_bundle_digest == bundle.prompt_bundle_digest
    assert len(server.requests) == count
    fresh = authority_for(server)
    assert fresh.get(AGENT, binding.binding_id) == binding
    assert server.paths()[-1] == f"GET /v1/agents/{AGENT}/bindings/{binding.binding_id}"
    other = authority_for(server)
    assert other.resolution(binding).prompt_manifest_digest == binding.prompt_manifest_digest
    assert server.paths()[-1] == f"GET /v1/agents/{AGENT}/bindings/{binding.binding_id}"
    forged = binding.model_copy(update={"prompt_manifest_digest": "sha256:" + "1" * 64})
    with pytest.raises(PromptBindingError) as raised:
        authority_for(server).resolution(forged)
    assert raised.value.code == "binding_mismatch"
    with pytest.raises(ValueError):
        authority.create_or_get(
            AGENT, "t", "thread", {"channel": "production", "release_id": binding.release_id}
        )
    with pytest.raises(ValueError):
        CloudBindingAuthority(Client())


async def test_async_outage_uses_cached_binding(server: FakePromptServer, tmp_path: Path) -> None:
    authority = authority_for(server, PromptCache(tmp_path))
    key = thread_key(WORKSPACE, "async")
    binding, _, created = await authority.acreate_or_get(AGENT, key, "thread", PRODUCTION)
    assert created
    assert (await authority.aget(AGENT, binding.binding_id)) == binding
    assert (await authority.aresolution(binding)).release_id == binding.release_id
    server.outage = "transport_error"
    cached, _, cached_created = await authority.acreate_or_get(AGENT, key, "thread", PRODUCTION)
    assert (cached.binding_id, cached_created) == (binding.binding_id, False)
    server.outage = "http_403"
    with pytest.raises(ApiError):
        await authority.acreate_or_get(AGENT, key, "thread", PRODUCTION)
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError):
        await authority.acreate_or_get(AGENT, key, "thread", PRODUCTION)


async def test_local_binding_authority() -> None:
    engine = seeded_engine()
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    authority = LocalBindingAuthority(engine)
    key = thread_key(WORKSPACE, "local")
    binding, bundle, created = authority.create_or_get(AGENT, key, "thread", PRODUCTION)
    assert created
    assert bundle.version("planner.instructions").ref.prompt_id == "prm_planner"
    again, _, again_created = await authority.acreate_or_get(AGENT, key, "thread", PRODUCTION)
    assert (again.binding_id, again_created) == (binding.binding_id, False)
    assert authority.get(AGENT, binding.binding_id) == binding
    assert (await authority.aget(AGENT, binding.binding_id)) == binding
    assert authority.resolution(binding).prompt_manifest_digest == binding.prompt_manifest_digest
    assert (await authority.aresolution(binding)).release_id == root
    forged = binding.model_copy(update={"prompt_manifest_digest": "sha256:" + "1" * 64})
    with pytest.raises(PromptBindingError):
        authority.resolution(forged)
    with pytest.raises(ApiError) as raised:
        authority.create_or_get(
            AGENT, key, "thread", PRODUCTION, child_selectors={CHILD: {"channel": "production"}}
        )
    assert raised.value.code == "cloud_required"
    with pytest.raises(ValueError):
        authority.create_or_get(AGENT, key, "thread", {})
