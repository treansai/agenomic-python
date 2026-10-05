from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph_world import LogState, World, plan_node, supervise_node, thread, two_node_graph
from prompt_fakes import AGENT, WORKSPACE

from agenomic import _transport
from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.integrations.langgraph_binding import (
    LocalBindingStore,
    prompts_for,
)
from agenomic.prompts import (
    BundleTrust,
    PromptBindingError,
    PromptBundle,
    PromptIntegrityError,
    RegistryUnavailableError,
    execution_key,
    thread_key,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_transport, "_sleep", lambda delay: None)

    async def asleep(delay: float) -> None:
        return None

    monkeypatch.setattr(_transport, "_asleep", asleep)


def executions(key: str, thread_id: str = "conversation") -> RunnableConfig:
    return thread(thread_id, agenomic_execution_key=key)


@pytest.mark.parametrize("mode", ["cloud", "local"])
def test_promotion_affects_new_threads_only(mode: str) -> None:
    if mode == "cloud":
        world = World.create()
        managed = world.bind(two_node_graph())
    else:
        world, client = World.local()
        managed = world.bind(two_node_graph(), client=client)
    assert managed.invoke({"log": []}, thread("old"))["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    world.promote("v2")
    assert managed.invoke({"log": []}, thread("old"))["log"][-2:] == [
        "plan:PLAN v1",
        "sup:SUP v1",
    ]
    assert managed.invoke({"log": []}, thread("new"))["log"] == ["plan:PLAN v2", "sup:SUP v2"]


def test_in_flight_execution_keeps_admission_pin() -> None:
    world = World.create()

    def plan_then_promote(state: LogState, config: RunnableConfig) -> LogState:
        out = plan_node(state, config)
        world.promote("v2")
        return out

    builder = StateGraph(LogState)
    builder.add_node("plan", plan_then_promote)
    builder.add_node("supervise", supervise_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "supervise")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    assert managed.invoke({"log": []}, thread("inflight"))["log"] == [
        "plan:PLAN v1",
        "sup:SUP v1",
    ]
    assert managed.invoke({"log": []}, thread("after"))["log"] == ["plan:PLAN v2", "sup:SUP v2"]


def test_promotion_during_astream_keeps_pin() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())

    async def run() -> list[Any]:
        chunks = []
        async for chunk in managed.astream({"log": []}, thread("astream"), stream_mode="updates"):
            chunks.append(chunk)
            if len(chunks) == 1:
                world.promote("v2")
        return chunks

    chunks = asyncio.run(run())
    assert chunks == [{"plan": {"log": ["plan:PLAN v1"]}}, {"supervise": {"log": ["sup:SUP v1"]}}]


def test_execution_scope_new_binding_per_execution() -> None:
    world = World.create()
    managed = world.bind(two_node_graph(), pin_scope="execution")
    first = managed.invoke({"log": []}, executions("request-1"))
    world.promote("v2")
    second = managed.invoke({"log": []}, executions("request-2"))
    assert first["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    assert second["log"][-2:] == ["plan:PLAN v2", "sup:SUP v2"]
    keys = [post["thread_key"] for post in world.binding_posts()]
    assert keys == [execution_key(WORKSPACE, "request-1"), execution_key(WORKSPACE, "request-2")]
    assert all(post["scope"] == "execution" for post in world.binding_posts())


def test_execution_key_required_without_key() -> None:
    world = World.create()
    managed = world.bind(two_node_graph(), pin_scope="execution")
    for config in (thread("t"), executions("")):
        with pytest.raises(PromptBindingError) as raised:
            managed.invoke({"log": []}, config)
        assert raised.value.code == "execution_key_required"
    with pytest.raises(PromptBindingError) as batched:
        managed.batch([{"log": []}], [thread("b")])
    assert batched.value.code == "execution_key_required"
    assert world.binding_posts() == []


def test_execution_scope_retry_after_promotion_keeps_original_pin() -> None:
    world = World.create()
    attempts: list[str] = []

    def flaky(state: LogState, config: RunnableConfig) -> LogState:
        text = prompts_for(config).render_text("planner.instructions")
        attempts.append(text)
        if len(attempts) == 1:
            world.promote("v2")
            raise RuntimeError("transient model error")
        return {"log": [text]}

    builder = StateGraph(LogState)
    builder.add_node("flaky", flaky)
    builder.add_edge(START, "flaky")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()), pin_scope="execution")
    retrying = managed.with_retry(stop_after_attempt=2, wait_exponential_jitter=False)
    assert retrying.invoke({"log": []}, thread("retry"))["log"] == ["PLAN v1"]
    assert attempts == ["PLAN v1", "PLAN v1"]
    keys = {post["thread_key"] for post in world.binding_posts()}
    assert len(keys) == 1
    outputs = retrying.batch([{"log": []}, {"log": []}], [thread("b1"), thread("b2")])
    assert [out["log"] for out in outputs] == [["PLAN v2"], ["PLAN v2"]]
    assert len({post["thread_key"] for post in world.binding_posts()}) == 3
    supplied = retrying.invoke({"log": []}, executions("caller-key", "b3"))
    assert supplied["log"] == ["PLAN v2"]
    assert world.binding_posts()[-1]["thread_key"] == execution_key(WORKSPACE, "caller-key")
    async_out = asyncio.run(retrying.ainvoke({"log": []}, thread("b4")))
    assert async_out["log"] == ["PLAN v2"]
    async_batch = asyncio.run(retrying.abatch([{"log": []}], [thread("b5")]))
    assert async_batch[0]["log"] == ["PLAN v2"]
    thread_scoped = world.bind(two_node_graph()).with_retry(stop_after_attempt=2)
    assert thread_scoped.invoke({"log": []}, thread("plain"))["log"][-1] == "sup:SUP v2"


def test_binding_target_mismatch_production_vs_experiment() -> None:
    world = World.create()
    experiment = world.bind(two_node_graph(), channel=None, release_id=world.releases["v2"])
    assert experiment.invoke({"log": []}, thread("trial"))["log"][-1] == "sup:SUP v2"
    production = world.bind(two_node_graph())
    with pytest.raises(PromptBindingError) as conflict:
        production.invoke({"log": []}, thread("trial"))
    assert conflict.value.code == "execution_binding_conflict"

    def forge(request: httpx.Request, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.endswith("/bindings") and "binding" in payload:
            payload["binding"]["resolved_from"] = {"release_id": world.releases["v1"]}
        return payload

    world.server.rewrite = forge
    with pytest.raises(PromptBindingError) as mismatch:
        production.invoke({"log": []}, thread("forged"))
    assert mismatch.value.code == "binding_target_mismatch"


def test_new_thread_outage_fails_closed_and_existing_thread_continues() -> None:
    world = World.create()
    ran: list[str] = []

    def node(state: LogState, config: RunnableConfig) -> LogState:
        ran.append("node")
        return plan_node(state, config)

    builder = StateGraph(LogState)
    builder.add_node("plan", node)
    builder.add_edge(START, "plan")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    managed.invoke({"log": []}, thread("existing"))
    world.server.outage = "http_503"
    ran.clear()
    with pytest.raises(RegistryUnavailableError):
        managed.invoke({"log": []}, thread("brand-new"))
    assert ran == []
    assert managed.invoke({"log": []}, thread("existing"))["log"][-1] == "plan:PLAN v1"
    world.server.outage = "http_403"
    with pytest.raises(Exception) as refused:
        managed.invoke({"log": []}, thread("existing"))
    assert getattr(refused.value, "status", None) == 403
    world.server.outage = "http_503"
    with pytest.raises(RegistryUnavailableError):
        managed.invoke({"log": []}, thread("existing"))


def test_revalidate_never_uses_cached_binding() -> None:
    world = World.create()
    managed = world.bind(two_node_graph(), revalidate="never")
    managed.invoke({"log": []}, thread("cached"))
    posts = len(world.binding_posts())
    world.server.outage = "transport_error"
    assert managed.invoke({"log": []}, thread("cached"))["log"][-1] == "sup:SUP v1"
    assert len(world.binding_posts()) == posts
    local_world, client = World.local()
    local = local_world.bind(two_node_graph(), client=client, revalidate="never")
    local.invoke({"log": []}, thread("cached"))
    local_world.promote("v2")
    assert local.invoke({"log": []}, thread("cached"))["log"][-1] == "sup:SUP v1"


def export_pair(world: World, tmp_path: Path) -> tuple[Path, Path, SigningKey]:
    key = SigningKey.generate("orgkey_offline")
    first = world.export(tmp_path / "bundle-v1.json", key)
    world.promote("v2")
    second = world.export(tmp_path / "bundle-v2.json", key)
    return first, second, key


def test_offline_bundles_pin_threads_and_retained_bundles_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World.create()
    first, second, key = export_pair(world, tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline mode opened a socket")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    trust = BundleTrust({key.key_id: key.public_key()})
    saver = InMemorySaver()
    store = LocalBindingStore(tmp_path / "pins")
    options: dict[str, Any] = {
        "offline": True,
        "workspace_id": WORKSPACE,
        "trust": trust,
        "binding_store": store,
    }
    old = world.bind(two_node_graph(saver), bundle=first, **options)
    assert old.invoke({"log": []}, thread("old"))["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    retained = PromptBundle.load(
        first, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT, trust=trust
    )
    new = world.bind(two_node_graph(saver), bundle=second, retained_bundles=[retained], **options)
    assert new.invoke({"log": []}, thread("old"))["log"][-1] == "sup:SUP v1"
    assert new.invoke({"log": []}, thread("new"))["log"] == ["plan:PLAN v2", "sup:SUP v2"]
    digest = PromptBundle.load(
        first, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT, trust=trust
    ).prompt_bundle_digest
    pinned_pair = world.bind(
        two_node_graph(saver), bundle=second, retained_bundles=[(first, digest)], **options
    )
    assert pinned_pair.invoke({"log": []}, thread("old"))["log"][-1] == "sup:SUP v1"
    forgetful = world.bind(two_node_graph(saver), bundle=second, **options)
    with pytest.raises(PromptBindingError) as missing:
        forgetful.invoke({"log": []}, thread("old"))
    assert missing.value.code == "binding_mismatch"
    with pytest.raises(ValueError):
        world.bind(two_node_graph(saver), bundle=second, retained_bundles=[first], **options)


def test_offline_target_and_store_rules(tmp_path: Path) -> None:
    world = World.create()
    key = SigningKey.generate("orgkey_offline")
    path = world.export(tmp_path / "bundle.json", key)
    trust = BundleTrust({key.key_id: key.public_key()})
    common: dict[str, Any] = {"offline": True, "workspace_id": WORKSPACE, "trust": trust}
    for target in ({"channel": "staging"}, {"channel": None, "release_id": world.releases["v2"]}):
        with pytest.raises(PromptBindingError) as raised:
            world.bind(two_node_graph(), bundle=path, **common, **target)
        assert raised.value.code == "binding_target_mismatch"
    release = world.bind(
        two_node_graph(),
        bundle=path,
        channel=None,
        release_id=world.releases["v1"],
        **common,
        binding_store=LocalBindingStore(tmp_path / "pins"),
    )
    assert release.invoke({"log": []}, thread("r"))["log"][-1] == "sup:SUP v1"
    with pytest.raises(ValueError):
        world.bind(two_node_graph(), bundle=path, **common)
    stateless = StateGraph(LogState)
    stateless.add_node("plan", plan_node)
    stateless.add_edge(START, "plan")
    default_store = world.bind(stateless.compile(), bundle=path, **common)
    assert default_store.invoke({"log": []}, thread("s"))["log"] == ["plan:PLAN v1"]
    with pytest.raises(ValueError):
        world.bind(two_node_graph(), bundle=path, offline=True, workspace_id=WORKSPACE)
    digest = PromptBundle.load(
        path, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT, trust=trust
    ).prompt_bundle_digest
    pinned = world.bind(
        stateless.compile(),
        bundle=path,
        offline=True,
        workspace_id=WORKSPACE,
        expected_bundle_digest=digest,
    )
    assert pinned.invoke({"log": []}, thread("p"))["log"] == ["plan:PLAN v1"]
    other = "5a5a5a5a-1111-4222-8333-444455556666"
    with pytest.raises(PromptIntegrityError) as scope:
        world.bind(stateless.compile(), bundle=path, offline=True, workspace_id=other, trust=trust)
    assert scope.value.code == "bundle_scope_mismatch"


def test_offline_ungoverned_bundle_refused_without_opt_in(tmp_path: Path) -> None:
    world = World.create()
    world.engine.set_release_status(world.releases["v2"], "awaiting_approval")
    key = SigningKey.generate("orgkey_offline")
    path = world.export(tmp_path / "candidate.json", key, release_id=world.releases["v2"])
    trust = BundleTrust({key.key_id: key.public_key()})
    stateless = StateGraph(LogState)
    stateless.add_node("plan", plan_node)
    stateless.add_edge(START, "plan")
    options: dict[str, Any] = {
        "bundle": path,
        "offline": True,
        "workspace_id": WORKSPACE,
        "trust": trust,
        "channel": None,
        "release_id": world.releases["v2"],
    }
    with pytest.raises(PromptIntegrityError) as raised:
        world.bind(stateless.compile(), **options)
    assert raised.value.code == "bundle_ungoverned"
    allowed = world.bind(stateless.compile(), allow_ungoverned_bundle=True, **options)
    assert allowed.invoke({"log": []}, thread("c"))["log"] == ["plan:PLAN v2"]


def test_local_binding_store_crash_between_create_and_write(tmp_path: Path) -> None:
    world = World.create()
    key = SigningKey.generate("orgkey_offline")
    path = world.export(tmp_path / "bundle.json", key)
    trust = BundleTrust({key.key_id: key.public_key()})
    root = tmp_path / "pins"
    directory = root / WORKSPACE / AGENT
    directory.mkdir(parents=True)
    (directory / ".tmp-0123456789abcdef").write_bytes(b'{"schema": "agenomic.local_exec')
    store = LocalBindingStore(root)
    assert store.get(WORKSPACE, AGENT, thread_key(WORKSPACE, "t")) is None
    managed = world.bind(
        two_node_graph(),
        bundle=path,
        offline=True,
        workspace_id=WORKSPACE,
        trust=trust,
        binding_store=store,
    )
    assert managed.invoke({"log": []}, thread("t"))["log"][-1] == "sup:SUP v1"
    files = sorted(directory.glob("*.json"))
    assert len(files) == 1
    pinned = store.get(WORKSPACE, AGENT, thread_key(WORKSPACE, "t"))
    assert pinned is not None
    assert store.find(WORKSPACE, AGENT, pinned.binding_id) == pinned
    other = pinned.model_copy(update={"binding_id": "bnd_00000000000000000000000000"})
    winner, created = store.put(other)
    assert not created
    assert winner.binding_id == pinned.binding_id
    original = files[0].read_bytes()
    document = json.loads(original)
    document["binding"]["release_id"] = world.releases["v2"]
    files[0].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PromptBindingError) as tampered:
        managed.invoke({"log": []}, thread("t"))
    assert tampered.value.code == "binding_store_corrupt"
    assert tampered.value.details["reason"] == "digest_mismatch"
    files[0].write_bytes(original[: len(original) // 2])
    with pytest.raises(PromptBindingError) as truncated:
        managed.invoke({"log": []}, thread("t"))
    assert truncated.value.code == "binding_store_corrupt"
    assert truncated.value.details["path"] == str(files[0])
    memory = LocalBindingStore.in_memory()
    assert memory.directory is None
    stored, first = memory.put(pinned)
    again, second = memory.put(other)
    assert (first, second) == (True, False)
    assert again is stored
    assert memory.find(WORKSPACE, AGENT, pinned.binding_id) is stored
    assert memory.find(WORKSPACE, AGENT, "bnd_missing") is None


def test_async_offline_store_and_execution_recovery(tmp_path: Path) -> None:
    world = World.create()
    key = SigningKey.generate("orgkey_offline")
    path = world.export(tmp_path / "bundle.json", key)
    trust = BundleTrust({key.key_id: key.public_key()})
    store = LocalBindingStore(tmp_path / "pins")
    managed = world.bind(
        two_node_graph(),
        bundle=path,
        offline=True,
        workspace_id=WORKSPACE,
        trust=trust,
        binding_store=store,
        pin_scope="execution",
    )

    async def run() -> dict[str, Any]:
        first = await managed.ainvoke({"log": []}, executions("req-a", "E"))
        await managed.aupdate_state(thread("E"), {"log": ["edited"]}, as_node="supervise")
        snapshot = await managed.aget_state(thread("E"))
        history = [item async for item in managed.aget_state_history(thread("E"))]
        return {"first": first, "snapshot": snapshot, "history": history}

    out = asyncio.run(run())
    assert out["first"]["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    pinned = store.get(WORKSPACE, AGENT, execution_key(WORKSPACE, "req-a"))
    assert pinned is not None
    assert out["snapshot"].metadata["agenomic_binding_id"] == pinned.binding_id
    assert out["history"][0].metadata["agenomic_binding_id"] == pinned.binding_id
    for stored in (tmp_path / "pins").rglob("*.json"):
        stored.unlink()
    with pytest.raises(ApiError) as missing:
        asyncio.run(managed.aupdate_state(thread("E"), {"log": ["x"]}, as_node="supervise"))
    assert missing.value.code == "execution_binding_not_found"


def test_async_revalidate_never_uses_cached_binding() -> None:
    world, client = World.local()
    managed = world.bind(two_node_graph(), client=client, revalidate="never")

    async def run() -> list[str]:
        await managed.ainvoke({"log": []}, thread("cached"))
        world.promote("v2")
        out = await managed.ainvoke({"log": []}, thread("cached"))
        return list(out["log"])

    assert asyncio.run(run())[-1] == "sup:SUP v1"
