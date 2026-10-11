from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph_world import LogState, World, thread, two_node_graph
from prompt_fakes import AGENT, OTHER_WORKSPACE, WORKSPACE

from agenomic.integrations.langgraph_binding import bind_langgraph, prompts_for
from agenomic.prompts import (
    ExecutionBinding,
    PromptBindingError,
    PromptBundle,
    PromptCache,
    PromptIntegrityError,
)


async def slow_plan(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    await asyncio.sleep(0.005)
    after = prompts_for(config).render_text("planner.instructions")
    return {"log": [f"{before}|{after}"]}


def sync_plan(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    after = prompts_for(config).render_text("planner.instructions")
    return {"log": [f"{before}|{after}"]}


def one_node(node: Any) -> Any:
    builder = StateGraph(LogState)
    builder.add_node("plan", node)
    builder.add_edge(START, "plan")
    return builder.compile(checkpointer=InMemorySaver())


def test_concurrent_threads_different_releases_asyncio() -> None:
    world = World.create()
    managed = world.bind(one_node(slow_plan))

    async def run() -> list[tuple[str, list[str]]]:
        await managed.ainvoke({"log": []}, thread("old"))
        world.promote("v2")
        names = ["old", *(f"new-{index}" for index in range(6)), "old"]
        outputs = await asyncio.gather(
            *(managed.ainvoke({"log": []}, thread(name)) for name in names)
        )
        return [(name, out["log"]) for name, out in zip(names, outputs, strict=True)]

    for name, log in asyncio.run(run()):
        expected = "PLAN v1|PLAN v1" if name == "old" else "PLAN v2|PLAN v2"
        assert log[-1] == expected, name


def test_concurrent_threads_different_releases_threadpool() -> None:
    world = World.create()
    managed = world.bind(one_node(sync_plan))
    managed.invoke({"log": []}, thread("old"))
    world.promote("v2")
    names = ["old", "new-a", "old", "new-b", "new-c", "old"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(lambda name: managed.invoke({"log": []}, thread(name)), names))
    for name, out in zip(names, outputs, strict=True):
        expected = "PLAN v1|PLAN v1" if name == "old" else "PLAN v2|PLAN v2"
        assert out["log"][-1] == expected, name


def test_two_workspaces_two_clients_no_cross_artifacts() -> None:
    first = World.create(WORKSPACE)
    second = World.create(OTHER_WORKSPACE, tag="B:")
    shared = PromptCache()
    managed_a = first.bind(one_node(sync_plan), client=first.client(prompt_cache=shared))
    managed_b = second.bind(one_node(sync_plan), client=second.client(prompt_cache=shared))
    for _ in range(2):
        assert managed_a.invoke({"log": []}, thread("t"))["log"][-1] == "PLAN v1|PLAN v1"
        assert managed_b.invoke({"log": []}, thread("t"))["log"][-1] == "B:PLAN v1|B:PLAN v1"
    assert managed_a.workspace_id == WORKSPACE
    assert managed_b.workspace_id == OTHER_WORKSPACE
    key_a = first.binding_posts()[0]["thread_key"]
    key_b = second.binding_posts()[0]["thread_key"]
    assert key_a != key_b
    assert shared.get_binding(WORKSPACE, AGENT, key_b) is None
    assert shared.get_binding(OTHER_WORKSPACE, AGENT, key_a) is None


def preissued(world: World, release: str, trial: str) -> tuple[ExecutionBinding, PromptBundle]:
    document, artifacts, _ = world.engine.create_binding(
        AGENT, thread_key=f"seed:{trial}", scope="thread", release_id=world.releases[release]
    )
    document["thread_key"] = f"exp:exp_01:{trial}:a1"
    document["resolved_from"] = {"release_id": world.releases[release]}
    document["experiment"] = {
        "experiment_id": "exp_01",
        "trial_id": trial,
        "arm_key": f"arm_{release}",
        "attempt": 1,
    }
    binding = ExecutionBinding.model_validate(document)
    bundle = PromptBundle.from_online_response(
        artifacts,
        expected_workspace_id=WORKSPACE,
        expected_agent_id=AGENT,
        expected_manifest_digest=binding.prompt_manifest_digest,
    )
    return binding, bundle


def test_experiment_arms_isolated_state_and_store() -> None:
    world = World.create()
    arms = {name: preissued(world, name, f"extr_{name}") for name in ("v1", "v2")}
    savers = {name: InMemorySaver() for name in arms}
    stores = {name: InMemoryStore() for name in arms}
    metadata: dict[str, dict[str, Any]] = {}

    async def arm_node(state: LogState, config: RunnableConfig, store: BaseStore) -> LogState:
        prompts = prompts_for(config)
        text = prompts.render_text("planner.instructions")
        await store.aput(("trial",), "seen", {"text": text})
        metadata[text] = dict(config["metadata"])
        return {"log": [text]}

    def build(name: str) -> Any:
        builder = StateGraph(LogState)
        builder.add_node("plan", arm_node)
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=savers[name], store=stores[name])

    managed = {
        name: bind_langgraph(build(name), agent_id=AGENT, binding=binding, resolution=bundle)
        for name, (binding, bundle) in arms.items()
    }

    async def run() -> list[Any]:
        return list(
            await asyncio.gather(
                *(
                    managed[name].ainvoke({"log": []}, thread(arms[name][0].thread_key))
                    for name in arms
                )
            )
        )

    first, second = asyncio.run(run())
    assert first["log"] == ["PLAN v1"]
    assert second["log"] == ["PLAN v2"]
    for name in arms:
        threads = set(savers[name].storage)
        assert threads == {arms[name][0].thread_key}
        items = stores[name].search(("trial",))
        assert [item.value["text"] for item in items] == [f"PLAN {name}"]
    assert metadata["PLAN v1"]["agenomic_experiment_arm_key"] == "arm_v1"
    assert metadata["PLAN v2"]["agenomic_experiment_id"] == "exp_01"
    assert world.binding_posts() == []
    with pytest.raises(PromptBindingError) as other_thread:
        managed["v1"].invoke({"log": []}, thread("not-the-trial"))
    assert other_thread.value.code == "binding_target_mismatch"
    binding, _ = arms["v1"]
    with pytest.raises(PromptBindingError) as wrong_release:
        bind_langgraph(build("v1"), agent_id=AGENT, binding=binding, resolution=arms["v2"][1])
    assert wrong_release.value.code == "binding_mismatch"
    tampered = binding.model_copy(update={"prompt_manifest_digest": "sha256:" + "0" * 64})
    with pytest.raises((PromptBindingError, PromptIntegrityError)):
        bind_langgraph(
            build("v1"),
            agent_id=AGENT,
            binding=tampered,
            resolution=arms["v1"][1],
        )
    production = world.bind(two_node_graph())
    assert production.invoke({"log": []}, thread("prod"))["log"][-1] == "sup:SUP v1"


def test_preissued_binding_must_pin_every_child_of_the_resolution() -> None:
    world = World.create()
    binding, bundle = preissued(world, "v1", "extr_children")
    assert binding.children
    with pytest.raises(PromptIntegrityError) as refused:
        bind_langgraph(
            two_node_graph(),
            agent_id=AGENT,
            binding=binding.model_copy(update={"children": {}}),
            resolution=bundle,
        )
    assert refused.value.code == "manifest_digest_mismatch"


def test_preissued_arm_leaves_production_thread_store_and_channel_unchanged() -> None:
    world = World.create()
    saver = InMemorySaver()
    store = InMemoryStore()

    def node(state: LogState, config: RunnableConfig, store: BaseStore) -> LogState:
        text = prompts_for(config).render_text("planner.instructions")
        store.put(("seen",), str(config["configurable"]["thread_id"]), {"text": text})
        return {"log": [text]}

    builder = StateGraph(LogState)
    builder.add_node("plan", node)
    builder.add_edge(START, "plan")
    graph = builder.compile(checkpointer=saver, store=store)
    production = world.bind(graph)
    assert production.invoke({"log": []}, thread("prod"))["log"] == ["PLAN v1"]
    prod = thread("prod")

    def snapshot() -> tuple[Any, ...]:
        return (
            [item.config["configurable"]["checkpoint_id"] for item in saver.list(prod)],
            graph.get_state(prod).values,
            store.get(("seen",), "prod").value,
            world.engine.get_channel(AGENT, "production")["generation"],
            [world.engine.get_release(world.releases[name])["status"] for name in ("v1", "v2")],
        )

    before = snapshot()
    binding, bundle = preissued(world, "v2", "extr_cf")
    arm = bind_langgraph(graph, agent_id=AGENT, binding=binding, resolution=bundle)
    assert arm.invoke({"log": []}, thread(binding.thread_key))["log"] == ["PLAN v2"]
    for attempt in (
        lambda: arm.invoke({"log": []}, prod),
        lambda: arm.update_state(prod, {"log": ["forked"]}, as_node="plan"),
    ):
        with pytest.raises(PromptBindingError) as refused:
            attempt()
        assert refused.value.code == "binding_target_mismatch"
    assert snapshot() == before
    assert store.get(("seen",), binding.thread_key).value == {"text": "PLAN v2"}
    assert {item.config["configurable"]["thread_id"] for item in saver.list(None)} == {
        "prod",
        binding.thread_key,
    }
