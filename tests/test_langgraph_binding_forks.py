from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph_world import World, thread, two_node_graph


def after_plan(managed: Any, thread_id: str) -> Any:
    for snapshot in managed.get_state_history(thread(thread_id)):
        if snapshot.next == ("supervise",):
            return snapshot
    raise AssertionError("no checkpoint before supervise")


def test_time_travel_inherits_thread_pin() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    managed.invoke({"log": []}, thread("T"))
    world.promote("v2")
    fork = after_plan(managed, "T")
    replayed = managed.invoke(None, fork.config)
    assert replayed["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    digests = {
        snapshot.metadata.get("agenomic_prompt_manifest_digest")
        for snapshot in managed.get_state_history(thread("T"))
    }
    assert digests == {world.engine.get_release(world.releases["v1"])["prompt_manifest_digest"]}


def test_update_state_inherits_thread_pin() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    managed.invoke({"log": []}, thread("T"))
    world.promote("v2")
    fork = after_plan(managed, "T")
    edited = managed.update_state(fork.config, {"log": ["edited"]}, as_node="plan")
    snapshot = managed.get_state(edited)
    v1 = world.engine.get_release(world.releases["v1"])["prompt_manifest_digest"]
    assert snapshot.metadata["agenomic_prompt_manifest_digest"] == v1
    resumed = managed.invoke(None, edited)
    assert resumed["log"][-1] == "sup:SUP v1"


def test_execution_scope_fork_uses_the_forked_checkpoint_binding() -> None:
    world = World.create()
    saver = InMemorySaver()
    managed = world.bind(two_node_graph(saver), pin_scope="execution")
    managed.invoke({"log": []}, thread("T", agenomic_execution_key="turn-1"))
    world.promote("v2")
    managed.invoke({"log": []}, thread("T", agenomic_execution_key="turn-2"))
    first_turn = [
        snapshot
        for snapshot in managed.get_state_history(thread("T"))
        if snapshot.next == ("supervise",)
    ][-1]
    replayed = managed.invoke(None, first_turn.config)
    assert replayed["log"][-1] == "sup:SUP v1"
    assert len(world.binding_posts()) == 2
