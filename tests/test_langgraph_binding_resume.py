from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from langgraph_world import LogState, World, plan_node, thread, two_node_graph
from prompt_fakes import AGENT, WORKSPACE

from agenomic.integrations.langgraph_binding import prompts_for
from agenomic.prompts import PromptBindingError, execution_key, thread_key

CHILD_SCRIPT = Path(__file__).parent / "langgraph_restart_child.py"


def ask(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    answer = interrupt("approve?")
    after = prompts_for(config).render_text("planner.instructions")
    return {"log": [f"ask:{answer}:{before}:{after}"]}


async def aask(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    answer = interrupt("approve?")
    after = prompts_for(config).render_text("planner.instructions")
    return {"log": [f"ask:{answer}:{before}:{after}"]}


def interrupting_graph(saver: Any, node: Any = ask) -> Any:
    builder = StateGraph(LogState)
    builder.add_node("plan", plan_node)
    builder.add_node("ask", node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "ask")
    return builder.compile(checkpointer=saver)


def recorder(world: World) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []

    def keep(request: httpx.Request, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.endswith("/bindings"):
            answers.append({"created": payload.get("created"), **payload["binding"]})
        return payload

    world.server.rewrite = keep
    return answers


def test_interrupt_resume_after_release_change_simulated_restart() -> None:
    world = World.create()
    saver = InMemorySaver()
    first = world.bind(interrupting_graph(saver))
    paused = first.invoke({"log": []}, thread("T"))
    assert [item.value for item in paused["__interrupt__"]] == ["approve?"]
    world.promote("v2")
    restarted = world.bind(interrupting_graph(saver), client=world.client())
    resumed = restarted.invoke(Command(resume="yes"), thread("T"))
    assert resumed["log"] == ["plan:PLAN v1", "ask:yes:PLAN v1:PLAN v1"]
    fresh = restarted.invoke({"log": []}, thread("N"))
    assert fresh["log"] == ["plan:PLAN v2"]


def test_resume_never_reresolves_channel() -> None:
    world = World.create()
    answers = recorder(world)
    saver = InMemorySaver()
    world.bind(interrupting_graph(saver)).invoke({"log": []}, thread("T"))
    world.promote("v2")
    restarted = world.bind(interrupting_graph(saver), client=world.client())
    restarted.invoke(Command(resume="yes"), thread("T"))
    assert [answer["created"] for answer in answers] == [True, False]
    assert {answer["release_id"] for answer in answers} == {world.releases["v1"]}
    assert answers[1]["resolved_from"]["generation"] == 1
    paths = world.server.paths()
    assert not any(path.startswith("GET") and "/bindings/" in path for path in paths)
    assert not any(path.endswith("/resolve") for path in paths)


def test_execution_scope_resume_recovers_binding() -> None:
    world = World.create()
    saver = InMemorySaver()
    managed = world.bind(interrupting_graph(saver), pin_scope="execution")
    managed.invoke({"log": []}, thread("E", agenomic_execution_key="req-1"))
    world.promote("v2")
    restarted = world.bind(interrupting_graph(saver), pin_scope="execution")
    resumed = restarted.invoke(Command(resume="ok"), thread("E"))
    assert resumed["log"] == ["plan:PLAN v1", "ask:ok:PLAN v1:PLAN v1"]
    posts = world.binding_posts()
    assert [post["thread_key"] for post in posts] == [execution_key(WORKSPACE, "req-1")]
    assert any(path.startswith("GET") and "/bindings/bnd_" in path for path in world.server.paths())
    with pytest.raises(PromptBindingError) as unrecoverable:
        restarted.invoke(None, thread("never-ran"))
    assert unrecoverable.value.code == "execution_binding_unrecoverable"


def test_execution_scope_new_turn_after_promotion_and_restart_succeeds() -> None:
    world = World.create()
    saver = InMemorySaver()
    first = world.bind(two_node_graph(saver), pin_scope="execution")
    assert first.invoke({"log": []}, thread("T", agenomic_execution_key="turn-1"))["log"] == [
        "plan:PLAN v1",
        "sup:SUP v1",
    ]
    world.promote("v2")
    restarted = world.bind(two_node_graph(saver), pin_scope="execution", client=world.client())
    second = restarted.invoke({"log": []}, thread("T", agenomic_execution_key="turn-2"))
    assert second["log"][-2:] == ["plan:PLAN v2", "sup:SUP v2"]
    digests = [
        snapshot.metadata.get("agenomic_prompt_manifest_digest")
        for snapshot in restarted.get_state_history(thread("T"))
    ]
    assert len(set(digests) - {None}) == 2


def test_crash_before_first_checkpoint_same_binding() -> None:
    world = World.create()
    client = world.client()
    early, _, created = client.bindings.create(
        AGENT, thread_key=thread_key(WORKSPACE, "T"), scope="thread", channel="production"
    )
    assert created
    exec_early, _, _ = client.bindings.create(
        AGENT,
        thread_key=execution_key(WORKSPACE, "req-9"),
        scope="execution",
        channel="production",
    )
    world.promote("v2")
    answers = recorder(world)
    managed = world.bind(two_node_graph())
    assert managed.invoke({"log": []}, thread("T"))["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    by_execution = world.bind(two_node_graph(), pin_scope="execution")
    out = by_execution.invoke({"log": []}, thread("X", agenomic_execution_key="req-9"))
    assert out["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    assert [answer["binding_id"] for answer in answers] == [
        early.binding_id,
        exec_early.binding_id,
    ]
    assert [answer["created"] for answer in answers] == [False, False]


def test_binding_checkpoint_mismatch_fails_closed() -> None:
    saver = InMemorySaver()
    first = World.create()
    first.bind(two_node_graph(saver)).invoke({"log": []}, thread("T"))
    rebuilt = World.create()
    rebuilt.promote("v2")
    managed = rebuilt.bind(two_node_graph(saver))
    with pytest.raises(PromptBindingError) as raised:
        managed.invoke({"log": []}, thread("T"))
    assert raised.value.code == "binding_checkpoint_mismatch"
    legacy = InMemorySaver()

    def plain(state: LogState) -> LogState:
        return {"log": ["legacy"]}

    builder = StateGraph(LogState)
    builder.add_node("plain", plain)
    builder.add_edge(START, "plain")
    builder.compile(checkpointer=legacy).invoke({"log": []}, thread("L"))
    adopted = rebuilt.bind(two_node_graph(legacy)).invoke({"log": []}, thread("L"))
    assert adopted["log"] == ["legacy", "plan:PLAN v2", "sup:SUP v2"]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="langgraph interrupt() in asyncio tasks needs Python 3.11 contextvar propagation",
)
def test_async_interrupt_resume() -> None:
    world = World.create()
    saver = InMemorySaver()

    async def run() -> dict[str, Any]:
        managed = world.bind(interrupting_graph(saver, aask))
        await managed.ainvoke({"log": []}, thread("A"))
        world.promote("v2")
        restarted = world.bind(interrupting_graph(saver, aask), client=world.client())
        return await restarted.ainvoke(Command(resume="yes"), thread("A"))

    assert asyncio.run(run())["log"] == ["plan:PLAN v1", "ask:yes:PLAN v1:PLAN v1"]


def run_child(workdir: Path, mode: str, phase: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(CHILD_SCRIPT), str(workdir), mode, phase],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_interrupt_resume_after_real_process_restart_sqlite(tmp_path: Path, mode: str) -> None:
    pytest.importorskip("langgraph.checkpoint.sqlite")
    paused = run_child(tmp_path, mode, "1")
    assert paused["T"] == ["approve?"]
    resumed = run_child(tmp_path, mode, "2")
    assert resumed["T"] == ["plan:PLAN v1", "ask:yes:PLAN v1:PLAN v1"]
    assert resumed["N"] == ["plan:PLAN v2"]
    if mode == "online":
        assert resumed["E"] == ["plan:PLAN v1", "ask:ok:PLAN v1:PLAN v1"]
        assert resumed["update_digest"] == resumed["v1_digest"]
        assert not any(path.endswith("/resolve") for path in resumed["paths"])


def test_factory_execution_scope_resume_after_real_process_restart_sqlite(
    tmp_path: Path,
) -> None:
    pytest.importorskip("langgraph.checkpoint.sqlite")
    paused = run_child(tmp_path, "factory", "1")
    assert paused == {"E": ["approve?"], "U": ["approve?"], "S": ["approve?"]}
    resumed = run_child(tmp_path, "factory", "2")
    v1, v2 = resumed["v1_digest"], resumed["v2_digest"]
    assert resumed["E"] == ["plan:PLAN v1", "ask:ok:PLAN v1", "finish:PLAN v1"]
    assert resumed["update_digest"] == v1
    assert resumed["U"] == ["plan:PLAN v1", "edit", "finish:PLAN v1"]
    assert resumed["S"] == ["plan:PLAN v1", "approved", "finish:PLAN v1"]
    assert resumed["posts_before_new"] == 0
    assert resumed["built_before_new"] == [v1, v1, v1]
    assert resumed["N"] == ["plan:PLAN v2"]
    assert resumed["built"] == [v1, v1, v1, v2]
    assert not any(path.endswith("/resolve") for path in resumed["paths"])


def test_outage_after_real_process_restart_resumes_disk_cached_binding(tmp_path: Path) -> None:
    pytest.importorskip("langgraph.checkpoint.sqlite")
    assert run_child(tmp_path, "outage", "1") == {"T": ["approve?"]}
    resumed = run_child(tmp_path, "outage", "2")
    assert resumed["T"] == ["plan:PLAN v1", "ask:yes:PLAN v1:PLAN v1"]
    assert resumed["N"] == "registry_unavailable"
    assert resumed["N_state"] == {}
    assert resumed["unknown_workspace"] == "registry_unavailable"
    assert resumed["posts"] == 0
    assert not any(path.endswith("/resolve") for path in resumed["paths"])
