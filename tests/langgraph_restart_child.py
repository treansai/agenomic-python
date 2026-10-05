from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from langgraph_world import LogState, World, captured_build, plan_node, thread
from prompt_fakes import AGENT, WORKSPACE

from agenomic import _transport
from agenomic.crypto.signing import SigningKey
from agenomic.integrations.langgraph_binding import (
    AgentFactory,
    LocalBindingStore,
    bind_langgraph,
    prompts_for,
)
from agenomic.prompts import BundleTrust, PromptBundle, PromptCache, RegistryUnavailableError
from agenomic.prompts.local import LocalPromptEngine


def ask(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    answer = interrupt("approve?")
    after = prompts_for(config).render_text("planner.instructions")
    return {"log": [f"ask:{answer}:{before}:{after}"]}


def build(saver: SqliteSaver) -> Any:
    builder = StateGraph(LogState)
    builder.add_node("plan", plan_node)
    builder.add_node("ask", ask)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "ask")
    return builder.compile(checkpointer=saver)


def online(workdir: Path, phase: str) -> dict[str, Any]:
    engine_path = workdir / "engine.json"
    results: dict[str, Any] = {}
    if phase == "1":
        world = World.create(state_path=engine_path)
        (workdir / "releases.json").write_text(json.dumps(world.releases))
    else:
        engine = LocalPromptEngine(WORKSPACE, state_path=engine_path)
        world = World(engine, json.loads((workdir / "releases.json").read_text()))
    with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
        graph = build(saver)
        by_thread = world.bind(graph)
        by_execution = world.bind(graph, pin_scope="execution")
        if phase == "1":
            out = by_thread.invoke({"log": []}, thread("T"))
            results["T"] = [item.value for item in out["__interrupt__"]]
            out = by_execution.invoke({"log": []}, thread("E", agenomic_execution_key="req-1"))
            results["E"] = [item.value for item in out["__interrupt__"]]
            world.promote("v2")
        else:
            results["T"] = by_thread.invoke(Command(resume="yes"), thread("T"))["log"]
            results["E"] = by_execution.invoke(Command(resume="ok"), thread("E"))["log"]
            results["N"] = by_thread.invoke({"log": []}, thread("N"))["log"]
            results["paths"] = world.server.paths()
            stamped = by_thread.update_state(thread("T"), {"log": ["edit"]}, as_node="ask")
            results["update_digest"] = graph.get_state(stamped).metadata.get(
                "agenomic_prompt_manifest_digest"
            )
            results["v1_digest"] = world.engine.get_release(world.releases["v1"])[
                "prompt_manifest_digest"
            ]
    return results


def persisted_world(workdir: Path, phase: str) -> World:
    engine_path = workdir / "engine.json"
    if phase == "1":
        world = World.create(state_path=engine_path)
        (workdir / "releases.json").write_text(json.dumps(world.releases))
        return world
    engine = LocalPromptEngine(WORKSPACE, state_path=engine_path)
    return World(engine, json.loads((workdir / "releases.json").read_text()))


def binding_posts(world: World) -> int:
    return sum(1 for path in world.server.paths() if path == f"POST /v1/agents/{AGENT}/bindings")


def factory(workdir: Path, phase: str) -> dict[str, Any]:
    world = persisted_world(workdir, phase)
    results: dict[str, Any] = {}
    built: list[str] = []
    with SqliteSaver.from_conn_string(str(workdir / "factory.sqlite")) as saver:
        managed = world.bind(
            AgentFactory(captured_build(saver, built), checkpointer=saver), pin_scope="execution"
        )
        if phase == "1":
            for thread_id, key in (("E", "req-1"), ("U", "req-2"), ("S", "req-3")):
                out = managed.invoke({"log": []}, thread(thread_id, agenomic_execution_key=key))
                results[thread_id] = [item.value for item in out["__interrupt__"]]
            managed.update_state(thread("S"), {"log": ["approved"]}, as_node="ask")
            world.promote("v2")
            return results
        results["E"] = managed.invoke(Command(resume="ok"), thread("E"))["log"]
        second = world.bind(
            AgentFactory(captured_build(saver, built), checkpointer=saver), pin_scope="execution"
        )
        stamped = second.update_state(thread("U"), {"log": ["edit"]}, as_node="ask")
        results["update_digest"] = second.get_state(stamped).metadata.get(
            "agenomic_prompt_manifest_digest"
        )
        results["U"] = second.invoke(None, thread("U"))["log"]
        third = world.bind(
            AgentFactory(captured_build(saver, built), checkpointer=saver), pin_scope="execution"
        )
        results["S"] = third.invoke(None, thread("S"))["log"]
        results["posts_before_new"] = binding_posts(world)
        results["built_before_new"] = list(built)
        third.invoke({"log": []}, thread("N", agenomic_execution_key="req-9"))
        results["N"] = third.get_state(thread("N")).values["log"]
        results["built"] = list(built)
        results["paths"] = world.server.paths()
        for name in ("v1", "v2"):
            results[f"{name}_digest"] = world.engine.get_release(world.releases[name])[
                "prompt_manifest_digest"
            ]
    return results


def outage(workdir: Path, phase: str) -> dict[str, Any]:
    world = persisted_world(workdir, phase)
    results: dict[str, Any] = {}
    cache = PromptCache(workdir / "cache")
    with SqliteSaver.from_conn_string(str(workdir / "outage.sqlite")) as saver:
        graph = build(saver)
        if phase == "1":
            managed = world.bind(
                graph, client=world.client(workspace_id=WORKSPACE, prompt_cache=cache)
            )
            out = managed.invoke({"log": []}, thread("T"))
            results["T"] = [item.value for item in out["__interrupt__"]]
            world.promote("v2")
            return results
        _transport._sleep = lambda delay: None
        world.server.outage = "transport_error"
        managed = world.bind(graph, client=world.client(workspace_id=WORKSPACE, prompt_cache=cache))
        results["T"] = managed.invoke(Command(resume="yes"), thread("T"))["log"]
        try:
            managed.invoke({"log": []}, thread("N"))
            results["N"] = "ran"
        except RegistryUnavailableError as error:
            results["N"] = error.code
        results["N_state"] = graph.get_state(thread("N")).values
        try:
            world.bind(graph, client=world.client(prompt_cache=cache))
            results["unknown_workspace"] = "bound"
        except RegistryUnavailableError as error:
            results["unknown_workspace"] = error.code
        results["posts"] = binding_posts(world)
        results["paths"] = world.server.paths()
    return results


def offline(workdir: Path, phase: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    store = LocalBindingStore(workdir / "pins")
    if phase == "1":
        world = World.create()
        signer = SigningKey.generate("orgkey_restart")
        (workdir / f"{signer.key_id}.pem").write_text(signer.public_pem())
        world.export(workdir / "bundle-v1.json", signer)
        world.promote("v2")
        world.export(workdir / "bundle-v2.json", signer)
    trust = BundleTrust.from_pem_files(workdir / "orgkey_restart.pem")
    with SqliteSaver.from_conn_string(str(workdir / "offline.sqlite")) as saver:
        graph = build(saver)
        options: dict[str, Any] = {
            "offline": True,
            "workspace_id": WORKSPACE,
            "trust": trust,
            "binding_store": store,
        }
        if phase == "1":
            managed = bind_langgraph(
                graph,
                agent_id=AGENT,
                channel="production",
                bundle=workdir / "bundle-v1.json",
                **options,
            )
            out = managed.invoke({"log": []}, thread("T"))
            results["T"] = [item.value for item in out["__interrupt__"]]
        else:
            retained = PromptBundle.load(
                workdir / "bundle-v1.json",
                expected_workspace_id=WORKSPACE,
                expected_agent_id=AGENT,
                trust=trust,
            )
            managed = bind_langgraph(
                graph,
                agent_id=AGENT,
                channel="production",
                bundle=workdir / "bundle-v2.json",
                retained_bundles=[retained],
                **options,
            )
            results["T"] = managed.invoke(Command(resume="yes"), thread("T"))["log"]
            out = managed.invoke({"log": []}, thread("N"))
            results["N"] = out["log"]
    return results


if __name__ == "__main__":
    folder, mode, step = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    run = {"online": online, "offline": offline, "factory": factory, "outage": outage}[mode]
    print(json.dumps(run(folder, step)))
