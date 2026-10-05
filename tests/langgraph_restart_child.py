from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from langgraph_world import LogState, World, plan_node, thread
from prompt_fakes import AGENT, WORKSPACE

from agenomic.crypto.signing import SigningKey
from agenomic.integrations.langgraph_binding import (
    LocalBindingStore,
    bind_langgraph,
    prompts_for,
)
from agenomic.prompts import BundleTrust, PromptBundle
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
    run = online if mode == "online" else offline
    print(json.dumps(run(folder, step)))
