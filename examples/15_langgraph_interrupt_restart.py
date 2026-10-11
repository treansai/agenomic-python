from __future__ import annotations

import itertools
import json
import operator
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from agenomic.crypto.signing import SigningKey
from agenomic.integrations import LocalBindingStore, bind_langgraph, prompts_for
from agenomic.prompts import BundleTrust, LocalPromptEngine, PromptBundle

WORKSPACE_ID = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"
AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
KEY_ID = "orgkey_example"
PAUSED_THREAD = "refund-42"
NEW_THREAD = "refund-43"
PLANS = ("Refund in three steps.", "Refund in two steps and offer a voucher.")


def text_prompt(body: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def build_graph(saver: Any) -> Any:
    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="plan drafted")]))

    def plan(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        system = prompts.render_text("planner.instructions")
        model.invoke(
            [SystemMessage(system), HumanMessage("refund order 42")],
            prompts.config_for("planner.instructions"),
        )
        return {"log": [f"plan: {system}"]}

    def approve(state: State, config: RunnableConfig) -> State:
        answer = interrupt("approve the refund plan?")
        system = prompts_for(config).render_text("planner.instructions")
        return {"log": [f"approve: {answer}: {system}"]}

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("approve", approve)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approve")
    return builder.compile(checkpointer=saver)


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def bind(graph: Any, workdir: Path, bundle: str, retained: list[PromptBundle]) -> Any:
    return bind_langgraph(
        graph,
        agent_id=AGENT_ID,
        channel="production",
        offline=True,
        workspace_id=WORKSPACE_ID,
        bundle=workdir / bundle,
        retained_bundles=retained,
        trust=BundleTrust.from_pem_files(workdir / f"{KEY_ID}.pem"),
        binding_store=LocalBindingStore(workdir / "pins"),
    )


def export(engine: LocalPromptEngine, signer: SigningKey, path: Path) -> str:
    document = engine.export_bundle(AGENT_ID, signer=signer, channel="production")
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(document["prompt_manifest_digest"])


def first_process(workdir: Path) -> dict[str, str]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    engine = LocalPromptEngine(WORKSPACE_ID)
    engine.create_prompt("prm_planner", kind="text", name="Planner")
    releases = []
    for number, body in enumerate(PLANS, start=1):
        engine.publish(
            "prm_planner",
            text_prompt(body),
            parent_version=None if number == 1 else number - 1,
            change_message=f"Version {number}",
        )
        releases.append(
            engine.create_release(AGENT_ID, {"planner.instructions": f"prm_planner:{number}"})
        )
    signer = SigningKey.generate(KEY_ID)
    (workdir / f"{KEY_ID}.pem").write_text(signer.public_pem(), encoding="utf-8")

    engine.move_channel(AGENT_ID, "production", releases[0], expected_generation=0)
    old_digest = export(engine, signer, workdir / "bundle-v1.json")
    with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
        managed = bind(build_graph(saver), workdir, "bundle-v1.json", [])
        paused = managed.invoke({"log": []}, thread(PAUSED_THREAD))
        print(
            "process 1:",
            PAUSED_THREAD,
            "paused on",
            [item.value for item in paused["__interrupt__"]],
        )
        print("process 1:", paused["log"][-1])

    engine.move_channel(AGENT_ID, "production", releases[1], expected_generation=1)
    new_digest = export(engine, signer, workdir / "bundle-v2.json")
    print("simulation of the governed path: production now points to version 2, shipped as a")
    print("new signed bundle; in Agenomic Cloud that move is an approved, signed-in action")
    return {"old": old_digest, "new": new_digest}


def second_process(workdir: Path) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    earlier = PromptBundle.load(
        workdir / "bundle-v1.json",
        expected_workspace_id=WORKSPACE_ID,
        expected_agent_id=AGENT_ID,
        trust=BundleTrust.from_pem_files(workdir / f"{KEY_ID}.pem"),
    )
    with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
        graph = build_graph(saver)
        managed = bind(graph, workdir, "bundle-v2.json", [earlier])
        resumed = managed.invoke(Command(resume="approved"), thread(PAUSED_THREAD))
        fresh = managed.invoke({"log": []}, thread(NEW_THREAD))
        summary = {
            name: {
                "log": output["log"],
                "digest": graph.get_state(thread(name)).metadata["agenomic_prompt_manifest_digest"],
            }
            for name, output in ((PAUSED_THREAD, resumed), (NEW_THREAD, fresh))
        }
    print(json.dumps(summary))


def main() -> None:
    if find_spec("langgraph.checkpoint.sqlite") is None:
        print("this example needs the SQLite saver: pip install langgraph-checkpoint-sqlite")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        workdir = Path(directory)
        digests = first_process(workdir)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--resume", str(workdir)],
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
    if child.returncode != 0:
        raise SystemExit(f"the restarted process failed:\n{child.stderr}")
    summary = json.loads(child.stdout.strip().splitlines()[-1])
    for name, entry in summary.items():
        print(f"process 2: {name} -> {entry['log']}")
    paused, fresh = summary[PAUSED_THREAD], summary[NEW_THREAD]
    assert paused["log"] == [f"plan: {PLANS[0]}", f"approve: approved: {PLANS[0]}"]
    assert paused["digest"] == digests["old"]
    assert fresh["log"] == [f"plan: {PLANS[1]}"]
    assert fresh["digest"] == digests["new"]
    print(
        "the resumed thread kept",
        digests["old"][:19],
        "and the new thread uses",
        digests["new"][:19],
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--resume"]:
        second_process(Path(sys.argv[2]))
    else:
        main()
