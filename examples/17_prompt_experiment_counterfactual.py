from __future__ import annotations

import asyncio
import itertools
import json
import operator
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from agenomic import Client
from agenomic.experiments import ExperimentRunner, GraphNodeEntryPoint, GraphTarget
from agenomic.experiments.runner import local_assignment, snapshot_case
from agenomic.integrations import bind_langgraph, prompts_for

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
PLANNER_BODIES = (
    "Plan the next step for: {request}",
    "Plan the next step for: {request}. Check the order status before any refund.",
)
ENTRY_POINT = {
    "name": "planner",
    "kind": "graph_node",
    "node_path": "plan",
    "seed_as_node": "intake",
    "slot_paths": ["planner.instructions"],
}


def text_prompt(body: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {"request": {"type": "string", "required": True}},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]


def build_graph(checkpointer: Any, store: Any) -> Any:
    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="lookup_order")]))

    def intake(state: State) -> State:
        return {"log": ["intake: refund request received"]}

    def plan(state: State, config: RunnableConfig, store: BaseStore) -> State:
        prompts = prompts_for(config)
        instructions = prompts.render_text(
            "planner.instructions", {"request": str(state["messages"][0].content)}
        )
        reply = model.invoke(
            [SystemMessage(instructions), *state["messages"]],
            prompts.config_for("planner.instructions"),
        )
        thread_id = str(config["configurable"]["thread_id"])
        store.put(("plans",), thread_id, {"instructions": instructions})
        return {"log": [f"plan: {instructions} -> {reply.content}"]}

    def respond(state: State) -> State:
        return {"log": ["respond: reply sent to the customer"]}

    builder = StateGraph(State)
    builder.add_node("intake", intake)
    builder.add_node("plan", plan)
    builder.add_node("respond", respond)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "plan")
    builder.add_edge("plan", "respond")
    return builder.compile(checkpointer=checkpointer, store=store)


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def main() -> None:
    client = Client()
    engine = client.prompts.local
    client.prompts.create("prm_planner", name="Planner", kind="text")
    releases = []
    for number, body in enumerate(PLANNER_BODIES, start=1):
        client.prompts.publish(
            "prm_planner",
            text_prompt(body),
            parent_version=None if number == 1 else number - 1,
            change_message=f"Version {number}",
        )
        releases.append(
            engine.create_release(AGENT_ID, {"planner.instructions": f"prm_planner:{number}"})
        )
    baseline, candidate = releases
    engine.move_channel(AGENT_ID, "production", baseline, expected_generation=0)

    production_saver, production_store = InMemorySaver(), InMemoryStore()
    production_graph = build_graph(production_saver, production_store)
    production = bind_langgraph(
        production_graph, client=client, agent_id=AGENT_ID, channel="production"
    )
    production.invoke({"messages": [HumanMessage("Refund order 1234")]}, thread("customer-7"))

    def fingerprint() -> str:
        return json.dumps(
            {
                "checkpoints": [
                    item.config["configurable"]["checkpoint_id"]
                    for item in production_saver.list(thread("customer-7"))
                ],
                "log": production_graph.get_state(thread("customer-7")).values["log"],
                "store": [item.value for item in production_store.search(("plans",))],
                "channel": engine.get_channel(AGENT_ID, "production")["generation"],
                "statuses": [engine.get_release(release)["status"] for release in releases],
            },
            sort_keys=True,
        )

    before = fingerprint()
    history = list(production_graph.get_state_history(thread("customer-7")))
    before_plan = next(item for item in history if item.next == ("plan",))
    case = snapshot_case(
        production_graph,
        "customer-7",
        agent_id=AGENT_ID,
        case_id="refund-1234-counterfactual",
        checkpoint_id=before_plan.config["configurable"]["checkpoint_id"],
    )
    print("snapshot provenance:", case["input"]["provenance"])

    print("simulation: the trial assignments are built in this process; in Agenomic Cloud the")
    print("server leases them to a registered runner and creates each trial binding itself")
    runner = ExperimentRunner(
        targets={
            AGENT_ID: GraphTarget(
                factory=lambda ctx: build_graph(ctx.checkpointer, ctx.store),
                runtime_digest=engine.get_release(baseline)["bundle_hash"],
                entry_points={"planner": GraphNodeEntryPoint("plan", seed_as_node="intake")},
            )
        }
    )
    arms = {
        name: local_assignment(
            engine,
            agent_id=AGENT_ID,
            release_id=release,
            case=case,
            experiment_id="exp_example17",
            level="node",
            entry_point=ENTRY_POINT,
            seed="42",
        )
        for name, release in (("baseline", baseline), ("candidate", candidate))
    }

    async def run_arms() -> list[Any]:
        return list(await asyncio.gather(*(runner.arun_trial(arm) for arm in arms.values())))

    results = dict(zip(arms, asyncio.run(run_arms()), strict=True))
    for name, run in results.items():
        assert run.kind == "result"
        result = run.result
        assert result is not None
        binding = arms[name]["view"]["binding"]
        print(
            f"{name} arm {arms[name]['view']['arm']['arm_key']} on thread {binding['thread_key']}"
        )
        print("  parent binding:", binding["parent_binding_id"])
        print("  state update:", result["output"]["state_update"]["log"][-1])
        print("  prompt refs:", [call["prompt_ref"] for call in result["model_calls"]])
    disclosure = results["candidate"].result["isolation"] if results["candidate"].result else {}
    print("isolation record:", json.dumps(disclosure, sort_keys=True))
    after = fingerprint()
    print("production thread, store, channel and releases unchanged:", after == before)

    assert after == before
    assert results["baseline"].result is not None
    assert results["candidate"].result is not None
    baseline_plan = results["baseline"].result["output"]["state_update"]["log"][-1]
    candidate_plan = results["candidate"].result["output"]["state_update"]["log"][-1]
    assert "before any refund" not in baseline_plan
    assert "Check the order status before any refund." in candidate_plan
    assert [c["prompt_ref"] for c in results["candidate"].result["model_calls"]] == [
        "prm_planner:2"
    ]
    assert disclosure["fork_fidelity"] == "state_values_only"
    assert "checkpoint_history" in disclosure["not_preserved"]
    assert all(
        arm["view"]["binding"]["parent_binding_id"] == before_plan.metadata["agenomic_binding_id"]
        for arm in arms.values()
    )


if __name__ == "__main__":
    main()
