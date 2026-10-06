from __future__ import annotations

import itertools
import operator
from contextlib import closing
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agenomic import Client
from agenomic.integrations import bind_langgraph, prompts_for
from agenomic.prompts import LocalPromptEngine

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
BODIES = ("{step} the reply politely.", "{step} the reply politely and offer a callback.")


def text_prompt(body: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {"step": {"type": "string", "required": True}},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]


def build_graph() -> Any:
    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="Sure, done.")]))

    def step(name: str, verb: str) -> Any:
        def node(state: State, config: RunnableConfig) -> State:
            prompts = prompts_for(config)
            system = prompts.render_text("support.system", {"step": verb})
            reply = model.invoke(
                [SystemMessage(system), *state["messages"]], prompts.config_for("support.system")
            )
            entry = f"{name}: {prompts.version('support.system').ref}: {system}"
            return {"log": [entry], **({"messages": [reply]} if name == "finalize" else {})}

        return node

    builder = StateGraph(State)
    builder.add_node("draft", step("draft", "Draft"))
    builder.add_node("finalize", step("finalize", "Finalize"))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "finalize")
    return builder.compile(checkpointer=InMemorySaver())


def turn(text: str) -> State:
    return {"messages": [HumanMessage(text)]}


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def promote(engine: LocalPromptEngine, release_id: str) -> None:
    current = engine.get_channel(AGENT_ID, "production")["generation"]
    engine.move_channel(AGENT_ID, "production", release_id, expected_generation=current)


def main() -> None:
    client = Client()
    engine = client.prompts.local
    client.prompts.create("prm_support", name="Support", kind="text")
    releases = []
    for number, body in enumerate(BODIES, start=1):
        client.prompts.publish(
            "prm_support",
            text_prompt(body),
            parent_version=None if number == 1 else number - 1,
            change_message=f"Version {number}",
        )
        releases.append(
            engine.create_release(AGENT_ID, {"support.system": f"prm_support:{number}"})
        )
    promote(engine, releases[0])
    managed = bind_langgraph(build_graph(), client=client, agent_id=AGENT_ID, channel="production")

    print("simulation of the governed path: the local engine moves the production channel")
    print("directly; in Agenomic Cloud a promotion is an approved action taken in a signed-in")
    print("session, and the SDK has no promote")

    in_flight: list[str] = []
    with closing(
        managed.stream(turn("Where is my parcel?"), thread("customer-a"), stream_mode="updates")
    ) as updates:
        for update in updates:
            for node, values in update.items():
                in_flight.extend(values["log"])
                if node == "draft":
                    promote(engine, releases[1])
                    print("production moved to version 2 while customer-a was between two nodes")
    print("customer-a, turn 1 (in flight during the promotion):", in_flight)

    later = managed.invoke(turn("Can you call me back?"), thread("customer-a"))["log"][-2:]
    print("customer-a, turn 2 (existing thread):", later)
    fresh = managed.invoke(turn("Where is my parcel?"), thread("customer-b"))["log"]
    print("customer-b, turn 1 (new thread):", fresh)

    for event in engine.get_channel(AGENT_ID, "production")["history"]:
        print(
            "channel history:",
            event["generation"],
            event["action"],
            engine.get_release(event["to_release_id"])["name"],
        )

    assert all(": prm_support:1: " in entry for entry in in_flight + later)
    assert all(": prm_support:2: " in entry for entry in fresh)
    assert len(in_flight) == len(later) == len(fresh) == 2


if __name__ == "__main__":
    main()
