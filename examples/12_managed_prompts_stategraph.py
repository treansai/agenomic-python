from __future__ import annotations

import itertools
import operator
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

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
SLOTS = ("supervisor.system", "planner.instructions", "writer.response")


def text_prompt(body: str, *names: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {name: {"type": "string", "required": True} for name in names},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


def chat_prompt(system: str, *names: str) -> dict[str, Any]:
    document = text_prompt("", *names)
    document["kind"] = "chat"
    document["body"] = [
        {"role": "system", "content": system},
        {"placeholder": "messages", "optional": True},
    ]
    document["variables"]["messages"] = {"type": "messages", "required": False}
    return document


class State(TypedDict, total=False):
    customer: str
    locale: str
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    plan: str
    used: Annotated[list[str], operator.add]


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(content=r) for r in replies]))


def build_graph() -> Any:
    supervisor_model = fake_model("route: plan then write")
    planner_model = fake_model("1. Find the order. 2. Check the carrier. 3. Reply.")
    writer_model = fake_model("Hello Acme, your order left the warehouse today.")

    def supervise(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        messages = prompts.render_messages(
            "supervisor.system", {"customer": state["customer"], "messages": state["messages"]}
        )
        reply = supervisor_model.invoke(messages, prompts.config_for("supervisor.system"))
        return {
            "route": str(reply.content),
            "used": [f"supervisor.system -> {prompts.version('supervisor.system').ref}"],
        }

    def plan(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        system = prompts.render_text("planner.instructions", {"locale": state["locale"]})
        reply = planner_model.invoke(
            [SystemMessage(system), *state["messages"]],
            prompts.config_for("planner.instructions"),
        )
        return {
            "plan": str(reply.content),
            "used": [f"planner.instructions -> {prompts.version('planner.instructions').ref}"],
        }

    def write(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        system = prompts.render_text(
            "writer.response", {"customer": state["customer"], "plan": state["plan"]}
        )
        reply = writer_model.invoke(
            [SystemMessage(system), *state["messages"]], prompts.config_for("writer.response")
        )
        return {
            "messages": [reply],
            "used": [f"writer.response -> {prompts.version('writer.response').ref}"],
        }

    builder = StateGraph(State)
    builder.add_node("supervise", supervise)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "supervise")
    builder.add_edge("supervise", "plan")
    builder.add_edge("plan", "write")
    return builder.compile(checkpointer=InMemorySaver())


def publish(client: Client, prompt_id: str, kind: Any, document: dict[str, Any]) -> None:
    client.prompts.create(prompt_id, name=prompt_id, kind=kind)
    client.prompts.publish(prompt_id, document, parent_version=None, change_message="First version")


def main() -> None:
    client = Client()
    publish(
        client,
        "prm_supervisor",
        "chat",
        chat_prompt("You supervise the support desk of {customer}.", "customer"),
    )
    publish(
        client,
        "prm_planner",
        "text",
        text_prompt("Plan the answer in three short steps, in the {locale} locale.", "locale"),
    )
    publish(
        client,
        "prm_writer",
        "text",
        text_prompt(
            "Write the reply to {customer} following this plan: {plan}", "customer", "plan"
        ),
    )

    planner = client.prompts.get("prm_planner:1")
    print("render_text:", planner.render_text({"locale": "en"}))
    print("to_langchain:", planner.to_langchain().format(locale="fr"))
    supervisor = client.prompts.get("prm_supervisor:1").to_langchain()
    rendered = supervisor.format_messages(customer="Acme", messages=[HumanMessage("Hi")])
    print("chat to_langchain:", [message.type for message in rendered])

    engine = client.prompts.local
    release_id = engine.create_release(
        AGENT_ID,
        {
            "supervisor.system": "prm_supervisor:1",
            "planner.instructions": "prm_planner:1",
            "writer.response": "prm_writer:1",
        },
    )
    engine.move_channel(AGENT_ID, "production", release_id, expected_generation=0)

    managed = bind_langgraph(build_graph(), client=client, agent_id=AGENT_ID, channel="production")
    config: RunnableConfig = {"configurable": {"thread_id": "ticket-1001"}}
    result = managed.invoke(
        {
            "customer": "Acme",
            "locale": "en",
            "messages": [HumanMessage("Where is my order?")],
        },
        config,
    )

    print("slots used by the run:")
    for line in result["used"]:
        print(f"  {line}")
    print("reply:", result["messages"][-1].content)

    metadata = managed.get_state(config).metadata
    release = engine.get_release(release_id)
    print("checkpoint pin:", metadata["agenomic_binding_id"], metadata["agenomic_release_id"])

    assert [line.split(" -> ")[0] for line in result["used"]] == list(SLOTS)
    assert metadata["agenomic_release_id"] == release_id
    assert metadata["agenomic_prompt_manifest_digest"] == release["prompt_manifest_digest"]
    assert not any(isinstance(message, SystemMessage) for message in result["messages"])


if __name__ == "__main__":
    main()
