from __future__ import annotations

import itertools
import operator
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from agenomic import Client
from agenomic.integrations import bind_langgraph, prompts_for, scope_config

SUPERVISOR_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
RESEARCHER_ID = "7f3c9a1e-0b2d-4c5e-8f6a-9b0c1d2e3f4a"
REVIEWER_ID = "3c4d5e6f-7a8b-4c9d-8e0f-1a2b3c4d5e6f"
NAMES = {SUPERVISOR_ID: "supervisor", RESEARCHER_ID: "researcher", REVIEWER_ID: "reviewer"}


def text_prompt(body: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {"topic": {"type": "string", "required": True}},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


class State(TypedDict, total=False):
    topic: str
    route: str
    log: Annotated[list[str], operator.add]


def fake_model(reply: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(content=reply)]))


def call(model: GenericFakeChatModel, config: RunnableConfig, slot: str, topic: str) -> str:
    prompts = prompts_for(config)
    system = prompts.render_text(slot, {"topic": topic})
    reply = model.invoke([SystemMessage(system), HumanMessage(topic)], prompts.config_for(slot))
    ref = prompts.version(slot).ref
    return f"{NAMES[prompts.agent_id]}: {slot} -> {ref}: {system} => {reply.content}"


def build_graph() -> Any:
    supervisor_model = fake_model("ask research and review in parallel")
    researcher_model = fake_model("three sources found")
    reviewer_model = fake_model("claims checked")

    def supervise(state: State, config: RunnableConfig) -> State:
        return {"route": call(supervisor_model, config, "supervisor.system", state["topic"])}

    def search(state: State, config: RunnableConfig) -> State:
        return {"log": [call(researcher_model, config, "researcher.system", state["topic"])]}

    def check(state: State, config: RunnableConfig) -> State:
        return {"log": [call(reviewer_model, config, "reviewer.system", state["topic"])]}

    research = StateGraph(State)
    research.add_node("search", search)
    research.add_edge(START, "search")
    research_graph = research.compile()

    review = StateGraph(State)
    review.add_node("check", check)
    review.add_edge(START, "check")
    reviewer_graph = review.compile()

    def review_wrapper(state: State, config: RunnableConfig) -> State:
        result = reviewer_graph.invoke(
            {"topic": state["topic"], "log": []}, scope_config(config, REVIEWER_ID)
        )
        return {"log": result["log"]}

    builder = StateGraph(State)
    builder.add_node("supervise", supervise)
    builder.add_node("research", research_graph)
    builder.add_node("review", review_wrapper)
    builder.add_edge(START, "supervise")
    builder.add_edge("supervise", "research")
    builder.add_edge("supervise", "review")
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    client = Client()
    engine = client.prompts.local
    prompts = {
        "prm_supervisor": "Split the question about {topic} between your subagents.",
        "prm_researcher": "Collect sources about {topic}.",
        "prm_reviewer": "Check every claim about {topic}.",
    }
    for prompt_id, body in prompts.items():
        client.prompts.create(prompt_id, name=prompt_id, kind="text")
        client.prompts.publish(
            prompt_id, text_prompt(body), parent_version=None, change_message="First version"
        )

    researcher = engine.create_release(RESEARCHER_ID, {"researcher.system": "prm_researcher:1"})
    reviewer = engine.create_release(REVIEWER_ID, {"reviewer.system": "prm_reviewer:1"})
    supervisor = engine.create_release(
        SUPERVISOR_ID,
        {"supervisor.system": "prm_supervisor:1"},
        children={RESEARCHER_ID: researcher, REVIEWER_ID: reviewer},
    )
    engine.move_channel(SUPERVISOR_ID, "production", supervisor, expected_generation=0)

    managed = bind_langgraph(
        build_graph(),
        client=client,
        agent_id=SUPERVISOR_ID,
        channel="production",
        children={"research": RESEARCHER_ID},
    )
    config: RunnableConfig = {"configurable": {"thread_id": "case-7"}}
    result = managed.invoke({"topic": "solar panel subsidies", "log": []}, config)

    print("one binding pins the supervisor and both subagents:")
    print(f"  {result['route']}")
    for line in sorted(result["log"]):
        print(f"  {line}")

    by_agent = {line.split(":", 1)[0]: line for line in [result["route"], *result["log"]]}
    assert set(by_agent) == {"supervisor", "researcher", "reviewer"}
    assert "prm_supervisor:1" in by_agent["supervisor"]
    assert "prm_researcher:1" in by_agent["researcher"]
    assert "prm_reviewer:1" in by_agent["reviewer"]
    binding = managed.get_state(config).metadata["agenomic_binding_id"]
    assert binding.startswith("bnd_")
    print("binding:", binding)


if __name__ == "__main__":
    main()
