from __future__ import annotations

import asyncio
import itertools
from contextlib import aclosing
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agenomic import Client
from agenomic.integrations import bind_langgraph, prompts_for

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
REPLY = "Your parcel left the depot this morning and arrives tomorrow."


def chat_prompt(system: str) -> dict[str, Any]:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "chat",
        "body": [
            {"role": "system", "content": system},
            {"placeholder": "messages", "optional": True},
        ],
        "variables": {
            "customer": {"type": "string", "required": True},
            "messages": {"type": "messages", "required": False},
        },
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


class State(TypedDict, total=False):
    customer: str
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph() -> Any:
    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content=REPLY)]))

    async def answer(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        messages = prompts.render_messages(
            "assistant.chat", {"customer": state["customer"], "messages": state["messages"]}
        )
        reply = await model.ainvoke(messages, prompts.config_for("assistant.chat"))
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    return builder.compile(checkpointer=InMemorySaver())


def question() -> State:
    return {"customer": "Acme", "messages": [HumanMessage("Where is my parcel?")]}


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def main() -> None:
    client = Client()
    client.prompts.create("prm_assistant", name="Assistant", kind="chat")
    client.prompts.publish(
        "prm_assistant",
        chat_prompt("You answer delivery questions for {customer}."),
        parent_version=None,
        change_message="First version",
    )
    engine = client.prompts.local
    release_id = engine.create_release(AGENT_ID, {"assistant.chat": "prm_assistant:1"})
    engine.move_channel(AGENT_ID, "production", release_id, expected_generation=0)
    managed = bind_langgraph(build_graph(), client=client, agent_id=AGENT_ID, channel="production")

    result = await managed.ainvoke(question(), thread("async-invoke"))
    print("ainvoke:", result["messages"][-1].content)
    assert result["messages"][-1].content == REPLY

    tokens: list[str] = []
    refs: set[str] = set()
    print("astream messages: ", end="")
    async with aclosing(
        managed.astream(question(), thread("async-stream"), stream_mode="messages")
    ) as stream:
        async for chunk, metadata in stream:
            if chunk.content:
                tokens.append(str(chunk.content))
                refs.add(str(metadata.get("agenomic_prompt_refs")))
                print(f"[{chunk.content}]", end="", flush=True)
    print()
    print("token chunks:", len(tokens), "prompt refs in stream metadata:", sorted(refs))
    assert len(tokens) > 1
    assert "".join(tokens) == REPLY
    assert refs == {"prm_assistant:1"}

    streamed: list[str] = []
    releases: set[str] = set()
    async with aclosing(
        managed.astream_events(question(), thread("async-events"), version="v2")
    ) as events:
        async for event in events:
            if event["event"] == "on_chat_model_stream":
                streamed.append(str(event["data"]["chunk"].content))
                releases.add(str(event["metadata"].get("agenomic_release_id")))
    print("astream_events v2 chat chunks:", len(streamed), "release:", sorted(releases))
    assert len([part for part in streamed if part]) > 1
    assert "".join(streamed) == REPLY
    assert releases == {release_id}


if __name__ == "__main__":
    asyncio.run(main())
