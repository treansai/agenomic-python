from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph_world import LogState, World, plan_node, thread, two_node_graph

from agenomic.integrations.langgraph_binding import AgentFactory, managed_prompt, prompts_for
from agenomic.prompts import PromptBindingError, PromptRenderError

pytestmark = pytest.mark.filterwarnings("ignore::langgraph.warnings.LangGraphDeprecatedSinceV10")


class ToolFreeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def model() -> ToolFreeModel:
    return ToolFreeModel(messages=iter([AIMessage(content=f"answer {n}") for n in range(20)]))


class Prompts(BaseCallbackHandler):
    def __init__(self) -> None:
        self.systems: list[str] = []

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        for message in messages[0]:
            if isinstance(message, SystemMessage):
                self.systems.append(str(message.content))


def turn(text: str) -> dict[str, Any]:
    return {"messages": [("user", text)]}


def test_managed_prompt_runnable_in_react_agent() -> None:
    world = World.create()
    recorder = Prompts()
    agent = create_react_agent(
        model(), [], prompt=managed_prompt("planner.instructions"), checkpointer=InMemorySaver()
    )
    managed = world.bind(agent)
    managed.invoke(turn("hi"), {**thread("T"), "callbacks": [recorder]})
    world.promote("v2")
    managed.invoke(turn("again"), {**thread("T"), "callbacks": [recorder]})
    managed.invoke(turn("new"), {**thread("N"), "callbacks": [recorder]})
    assert recorder.systems == ["PLAN v1", "PLAN v1", "PLAN v2"]
    chat_recorder = Prompts()
    chat_agent = create_react_agent(
        model(),
        [],
        prompt=managed_prompt("assistant.chat", variables={"customer": "ACME"}),
        checkpointer=InMemorySaver(),
    )
    world.bind(chat_agent).invoke(turn("hello"), {**thread("C"), "callbacks": [chat_recorder]})
    assert chat_recorder.systems == ["CHAT v2 for ACME"]
    computed = create_react_agent(
        model(),
        [],
        prompt=managed_prompt(
            "assistant.chat", variables=lambda state: {"customer": f"n={len(state['messages'])}"}
        ),
        checkpointer=InMemorySaver(),
    )
    computed_recorder = Prompts()
    world.bind(computed).invoke(turn("x"), {**thread("D"), "callbacks": [computed_recorder]})
    assert computed_recorder.systems == ["CHAT v2 for n=1"]


def test_managed_prompt_history_conflict() -> None:
    world = World.create()
    agent = create_react_agent(
        model(),
        [],
        prompt=managed_prompt(
            "assistant.chat", variables={"customer": "ACME"}, history_key="history"
        ),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(PromptRenderError) as raised:
        world.bind(agent).invoke(turn("hi"), thread("H"))
    assert raised.value.details["reason"] == "history_conflict"


def test_no_system_message_in_checkpointed_history() -> None:
    world = World.create()
    saver = InMemorySaver()
    agent = create_react_agent(
        model(), [], prompt=managed_prompt("planner.instructions"), checkpointer=saver
    )
    managed = world.bind(agent)
    managed.invoke(turn("hi"), thread("S"))
    managed.invoke(turn("again"), thread("S"))
    messages = managed.get_state(thread("S")).values["messages"]
    assert [message.type for message in messages] == ["human", "ai", "human", "ai"]
    stored = repr(dict(saver.storage)) + repr(dict(saver.writes)) + repr(dict(saver.blobs))
    assert "PLAN v1" not in stored


def test_factory_cached_by_digest() -> None:
    world = World.create()
    saver = InMemorySaver()
    builds: list[str] = []
    recorder = Prompts()

    def build(prompts: Any) -> Any:
        builds.append(prompts.prompt_manifest_digest)
        text = prompts.version("planner.instructions").render_text({})
        return create_react_agent(model(), [], prompt=text, checkpointer=saver)

    factory = AgentFactory(build, max_entries=4)
    managed = world.bind(factory)
    with pytest.raises(PromptBindingError) as early:
        managed.get_graph()
    assert early.value.code == "prompt_set_unavailable"
    with pytest.raises(AttributeError):
        _ = managed.nodes
    managed.invoke(turn("a"), {**thread("T1"), "callbacks": [recorder]})
    managed.invoke(turn("b"), {**thread("T2"), "callbacks": [recorder]})
    assert len(builds) == 1
    world.promote("v2")
    managed.invoke(turn("c"), {**thread("T3"), "callbacks": [recorder]})
    managed.invoke(turn("d"), {**thread("T1"), "callbacks": [recorder]})
    assert len(builds) == 2
    assert recorder.systems == ["PLAN v1", "PLAN v1", "PLAN v2", "PLAN v1"]
    assert managed.get_state(thread("T1")).values["messages"][-1].content.startswith("answer")
    assert set(managed.nodes) == {"__start__", "agent"}
    assert managed.get_graph() is not None
    tagged = managed.with_config({"tags": ["factory"], "configurable": {"thread_id": "T4"}})
    tagged.invoke(turn("e"), {"callbacks": [recorder]})
    assert recorder.systems[-1] == "PLAN v2"
    fresh = world.bind(AgentFactory(build))
    assert fresh.get_state(thread("T1")).values["messages"][0].content == "a"
    execution = world.bind(AgentFactory(build), pin_scope="execution")
    with pytest.raises(PromptBindingError) as unavailable:
        execution.get_state(thread("T1"))
    assert unavailable.value.code == "prompt_set_unavailable"


def test_factory_topology_mismatch() -> None:
    world = World.create()
    saver = InMemorySaver()

    def build(prompts: Any) -> Any:
        builder = StateGraph(LogState)
        builder.add_node("plan", plan_node)
        if prompts.prompt_manifest_digest == digest_v2:
            builder.add_node("extra", plan_node)
            builder.add_edge("plan", "extra")
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=saver)

    digest_v2 = world.engine.get_release(world.releases["v2"])["prompt_manifest_digest"]
    managed = world.bind(AgentFactory(build))
    managed.invoke({"log": []}, thread("A"))
    world.promote("v2")
    with pytest.raises(PromptBindingError) as raised:
        managed.invoke({"log": []}, thread("B"))
    assert raised.value.code == "factory_topology_mismatch"
    savers = iter([InMemorySaver(), InMemorySaver()])
    other = world.bind(AgentFactory(lambda prompts: two_node_graph(next(savers))))
    other.invoke({"log": []}, thread("C"))
    world.promote("v1")
    with pytest.raises(PromptBindingError) as checkpointer:
        other.invoke({"log": []}, thread("D"))
    assert checkpointer.value.code == "factory_topology_mismatch"
    with pytest.raises(ValueError):
        AgentFactory(build, max_entries=0)


def ask(state: LogState, config: RunnableConfig) -> LogState:
    before = prompts_for(config).render_text("planner.instructions")
    answer = interrupt("approve?")
    return {"log": [f"ask:{answer}:{before}"]}


def test_factory_execution_scope_resume_needs_a_built_graph() -> None:
    world = World.create()
    saver = InMemorySaver()

    def build(prompts: Any) -> Any:
        builder = StateGraph(LogState)
        builder.add_node("plan", plan_node)
        builder.add_node("ask", ask)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "ask")
        return builder.compile(checkpointer=saver)

    first = world.bind(AgentFactory(build), pin_scope="execution")
    paused = first.invoke({"log": []}, thread("E", agenomic_execution_key="req-1"))
    assert [item.value for item in paused["__interrupt__"]] == ["approve?"]
    world.promote("v2")
    restarted = world.bind(AgentFactory(build), pin_scope="execution", client=world.client())
    posts = len(world.binding_posts())
    for attempt in (
        lambda: restarted.invoke(Command(resume="ok"), thread("E")),
        lambda: restarted.update_state(thread("E"), {"log": ["x"]}, as_node="ask"),
    ):
        with pytest.raises(PromptBindingError) as raised:
            attempt()
        assert raised.value.code == "prompt_set_unavailable"
    assert len(world.binding_posts()) == posts
    fresh = restarted.invoke({"log": []}, thread("F", agenomic_execution_key="req-2"))
    assert fresh["log"] == ["plan:PLAN v2"]
    resumed = restarted.invoke(Command(resume="ok"), thread("E"))
    assert resumed["log"] == ["plan:PLAN v1", "ask:ok:PLAN v1"]


def test_factory_async_state_reads_build_from_the_thread_binding() -> None:
    import asyncio

    world = World.create()
    saver = InMemorySaver()

    def build(prompts: Any) -> Any:
        text = prompts.version("planner.instructions").render_text({})
        return create_react_agent(model(), [], prompt=text, checkpointer=saver)

    world.bind(AgentFactory(build)).invoke(turn("a"), thread("T"))
    fresh = world.bind(AgentFactory(build))

    async def run() -> tuple[Any, list[Any], Any]:
        snapshot = await fresh.aget_state(thread("T"))
        history = [item async for item in fresh.aget_state_history(thread("T"))]
        graph = await fresh.aget_graph()
        return snapshot, history, graph

    snapshot, history, graph = asyncio.run(run())
    assert snapshot.values["messages"][0].content == "a"
    assert history
    assert graph is not None
