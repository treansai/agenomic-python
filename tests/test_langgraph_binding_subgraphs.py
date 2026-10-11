from __future__ import annotations

import asyncio
import sys
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt
from langgraph_world import LogState, World, plan_node, thread
from prompt_fakes import CHILD, OTHER_AGENT
from typing_extensions import TypedDict

from agenomic.integrations.langgraph_binding import prompts_for, scope_config
from agenomic.prompts import PromptBindingError


def research(state: LogState, config: RunnableConfig) -> LogState:
    prompts = prompts_for(config)
    marker = "child" if prompts.agent_id == CHILD else "root"
    return {"log": [f"res:{marker}:{prompts.render_text('researcher.system')}"]}


def child_graph(checkpointer: Any = None) -> Any:
    builder = StateGraph(LogState)
    builder.add_node("research", research)
    builder.add_edge(START, "research")
    return builder.compile(checkpointer=checkpointer)


def root_with(node: Any, name: str = "research") -> Any:
    builder = StateGraph(LogState)
    builder.add_node("plan", plan_node)
    builder.add_node(name, node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", name)
    return builder.compile(checkpointer=InMemorySaver())


def promote_child(world: World, name: str) -> None:
    current = world.engine.get_channel(CHILD, "production")["generation"]
    world.engine.move_channel(
        CHILD, "production", world.releases[name], expected_generation=current
    )


def test_subgraph_node_uses_child_pin_by_node_path() -> None:
    world = World.create()
    managed = world.bind(root_with(child_graph()), children={"research": CHILD})
    first = managed.invoke({"log": []}, thread("sub"))["log"]
    assert first[0] == "plan:PLAN v1"
    assert first[-1] == "res:child:RES v1"
    world.promote("v2")
    assert managed.invoke({"log": []}, thread("sub2"))["log"][-1] == "res:child:RES v2"
    assert managed.invoke({"log": []}, thread("sub"))["log"][-1] == "res:child:RES v1"


def test_wrapper_node_scope_config() -> None:
    world = World.create()
    plain = child_graph()

    def wrap_scoped(state: LogState, config: RunnableConfig) -> LogState:
        return {"log": plain.invoke({"log": []}, scope_config(config, CHILD))["log"]}

    def wrap_mapped(state: LogState, config: RunnableConfig) -> LogState:
        return {"log": plain.invoke({"log": []}, config)["log"]}

    scoped = world.bind(root_with(wrap_scoped, "wrap"))
    assert scoped.invoke({"log": []}, thread("w1"))["log"][-1] == "res:child:RES v1"
    mapped = world.bind(root_with(wrap_mapped, "wrap"), children={"wrap": CHILD})
    assert mapped.invoke({"log": []}, thread("w2"))["log"][-1] == "res:child:RES v1"
    unmapped = world.bind(root_with(wrap_mapped, "wrap"))
    with pytest.raises(PromptBindingError) as root_scope:
        unmapped.invoke({"log": []}, thread("w3"))
    assert root_scope.value.code == "slot_not_in_manifest"


class Messages(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


def test_tool_subagent_scope_config() -> None:
    world = World.create()
    plain = child_graph()

    @tool
    def ask_researcher(question: str, config: RunnableConfig) -> str:
        """Ask the research subagent."""
        return str(plain.invoke({"log": []}, scope_config(config, CHILD))["log"][0])

    def agent(state: Messages, config: RunnableConfig) -> Messages:
        call = {"name": "ask_researcher", "args": {"question": "q"}, "id": "call_1"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(Messages)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([ask_researcher]))
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    out = managed.invoke({"messages": [("user", "go")]}, thread("tool"))
    tool_messages = [message.content for message in out["messages"] if message.type == "tool"]
    assert tool_messages == ["res:child:RES v1"]


def test_parallel_send_workers_no_leakage() -> None:
    world = World.create()

    def fan(state: LogState) -> list[Send]:
        return [Send("worker", {"log": [owner]}) for owner in ("root", "child") * 3]

    def worker(state: LogState, config: RunnableConfig) -> LogState:
        owner = state["log"][-1]
        if owner == "child":
            prompts = prompts_for(config, agent_id=CHILD)
            text = prompts.render_text("researcher.system")
        else:
            prompts = prompts_for(config)
            text = prompts.render_text("planner.instructions")
        return {"log": [f"{owner}:{text}"]}

    builder = StateGraph(LogState)
    builder.add_node("worker", worker)
    builder.add_conditional_edges(START, fan, ["worker"])
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    out = managed.invoke({"log": []}, thread("fan"))
    workers = sorted(item for item in out["log"] if ":" in item)
    assert workers == ["child:RES v1"] * 3 + ["root:PLAN v1"] * 3

    async def concurrent() -> list[dict[str, Any]]:
        first = asyncio.create_task(managed.ainvoke({"log": []}, thread("fan-a")))
        await asyncio.sleep(0)
        world.promote("v2")
        second = asyncio.create_task(managed.ainvoke({"log": []}, thread("fan-b")))
        return list(await asyncio.gather(first, second))

    first, second = asyncio.run(concurrent())
    assert sorted(item for item in first["log"] if ":" in item) == workers
    assert sorted(item for item in second["log"] if ":" in item) == (
        ["child:RES v2"] * 3 + ["root:PLAN v2"] * 3
    )


def test_child_not_pinned_fails_closed() -> None:
    world = World.create()
    errors: list[str] = []

    def probe(state: LogState, config: RunnableConfig) -> LogState:
        for attempt in (
            lambda: prompts_for(config, agent_id=OTHER_AGENT),
            lambda: scope_config(config, OTHER_AGENT),
        ):
            try:
                attempt()
            except PromptBindingError as error:
                errors.append(error.code)
        return {"log": ["probed"]}

    managed = world.bind(root_with(probe, "probe"), children={"elsewhere": OTHER_AGENT})
    assert managed.invoke({"log": []}, thread("np"))["log"][-1] == "probed"
    assert errors == ["child_agent_not_pinned", "child_agent_not_pinned"]
    mapped = world.bind(root_with(child_graph()), children={"research": OTHER_AGENT})
    with pytest.raises(PromptBindingError) as raised:
        mapped.invoke({"log": []}, thread("np2"))
    assert raised.value.code == "child_agent_not_pinned"


def test_nested_managed_graph_scope_switch_no_new_binding() -> None:
    world = World.create()
    child = world.bind(child_graph(), agent_id=CHILD)
    root = world.bind(root_with(child))
    out = root.invoke({"log": []}, thread("nested"))
    assert out["log"][0] == "plan:PLAN v1"
    assert out["log"][-1] == "res:child:RES v1"
    assert [post["thread_key"] for post in world.binding_posts()] == [
        world.binding_posts()[0]["thread_key"]
    ]
    assert world.binding_posts()[0]["selector"] == {"channel": "production"}


def test_subgraph_interrupt_resume_keeps_child_pin() -> None:
    world = World.create()

    def ask(state: LogState, config: RunnableConfig) -> LogState:
        prompts = prompts_for(config)
        before = prompts.render_text("researcher.system")
        answer = interrupt("continue?")
        after = prompts_for(config).render_text("researcher.system")
        return {"log": [f"child-ask:{answer}:{before}:{after}"]}

    builder = StateGraph(LogState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    saver = InMemorySaver()
    root_builder = StateGraph(LogState)
    root_builder.add_node("research", builder.compile())
    root_builder.add_edge(START, "research")
    graph = root_builder.compile(checkpointer=saver)
    managed = world.bind(graph, children={"research": CHILD})
    paused = managed.invoke({"log": []}, thread("si"))
    assert [item.value for item in paused["__interrupt__"]] == ["continue?"]
    world.promote("v2")
    promote_child(world, "child_v2")
    restarted = world.bind(graph, children={"research": CHILD}, client=world.client())
    resumed = restarted.invoke(Command(resume="go"), thread("si"))
    assert resumed["log"] == ["child-ask:go:RES v1:RES v1"]
    snapshot = restarted.get_state(thread("si"), subgraphs=True)
    assert snapshot.next == ()


def test_checkpointer_false_child_unchanged() -> None:
    world = World.create()
    detached = child_graph(checkpointer=False)
    raw = root_with(detached)
    managed = world.bind(raw, children={"research": CHILD})
    assert managed.invoke({"log": []}, thread("cf"))["log"][-1] == "res:child:RES v1"
    assert [name for name, _ in managed.get_subgraphs()] == [
        name for name, _ in raw.get_subgraphs()
    ]
    assert detached.checkpointer is False
    assert managed.checkpointer is raw.checkpointer


def test_child_own_thread_keeps_parent_pin_after_child_promotion() -> None:
    world = World.create()
    promote_child(world, "child_v1")
    child_saver = InMemorySaver()
    child = world.bind(child_graph(child_saver), agent_id=CHILD)

    def delegate(state: LogState, config: RunnableConfig) -> LogState:
        scoped = scope_config(config, CHILD, thread_id="child-thread")
        assert "checkpoint_ns" not in scoped["configurable"]
        assert not any(key.startswith("__pregel") for key in scoped["configurable"])
        return {"log": child.invoke({"log": []}, scoped)["log"][-1:]}

    root = world.bind(root_with(delegate, "delegate"))
    assert root.invoke({"log": []}, thread("parent"))["log"][-1] == "res:child:RES v1"
    root_posts = len(world.binding_posts())
    promote_child(world, "child_v2")
    assert root.invoke({"log": []}, thread("parent"))["log"][-1] == "res:child:RES v1"
    assert len(world.binding_posts()) == root_posts + 1
    assert all(post["selector"] == {"channel": "production"} for post in world.binding_posts())
    child_state = child_saver.get_tuple({"configurable": {"thread_id": "child-thread"}})
    assert child_state is not None
    assert child_state.metadata["agenomic_agent_scope"] == CHILD
    standalone = child.invoke({"log": []}, thread("standalone"))
    assert standalone["log"] == ["res:child:RES v2"]


def test_nested_bind_without_pin_refused() -> None:
    world = World.create()
    child = world.bind(child_graph(InMemorySaver()), agent_id=CHILD)

    def fresh_config(state: LogState, config: RunnableConfig) -> LogState:
        return {"log": child.invoke({"log": []}, {"configurable": {"thread_id": "x"}})["log"]}

    def no_config(state: LogState, config: RunnableConfig) -> LogState:
        return {"log": child.invoke({"log": []})["log"]}

    for node in (fresh_config, no_config):
        managed = world.bind(root_with(node, "delegate"))
        with pytest.raises(PromptBindingError) as raised:
            managed.invoke({"log": []}, thread(node.__name__))
        assert raised.value.code == "nested_bind_unsupported"
    assert len(world.binding_posts()) == 2


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="the run context variable is not propagated into asyncio tasks before Python 3.11",
)
def test_nested_bind_without_pin_refused_async() -> None:
    world = World.create()
    child = world.bind(child_graph(InMemorySaver()), agent_id=CHILD)

    async def fresh_config(state: LogState, config: RunnableConfig) -> LogState:
        out = await child.ainvoke({"log": []}, {"configurable": {"thread_id": "x"}})
        return {"log": out["log"]}

    managed = world.bind(root_with(fresh_config, "delegate"))
    with pytest.raises(PromptBindingError) as raised:
        asyncio.run(managed.ainvoke({"log": []}, thread("async")))
    assert raised.value.code == "nested_bind_unsupported"
