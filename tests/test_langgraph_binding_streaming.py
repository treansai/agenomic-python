from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import warnings
from typing import Annotated, Any
from uuid import UUID

import langgraph_world
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel import Pregel
from langgraph.types import StreamWriter
from langgraph_world import World, streaming_model, thread
from typing_extensions import TypedDict

from agenomic.integrations.langgraph_binding import (
    INFLIGHT,
    SET_KEY,
    counters,
    prompts_for,
)
from agenomic.prompts import PinnedPromptSet

MODES = ("values", "updates", "messages", "custom", "tasks", "checkpoints", "debug")
SUPPORTS_V2 = "version" in inspect.signature(Pregel.stream).parameters


class Chat(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


def sync_talk(model: Any) -> Any:
    def talk(state: Chat, config: RunnableConfig, writer: StreamWriter) -> Chat:
        prompts = prompts_for(config)
        writer({"progress": "rendering"})
        messages = prompts.render_messages(
            "assistant.chat", {"customer": "ACME", "messages": state["messages"]}
        )
        return {"messages": [model.invoke(messages, prompts.config_for("assistant.chat"))]}

    return talk


def async_talk(model: Any) -> Any:
    async def talk(state: Chat, config: RunnableConfig) -> Chat:
        prompts = prompts_for(config)
        messages = prompts.render_messages(
            "assistant.chat", {"customer": "ACME", "messages": state["messages"]}
        )
        reply = await model.ainvoke(messages, prompts.config_for("assistant.chat"))
        return {"messages": [reply]}

    return talk


def chat_graph(node: Any, *, nested: bool = False) -> Any:
    builder = StateGraph(Chat)
    if nested:
        inner = StateGraph(Chat)
        inner.add_node("talk", node)
        inner.add_edge(START, "talk")
        builder.add_node("inner", inner.compile())
        builder.add_edge(START, "inner")
    else:
        builder.add_node("talk", node)
        builder.add_edge(START, "talk")
    return builder.compile(checkpointer=InMemorySaver())


def hello() -> dict[str, Any]:
    return {"messages": [HumanMessage("hello")]}


def leaks(value: Any) -> list[str]:
    found: list[str] = []
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, PinnedPromptSet):
            found.append(repr(item))
        elif isinstance(item, dict):
            if SET_KEY in item:
                found.append(SET_KEY)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return found


def stream_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [{"stream_mode": mode} for mode in MODES]
    variants.append({"stream_mode": list(MODES)})
    variants.append({"stream_mode": list(MODES), "subgraphs": True})
    variants.append({"stream_mode": "checkpoints", "subgraphs": True})
    variants.append({"stream_mode": "debug", "subgraphs": True})
    if SUPPORTS_V2:
        variants.append({"stream_mode": list(MODES), "subgraphs": True, "version": "v2"})
        variants.append({"stream_mode": "debug", "version": "v2"})
    return variants


def test_stream_modes_passthrough() -> None:
    world = World.create()
    managed = world.bind(chat_graph(sync_talk(streaming_model()), nested=True))
    for index, options in enumerate(stream_variants()):
        chunks = list(managed.stream(hello(), thread(f"modes-{index}"), **options))
        assert chunks or options == {"stream_mode": "custom"}, options
        mode = options["stream_mode"]
        if options.get("version") == "v2":
            assert all(set(chunk) >= {"type", "ns", "data"} for chunk in chunks)
        elif isinstance(mode, list):
            width = 3 if options.get("subgraphs") else 2
            assert all(isinstance(chunk, tuple) and len(chunk) == width for chunk in chunks)
            assert {chunk[-2] for chunk in chunks} == set(MODES) - (
                set() if options.get("subgraphs") else {"custom"}
            )
        elif options.get("subgraphs"):
            assert all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)
            assert any(chunk[0] for chunk in chunks)
    messages = list(managed.stream(hello(), thread("messages"), stream_mode="messages"))
    _, metadata = messages[0]
    assert metadata["agenomic_binding_id"].startswith("bnd_")
    custom = list(managed.stream(hello(), thread("custom"), stream_mode="custom", subgraphs=True))
    assert [chunk[1] for chunk in custom] == [{"progress": "rendering"}]
    values = list(managed.stream(hello(), thread("values"), stream_mode="values"))
    assert values[-1]["messages"][-1].content == "one two three four five six"


def test_every_stream_mode_chunk_json_serializable_without_prompt_set() -> None:
    world = World.create()
    managed = world.bind(chat_graph(sync_talk(streaming_model()), nested=True))
    for index, options in enumerate(stream_variants()):
        for chunk in managed.stream(hello(), thread(f"json-{index}"), **options):
            json.dumps(chunk, default=str)
            assert leaks(chunk) == [], options
    result = managed.invoke(hello(), thread("invoke-debug"), stream_mode="debug")
    assert isinstance(result, list)
    assert leaks(result) == []

    async_managed = world.bind(chat_graph(async_talk(streaming_model()), nested=True))

    async def collect() -> list[Any]:
        return [
            chunk
            async for chunk in async_managed.astream(
                hello(), thread("async-json"), stream_mode=list(MODES), subgraphs=True
            )
        ]

    assert leaks(asyncio.run(collect())) == []


def interleaving() -> bool:
    log = langgraph_world.PRODUCED
    produced = [index for index, item in enumerate(log) if item.startswith("P")]
    consumed = [index for index, item in enumerate(log) if item.startswith("C")]
    return bool(produced and consumed) and consumed[0] < produced[-1]


def test_messages_tokens_not_buffered() -> None:
    world = World.create()
    managed = world.bind(chat_graph(async_talk(streaming_model())))
    langgraph_world.PRODUCED.clear()

    async def consume() -> None:
        index = 0
        async for _chunk, metadata in managed.astream(
            hello(), thread("tokens"), stream_mode="messages"
        ):
            assert metadata["agenomic_prompt_slots"] == "assistant.chat"
            langgraph_world.PRODUCED.append(f"C{index}")
            index += 1

    asyncio.run(consume())
    assert interleaving()
    sync_managed = world.bind(chat_graph(sync_talk(streaming_model())))
    langgraph_world.PRODUCED.clear()
    for index, _ in enumerate(
        sync_managed.stream(hello(), thread("sync-tokens"), stream_mode="messages")
    ):
        langgraph_world.PRODUCED.append(f"C{index}")
    assert interleaving()


def test_astream_events_v2_not_buffered_and_carries_pin() -> None:
    world = World.create()
    managed = world.bind(chat_graph(async_talk(streaming_model())))
    langgraph_world.PRODUCED.clear()
    seen: list[dict[str, Any]] = []

    async def consume() -> None:
        index = 0
        async for event in managed.astream_events(hello(), thread("events"), version="v2"):
            if event["event"] == "on_chat_model_stream":
                seen.append(event["metadata"])
                langgraph_world.PRODUCED.append(f"C{index}")
                index += 1

    asyncio.run(consume())
    assert interleaving()
    assert seen[0]["agenomic_prompt_refs"] == "prm_chat:1"
    assert seen[0]["agenomic_binding_id"].startswith("bnd_")
    assert counters()[INFLIGHT] == 0


def test_astream_events_v3_passthrough() -> None:
    world = World.create()
    managed = world.bind(chat_graph(async_talk(streaming_model())))
    plain = StateGraph(Chat)
    plain.add_node("noop", lambda state: {"messages": []})
    plain.add_edge(START, "noop")
    raw = plain.compile()

    async def attempt(graph: Any, config: Any) -> list[Any]:
        stream = await graph.astream_events(hello(), config, version="v3")
        return [item async for item in stream]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            asyncio.run(attempt(raw, None))
        except NotImplementedError:
            with pytest.raises(NotImplementedError):
                asyncio.run(attempt(managed, thread("v3")))
            return
        items = asyncio.run(attempt(managed, thread("v3")))
    assert items
    assert managed.get_state(thread("v3")).metadata["agenomic_binding_id"].startswith("bnd_")


def test_sync_stream_events_passthrough() -> None:
    world = World.create()
    managed = world.bind(chat_graph(sync_talk(streaming_model())))
    with pytest.raises(NotImplementedError):
        list(managed.stream_events(hello(), thread("sync-v2"), version="v2"))
    assert len(world.binding_posts()) == 1


def test_cancel_astream_cleans_up() -> None:
    world = World.create()
    model = streaming_model(" ".join(f"w{index}" for index in range(40)))
    managed = world.bind(chat_graph(async_talk(model)))
    langgraph_world.PRODUCED.clear()

    async def run() -> int:
        async def consume() -> None:
            async for _ in managed.astream(hello(), thread("cancel"), stream_mode="messages"):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert counters()[INFLIGHT] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        produced = len(langgraph_world.PRODUCED)
        await asyncio.sleep(0.1)
        return len(langgraph_world.PRODUCED) - produced

    assert asyncio.run(run()) == 0
    assert counters()[INFLIGHT] == 0
    assert managed.get_state(thread("cancel")).next == ("talk",)


def test_aclosing_early_close_cleans_up() -> None:
    world = World.create()
    managed = world.bind(chat_graph(async_talk(streaming_model())))

    async def run() -> int:
        async with contextlib.aclosing(
            managed.astream(hello(), thread("early"), stream_mode="messages")
        ) as stream:
            async for _ in stream:
                break
        return counters()[INFLIGHT]

    assert asyncio.run(run()) == 0
    sync_managed = world.bind(chat_graph(sync_talk(streaming_model())))
    with contextlib.closing(
        sync_managed.stream(hello(), thread("early-sync"), stream_mode="messages")
    ) as stream:
        for _ in stream:
            assert counters()[INFLIGHT] == 1
            break
    assert counters()[INFLIGHT] == 0


def test_wait_for_timeout_propagates() -> None:
    world = World.create()
    finished: list[str] = []

    async def slow(state: Chat, config: RunnableConfig) -> Chat:
        prompts_for(config).render_messages("assistant.chat", {"customer": "ACME"})
        await asyncio.sleep(1.0)
        finished.append("done")
        return {"messages": []}

    managed = world.bind(chat_graph(slow))

    async def run() -> None:
        await asyncio.wait_for(managed.ainvoke(hello(), thread("timeout")), timeout=0.05)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    assert counters()[INFLIGHT] == 0
    assert finished == []


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, metadata: Any = None, **kw: Any
    ) -> None:
        self.metadata.append(dict(metadata or {}))


def test_config_for_slot_refs_reach_callbacks_and_stream() -> None:
    world = World.create()
    managed = world.bind(chat_graph(async_talk(streaming_model())))
    recorder = Recorder()
    config = {**thread("refs"), "callbacks": [recorder]}

    async def consume() -> list[dict[str, Any]]:
        return [
            metadata
            async for _chunk, metadata in managed.astream(hello(), config, stream_mode="messages")
        ]

    streamed = asyncio.run(consume())
    release = world.engine.get_release(world.releases["v1"])
    digest = release["manifest"]["slots"]["assistant.chat"]["content_digest"]
    for metadata in (recorder.metadata[0], streamed[0]):
        assert metadata["agenomic_prompt_slots"] == "assistant.chat"
        assert metadata["agenomic_prompt_refs"] == "prm_chat:1"
        assert metadata["agenomic_prompt_content_digests"] == digest
        assert metadata["agenomic_rendered_hash"].startswith("sha256:")
    snapshot = managed.get_state(thread("refs"))
    assert "agenomic_prompt_slots" not in snapshot.metadata
    assert "agenomic_rendered_hash" not in snapshot.metadata
    assert snapshot.metadata["agenomic_prompt_manifest_digest"] == release["prompt_manifest_digest"]
