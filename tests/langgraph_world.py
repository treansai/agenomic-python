from __future__ import annotations

import asyncio
import json
import operator
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from prompt_fakes import AGENT, CHILD, WORKSPACE, FakePromptServer, chat_content, text_content
from typing_extensions import TypedDict

from agenomic._client import Client
from agenomic.crypto.signing import SigningKey
from agenomic.integrations.langgraph_binding import ManagedGraph, bind_langgraph, prompts_for
from agenomic.prompts.local import LocalPromptEngine

BASE = "https://registry.test"
ROOT_SLOTS = ("planner.instructions", "supervisor.system")


class LogState(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def publish_text(engine: LocalPromptEngine, prompt_id: str, *bodies: str) -> None:
    engine.create_prompt(prompt_id, kind="text", name=prompt_id)
    parent: Optional[int] = None
    for body in bodies:
        version = engine.publish(
            prompt_id, text_content(body), parent_version=parent, change_message="edit"
        )
        parent = version.ref.version


def seed(engine: LocalPromptEngine, tag: str = "") -> dict[str, str]:
    publish_text(engine, "prm_plan", f"{tag}PLAN v1", f"{tag}PLAN v2")
    publish_text(engine, "prm_sup", f"{tag}SUP v1", f"{tag}SUP v2")
    publish_text(engine, "prm_res", f"{tag}RES v1", f"{tag}RES v2")
    engine.create_prompt("prm_chat", kind="chat", name="Chat")
    for number in (1, 2):
        engine.publish(
            "prm_chat",
            chat_content(
                [
                    {"role": "system", "content": f"CHAT v{number} for {{customer}}"},
                    {"placeholder": "messages", "optional": True},
                ],
                {
                    "customer": {"type": "string", "required": True},
                    "messages": {"type": "messages", "required": False},
                },
            ),
            parent_version=None if number == 1 else 1,
            change_message="edit",
        )
    engine.create_prompt("prm_greet", kind="chat", name="Greeting")
    engine.publish(
        "prm_greet",
        chat_content([{"role": "system", "content": f"{tag}GREET"}]),
        parent_version=None,
        change_message="first",
    )
    releases: dict[str, str] = {}
    for number in (1, 2):
        child = engine.create_release(CHILD, {"researcher.system": f"prm_res:{number}"})
        releases[f"child_v{number}"] = child
        releases[f"v{number}"] = engine.create_release(
            AGENT,
            {
                "planner.instructions": f"prm_plan:{number}",
                "supervisor.system": f"prm_sup:{number}",
                "assistant.chat": f"prm_chat:{number}",
                "assistant.greeting": "prm_greet:1",
            },
            children={CHILD: child},
        )
    return releases


@dataclass
class World:
    engine: LocalPromptEngine
    releases: dict[str, str]
    server: FakePromptServer = field(init=False)

    def __post_init__(self) -> None:
        self.server = FakePromptServer(self.engine)

    @classmethod
    def create(
        cls,
        workspace_id: str = WORKSPACE,
        *,
        state_path: Any = None,
        promote: bool = True,
        tag: str = "",
    ) -> World:
        engine = LocalPromptEngine(workspace_id, state_path=state_path)
        world = cls(engine, seed(engine, tag))
        if promote:
            world.promote("v1")
        return world

    @classmethod
    def local(cls, workspace_id: str = WORKSPACE) -> tuple[World, Client]:
        client = Client(workspace_id=workspace_id)
        engine = client.prompts.local
        world = cls(engine, seed(engine))
        world.promote("v1")
        return world, client

    def promote(self, name: str, channel: str = "production") -> None:
        current = self.engine.get_channel(AGENT, channel)["generation"]
        self.engine.move_channel(AGENT, channel, self.releases[name], expected_generation=current)

    def export(self, path: Path, signer: SigningKey, **selector: str) -> Path:
        target = selector or {"channel": "production"}
        document = self.engine.export_bundle(AGENT, signer=signer, **target)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def client(self, **kwargs: Any) -> Client:
        return Client(
            api_key="agm_test", base_url=BASE, transport=self.server.transport(), **kwargs
        )

    def binding_posts(self) -> list[dict[str, Any]]:
        return [
            json.loads(request.content)
            for request in self.server.requests
            if request.method == "POST" and request.url.path.endswith("/bindings")
        ]

    def bind(self, graph: Any, **kwargs: Any) -> ManagedGraph:
        options: dict[str, Any] = {"agent_id": AGENT, "channel": "production"}
        options.update(kwargs)
        if "client" not in options and not options.get("offline") and "binding" not in options:
            options["client"] = self.client()
        return bind_langgraph(graph, **options)


def plan_node(state: LogState, config: RunnableConfig) -> LogState:
    prompts = prompts_for(config)
    return {"log": [f"plan:{prompts.render_text('planner.instructions')}"]}


def supervise_node(state: LogState, config: RunnableConfig) -> LogState:
    prompts = prompts_for(config)
    return {"log": [f"sup:{prompts.render_text('supervisor.system')}"]}


def two_node_graph(checkpointer: Any = None) -> Any:
    builder = StateGraph(LogState)
    builder.add_node("plan", plan_node)
    builder.add_node("supervise", supervise_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "supervise")
    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver()
    )


def thread(thread_id: str, **configurable: Any) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, **configurable}}


PRODUCED: list[str] = []


class StreamingFakeChatModel(GenericFakeChatModel):
    delay: float = 0.01

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for index, chunk in enumerate(super()._stream(*args, **kwargs)):
            PRODUCED.append(f"P{index}")
            time.sleep(self.delay)
            yield chunk

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message = next(self.messages)
        words = str(message.content).split(" ")
        for index, word in enumerate(words):
            await asyncio.sleep(self.delay)
            PRODUCED.append(f"P{index}")
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word if index == 0 else " " + word)
            )


def streaming_model(text: str = "one two three four five six", count: int = 50) -> Any:
    return StreamingFakeChatModel(messages=iter([AIMessage(content=text)] * count))
