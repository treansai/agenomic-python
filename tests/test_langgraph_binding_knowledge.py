from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

pytest.importorskip("langgraph.prebuilt")

from knowledge_fakes import AGENT, KB, FakeKnowledgeApi, body_of  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from langgraph_world import World, thread  # noqa: E402

from agenomic.integrations import knowledge_tool  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self


def model(tool_name: str) -> ToolCallingModel:
    replies: Iterator[AIMessage] = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": {"query": "refund window"}, "id": "call_kb_1"}
                ],
            ),
            AIMessage(content="Customers have 14 days to ask for a refund [e1]."),
        ]
    )
    return ToolCallingModel(messages=replies)


def test_create_react_agent_calls_the_knowledge_tool() -> None:
    api = FakeKnowledgeApi()
    tool = knowledge_tool(KB, "v3", client=api.client())
    saver = InMemorySaver()
    agent = create_react_agent(model(tool.name), [tool], checkpointer=saver)
    config: Any = {"configurable": {"thread_id": "kb-thread"}}
    result = agent.invoke({"messages": [HumanMessage(content="refund window?")]}, config)
    messages = result["messages"]
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.tool_call_id == "call_kb_1"
    assert "<knowledge_evidence" in tool_message.content
    assert "Citations:" in tool_message.content
    assert messages[-1].content.endswith("[e1].")
    assert body_of(api.last("POST", "/search$"))["version"] == 3
    state = agent.get_state(config)
    assert set(state.values) == {"messages"}
    assert not [key for key in state.config["configurable"] if "knowledge" in key]


def test_async_agent_with_agent_scope_and_no_binding() -> None:
    api = FakeKnowledgeApi()
    tool = knowledge_tool(KB, client=api.client(), agent_id=AGENT)
    agent = create_react_agent(model(tool.name), [tool])

    async def run() -> Any:
        return await agent.ainvoke({"messages": [HumanMessage(content="verify?")]})

    result = asyncio.run(run())
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Citations:" in tool_message.content
    assert "execution" not in body_of(api.last("POST", "/knowledge/search$"))


def test_bind_langgraph_passes_the_pinned_binding_as_execution_identity() -> None:
    world, local = World.local()
    api = FakeKnowledgeApi()
    tool = knowledge_tool(KB, client=api.client(), agent_id=AGENT)
    graph = create_react_agent(model(tool.name), [tool], checkpointer=InMemorySaver())
    managed = world.bind(graph, client=local)
    result = managed.invoke({"messages": [HumanMessage(content="refund?")]}, thread("kb-1"))
    assert isinstance(result["messages"][-1], AIMessage)
    binding_id = managed.get_state(thread("kb-1")).metadata["agenomic_binding_id"]
    assert isinstance(binding_id, str)
    assert binding_id
    sent = body_of(api.last("POST", "/knowledge/search$"))
    assert sent["execution"] == {"binding_id": binding_id}
    assert sent["knowledge_base"] == KB
