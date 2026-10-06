from __future__ import annotations

import asyncio
import copy
import operator
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import StateUpdate
from langgraph_world import (
    LogState,
    World,
    plan_node,
    thread,
    two_node_graph,
)
from prompt_fakes import AGENT, CHILD, WORKSPACE
from typing_extensions import TypedDict

from agenomic.exceptions import ApiError
from agenomic.integrations import langgraph_binding
from agenomic.integrations.langgraph_binding import (
    RESERVED_KEYS,
    SET_KEY,
    AgenomicUntestedVersionWarning,
    AgentFactory,
    ManagedGraph,
    bind_langgraph,
    prompts_for,
)
from agenomic.prompts import (
    PinnedPromptSet,
    PromptBindingError,
    PromptRefError,
    PromptRenderError,
    thread_key,
)


def digest_of(world: World, name: str) -> str:
    return str(world.engine.get_release(world.releases[name])["prompt_manifest_digest"])


def test_entry_points_inject_pin() -> None:
    world = World.create()
    graph = two_node_graph()
    managed = world.bind(graph)
    assert isinstance(managed, PregelProtocol)
    managed.invoke({"log": []}, thread("invoke"))
    asyncio.run(managed.ainvoke({"log": []}, thread("ainvoke")))
    assert list(managed.stream({"log": []}, thread("stream"), stream_mode="updates"))

    async def consume() -> None:
        async for _ in managed.astream({"log": []}, thread("astream")):
            pass
        async for _ in managed.astream_events({"log": []}, thread("events"), version="v2"):
            pass
        await managed.aupdate_state(thread("aupdate"), {"log": ["m"]}, as_node="supervise")
        await managed.abulk_update_state(
            thread("abulk"), [[StateUpdate({"log": ["m"]}, "supervise")]]
        )

    asyncio.run(consume())
    managed.update_state(thread("update"), {"log": ["m"]}, as_node="supervise")
    managed.bulk_update_state(thread("bulk"), [[StateUpdate({"log": ["m"]}, "supervise")]])
    names = ("invoke", "ainvoke", "stream", "astream", "events", "update", "aupdate")
    for name in (*names, "bulk", "abulk"):
        metadata = managed.get_state(thread(name)).metadata
        assert metadata["agenomic_binding_id"].startswith("bnd_"), name
        assert metadata["agenomic_prompt_manifest_digest"] == digest_of(world, "v1")
        assert metadata["agenomic_release_id"] == world.releases["v1"]
        assert metadata["agenomic_agent_id"] == AGENT
        assert SET_KEY not in metadata
    assert len(world.binding_posts()) == len(names) + 2


@pytest.mark.parametrize("member", ["configurable", "metadata"])
@pytest.mark.parametrize("key", sorted(RESERVED_KEYS))
def test_reserved_keys_refused(key: str, member: str) -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    config: dict[str, Any] = {"configurable": {"thread_id": "forged"}}
    config.setdefault(member, {})[key] = "bnd_forged"
    with pytest.raises(PromptBindingError) as raised:
        managed.invoke({"log": []}, config)
    assert raised.value.code == "agenomic_reserved_key"
    assert raised.value.details["key"] == key
    with pytest.raises(PromptBindingError) as rewrap:
        managed.with_config({member: {key: "x"}})
    assert rewrap.value.code == "agenomic_reserved_key"
    assert world.binding_posts() == []


def test_inner_graph_config_reserved_key_refused_at_bind() -> None:
    world = World.create()
    graph = two_node_graph().with_config({"configurable": {"agenomic_binding_id": "bnd_x"}})
    with pytest.raises(PromptBindingError) as raised:
        world.bind(graph)
    assert raised.value.code == "agenomic_reserved_key"
    graph = two_node_graph().with_config({"metadata": {"agenomic_prompt_refs": "prm_x:1"}})
    with pytest.raises(PromptBindingError):
        world.bind(graph)


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.model_metadata: list[dict[str, Any]] = []
        self.chain_tags: set[str] = set()

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, metadata: Any = None, **kw: Any
    ) -> None:
        self.model_metadata.append(dict(metadata or {}))

    def on_chain_start(
        self, serialized: Any, inputs: Any, *, run_id: UUID, tags: Any = None, **kw: Any
    ) -> None:
        self.chain_tags.update(tags or [])


def test_callbacks_tags_metadata_recursion_limit_preserved() -> None:
    world = World.create()
    seen: list[RunnableConfig] = []

    def call_model(state: LogState, config: RunnableConfig) -> LogState:
        prompts = prompts_for(config)
        seen.append(config)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        text = prompts.render_text("planner.instructions")
        model.invoke([HumanMessage(text)], prompts.config_for("planner.instructions"))
        return {"log": ["called"]}

    builder = StateGraph(LogState)
    builder.add_node("call", call_model)
    builder.add_edge(START, "call")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    recorder = Recorder()
    managed.invoke(
        {"log": []},
        {
            "configurable": {"thread_id": "cb", "user_setting": 3},
            "callbacks": [recorder],
            "tags": ["user-tag"],
            "metadata": {"user_key": "u"},
            "recursion_limit": 7,
            "max_concurrency": 2,
            "run_name": "user-run",
        },
    )
    metadata = recorder.model_metadata[0]
    assert metadata["user_key"] == "u"
    assert metadata["agenomic_binding_id"].startswith("bnd_")
    assert metadata["agenomic_prompt_manifest_digest"] == digest_of(world, "v1")
    assert metadata["agenomic_prompt_slots"] == "planner.instructions"
    assert metadata["agenomic_prompt_refs"] == "prm_plan:1"
    assert metadata["agenomic_rendered_hash"].startswith("sha256:")
    assert SET_KEY not in metadata
    assert "user-tag" in recorder.chain_tags
    node_config = seen[0]
    assert node_config["recursion_limit"] == 7
    assert node_config["configurable"]["user_setting"] == 3
    assert node_config["metadata"]["user_key"] == "u"

    def loop(state: LogState) -> LogState:
        return {"log": ["x"]}

    looping = StateGraph(LogState)
    looping.add_node("a", loop)
    looping.add_node("b", loop)
    looping.add_edge(START, "a")
    looping.add_edge("a", "b")
    looping.add_edge("b", "a")
    managed_loop = world.bind(looping.compile(checkpointer=InMemorySaver()))
    with pytest.raises(GraphRecursionError):
        managed_loop.invoke({"log": []}, {**thread("loop"), "recursion_limit": 5})


def test_batch_abatch_admit_per_input() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    managed.invoke({"log": []}, thread("old"))
    world.promote("v2")
    before = len(world.binding_posts())
    outputs = managed.batch([{"log": []}, {"log": []}], [thread("old"), thread("new")])
    assert outputs[0]["log"][-1] == "sup:SUP v1"
    assert outputs[1]["log"][-1] == "sup:SUP v2"
    assert len(world.binding_posts()) - before == 2
    aoutputs = asyncio.run(
        managed.abatch([{"log": []}, {"log": []}], [thread("old"), thread("newer")])
    )
    assert aoutputs[0]["log"][-1] == "sup:SUP v1"
    assert aoutputs[1]["log"][-1] == "sup:SUP v2"
    assert len(world.binding_posts()) - before == 4


def test_with_config_rewraps() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    tagged = managed.with_config({"tags": ["wc-tag"]}, run_name="renamed")
    assert isinstance(tagged, ManagedGraph)
    first = tagged.invoke({"log": []}, thread("shared"))
    second = managed.invoke({"log": []}, thread("shared"))
    assert first["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    assert second["log"][-1] == "sup:SUP v1"
    ids = {post["thread_key"] for post in world.binding_posts()}
    assert ids == {thread_key(WORKSPACE, "shared")}
    assert tagged.agent_id == managed.agent_id == AGENT
    assert tagged.name == managed.name
    bound = managed.with_config(configurable={"thread_id": "bound-thread"})
    assert bound.invoke({"log": []})["log"] == ["plan:PLAN v1", "sup:SUP v1"]
    assert world.binding_posts()[-1]["thread_key"] == thread_key(WORKSPACE, "bound-thread")


def test_copy_refused() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    with pytest.raises(TypeError):
        managed.copy()


def test_unwrapped_graph_accessor_fails_closed() -> None:
    graph = two_node_graph()
    with pytest.raises(PromptBindingError) as raised:
        graph.invoke({"log": []}, thread("raw"))
    assert raised.value.code == "binding_missing"
    with pytest.raises(PromptBindingError) as empty:
        prompts_for({})
    assert empty.value.code == "binding_missing"


def test_thread_id_required() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    with pytest.raises(PromptBindingError) as raised:
        managed.invoke({"log": []}, {})
    assert raised.value.code == "thread_id_required"
    assert world.binding_posts() == []


@pytest.mark.parametrize("scopes", [[], ["write"], ["admin"], ["read", "write"], None])
def test_privileged_key_refused_without_opt_in(scopes: Any) -> None:
    world = World.create()
    world.server.api_key_scopes = scopes
    with pytest.raises(PromptBindingError) as raised:
        world.bind(two_node_graph())
    assert raised.value.code == "privileged_credential"
    managed = world.bind(two_node_graph(), allow_privileged_credential=True)
    assert managed.invoke({"log": []}, thread("ok"))["log"][-1] == "sup:SUP v1"


def test_read_key_binds_without_opt_in() -> None:
    world = World.create()
    world.server.api_key_scopes = ["read"]
    assert world.bind(two_node_graph()).invoke({"log": []}, thread("ok"))["log"]


def test_no_raw_thread_id_in_binding_request() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    raw = "customer-42@example.com/conversation-9"
    managed.invoke({"log": []}, thread(raw))
    posts = world.binding_posts()
    assert posts[0]["thread_key"] == thread_key(WORKSPACE, raw)
    assert len(posts[0]["thread_key"]) == 78
    assert all(raw.encode() not in request.content for request in world.server.requests)
    assert all(raw not in str(request.url) for request in world.server.requests)
    numeric = world.bind(two_node_graph())
    numeric.invoke({"log": []}, thread(UUID(int=7)))
    assert world.binding_posts()[-1]["thread_key"] == thread_key(WORKSPACE, str(UUID(int=7)))


def captured_set(world: World) -> PinnedPromptSet:
    captured: list[PinnedPromptSet] = []

    def node(state: LogState, config: RunnableConfig) -> LogState:
        captured.append(config["configurable"][SET_KEY])
        return {"log": ["x"]}

    builder = StateGraph(LogState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    world.bind(builder.compile(checkpointer=InMemorySaver())).invoke({"log": []}, thread("cap"))
    return captured[0]


def test_prompt_set_not_picklable() -> None:
    pinned = captured_set(World.create())
    for attempt in (lambda: pickle.dumps(pinned), lambda: pinned.__reduce__()):
        with pytest.raises(PromptBindingError) as raised:
            attempt()
        assert raised.value.code == "prompt_set_not_serializable"
    assert copy.copy(pinned) is pinned
    assert copy.deepcopy({"set": pinned})["set"] is pinned
    assert "PLAN" not in repr(pinned)
    assert repr(pinned).startswith("PinnedPromptSet(binding_id=bnd_")
    with pytest.raises(AttributeError):
        pinned._binding_id = "bnd_other"
    with pytest.raises(TypeError):
        pinned.node_children["x"] = CHILD


def test_prompt_set_unavailable_trigger() -> None:
    world = World.create()
    managed = world.bind(two_node_graph())
    chunks = list(managed.stream({"log": []}, thread("strip"), stream_mode="checkpoints"))
    stripped = chunks[-1]["config"]
    assert stripped["configurable"]["agenomic_binding_id"].startswith("bnd_")
    with pytest.raises(PromptBindingError) as raised:
        prompts_for(stripped)
    assert raised.value.code == "prompt_set_unavailable"
    rebuilt = {"configurable": {"agenomic_binding_id": "bnd_x"}}
    with pytest.raises(PromptBindingError) as metadata_only:
        prompts_for(rebuilt)
    assert metadata_only.value.code == "prompt_set_unavailable"
    factory = AgentFactory(lambda prompts: two_node_graph())
    unbuilt = world.bind(factory)
    with pytest.raises(PromptBindingError) as early:
        unbuilt.get_graph()
    assert early.value.code == "prompt_set_unavailable"


def test_nested_bind_unsupported_trigger() -> None:
    world = World.create()
    child = world.bind(child_graph(), agent_id=CHILD)

    def wrapper(state: LogState, config: RunnableConfig) -> LogState:
        out = child.invoke({"log": []}, {"configurable": {"thread_id": "fresh"}})
        return {"log": out["log"]}

    builder = StateGraph(LogState)
    builder.add_node("wrap", wrapper)
    builder.add_edge(START, "wrap")
    root = world.bind(builder.compile(checkpointer=InMemorySaver()))
    before = len(world.binding_posts())
    with pytest.raises(PromptBindingError) as raised:
        root.invoke({"log": []}, thread("nested"))
    assert raised.value.code == "nested_bind_unsupported"
    assert len(world.binding_posts()) - before == 1


def child_graph(checkpointer: Any = None) -> Any:
    def research(state: LogState, config: RunnableConfig) -> LogState:
        prompts = prompts_for(config)
        return {
            "log": [f"res:{prompts.agent_id == CHILD}:{prompts.render_text('researcher.system')}"]
        }

    builder = StateGraph(LogState)
    builder.add_node("research", research)
    builder.add_edge(START, "research")
    return builder.compile(checkpointer=checkpointer)


class Hijack(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    agenomic_prompt_manifest_digest: str
    agenomic_release_id: str


def test_user_state_cannot_override_binding() -> None:
    world = World.create()
    builder = StateGraph(Hijack)
    builder.add_node("plan", plan_node)
    builder.add_edge(START, "plan")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    out = managed.invoke(
        {
            "log": [],
            "agenomic_prompt_manifest_digest": digest_of(world, "v2"),
            "agenomic_release_id": world.releases["v2"],
        },
        thread("hijack"),
    )
    assert out["log"] == ["plan:PLAN v1"]
    forged = thread("hijack-2", agenomic_release_id=world.releases["v2"])
    with pytest.raises(PromptBindingError):
        managed.invoke({"log": []}, forged)


def test_no_credentials_or_prompt_text_in_checkpoints() -> None:
    world = World.create()
    saver = InMemorySaver()

    def quiet(state: LogState, config: RunnableConfig) -> LogState:
        prompts_for(config).render_text("planner.instructions")
        return {"log": [prompts_for(config).version("planner.instructions").ref.prompt_id]}

    builder = StateGraph(LogState)
    builder.add_node("quiet", quiet)
    builder.add_edge(START, "quiet")
    managed = world.bind(builder.compile(checkpointer=saver))
    managed.invoke({"log": []}, thread("secret"))
    stored = repr(dict(saver.storage)) + repr(dict(saver.writes)) + repr(dict(saver.blobs))
    assert "agm_test" not in stored
    assert "PLAN v1" not in stored
    assert "PinnedPromptSet" not in stored
    assert "Client" not in stored


def test_untested_version_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World.create()
    monkeypatch.setattr(langgraph_binding, "_installed", lambda name: "0.0.1")
    monkeypatch.setitem(langgraph_binding._WARNED, "done", False)
    with pytest.warns(AgenomicUntestedVersionWarning):
        world.bind(two_node_graph())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        world.bind(two_node_graph())


def test_tested_versions_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World.create()
    versions = {"langgraph": "1.2.11", "langchain-core": "1.6.3"}
    monkeypatch.setattr(langgraph_binding, "_installed", versions.get)
    monkeypatch.setitem(langgraph_binding._WARNED, "done", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        world.bind(two_node_graph())


def test_lazy_export_keeps_langgraph_unloaded() -> None:
    code = (
        "import sys; import agenomic.integrations as integrations; "
        "assert 'langgraph' not in sys.modules, 'langgraph loaded'; "
        "assert 'agenomic.integrations.langgraph_binding' not in sys.modules; "
        "bind = integrations.bind_langgraph; "
        "assert 'agenomic.integrations.langgraph_binding' in sys.modules; "
        "assert callable(bind)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    import agenomic.integrations as integrations

    assert integrations.prompts_for is prompts_for
    with pytest.raises(AttributeError):
        _ = integrations.not_a_member


def test_bind_argument_validation(tmp_path: Path) -> None:
    world = World.create()
    client = world.client()
    graph = two_node_graph()
    cases: list[dict[str, Any]] = [
        {"client": client, "agent_id": AGENT},
        {"client": client, "agent_id": AGENT, "channel": "production", "release_id": "x"},
        {"client": client, "agent_id": "agt_support", "channel": "production"},
        {"client": client, "agent_id": AGENT, "channel": "production", "pin_scope": "user"},
        {"client": client, "agent_id": AGENT, "channel": "production", "revalidate": "x"},
        {"agent_id": AGENT, "channel": "production"},
        {"client": client, "agent_id": AGENT, "channel": "production", "offline": True},
        {"agent_id": AGENT, "channel": "production", "offline": True, "bundle": "b.json"},
        {
            "agent_id": AGENT,
            "channel": "production",
            "offline": True,
            "bundle": str(tmp_path / "b.json"),
            "workspace_id": WORKSPACE,
        },
        {"client": client, "agent_id": AGENT, "channel": "production", "bundle": "b.json"},
        {"client": client, "agent_id": AGENT, "channel": "production", "children": {"a b": CHILD}},
        {"client": client, "agent_id": AGENT, "channel": "production", "children": {"a": "x"}},
        {
            "client": client,
            "agent_id": AGENT,
            "channel": "production",
            "child_selectors": {CHILD: {"channel": "a", "release_id": "b"}},
        },
        {"agent_id": AGENT, "binding": {"binding_id": "bnd_x"}},
    ]
    for case in cases:
        with pytest.raises(ValueError):
            bind_langgraph(graph, **case)
    with pytest.raises(TypeError):
        bind_langgraph(object(), client=client, agent_id=AGENT, channel="production")
    managed = world.bind(graph)
    with pytest.raises(ValueError):
        bind_langgraph(managed, client=client, agent_id=AGENT, channel="production")
    with pytest.raises(PromptRefError) as other:
        world.bind(graph, workspace_id="5a5a5a5a-1111-4222-8333-444455556666")
    assert other.value.code == "workspace_mismatch"


def test_child_selectors_reach_the_binding_request() -> None:
    world = World.create()
    selectors = {CHILD: {"channel": "production"}}
    managed = world.bind(two_node_graph(), child_selectors=selectors)
    managed.invoke({"log": []}, thread("sel"))
    assert world.binding_posts()[0]["child_selectors"] == selectors
    local_world, client = World.local()
    with pytest.raises(ApiError) as raised:
        local_world.bind(two_node_graph(), client=client, child_selectors=selectors)
    assert raised.value.code == "cloud_required"


@pytest.mark.filterwarnings("ignore::langgraph.warnings.LangGraphDeprecatedSinceV10")
def test_schema_members_delegate() -> None:
    world = World.create()
    inner = two_node_graph()
    managed = world.bind(inner)
    assert managed.InputType == inner.InputType
    assert managed.OutputType == inner.OutputType
    assert managed.get_input_schema().model_json_schema() == (
        inner.get_input_schema().model_json_schema()
    )
    assert managed.get_output_schema().model_json_schema() == (
        inner.get_output_schema().model_json_schema()
    )
    assert managed.config_specs == inner.config_specs
    assert managed.config_schema().model_json_schema() == inner.config_schema().model_json_schema()
    assert managed.get_name() == inner.get_name()
    assert set(asyncio.run(managed.aget_graph()).nodes) == set(inner.get_graph().nodes)
    assert (managed.pin_scope, managed.workspace_id) == ("thread", WORKSPACE)
    with pytest.raises(AttributeError):
        _ = managed._private_member


def test_genome_version_injected_when_release_has_one() -> None:
    world = World.create()
    genome = "sha256:" + "ab" * 32
    world.engine._state["releases"][world.releases["v1"]]["genome_version"] = genome
    seen: list[dict[str, Any]] = []

    def node(state: LogState, config: RunnableConfig) -> LogState:
        seen.append(dict(config["metadata"]))
        assert prompts_for(config).genome_version == genome
        return {"log": ["x"]}

    builder = StateGraph(LogState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    managed = world.bind(builder.compile(checkpointer=InMemorySaver()))
    managed.invoke({"log": []}, thread("genome"))
    assert seen[0]["agenomic_genome_version"] == genome
    assert managed.get_state(thread("genome")).metadata["agenomic_genome_version"] == genome


def test_accessor_compose_render_and_config_for() -> None:
    world = World.create()
    results: dict[str, Any] = {}

    def node(state: LogState, config: RunnableConfig) -> LogState:
        prompts = prompts_for(config)
        history = [HumanMessage("earlier"), AIMessage("reply")]
        results["compose"] = prompts.compose("assistant.greeting", history=history)
        results["messages"] = prompts.render_messages(
            "assistant.chat", {"customer": "ACME", "messages": history}
        )
        results["single"] = prompts.config_for("assistant.greeting")["metadata"]
        prompts.render_text("planner.instructions")
        results["pair"] = prompts.config_for("assistant.greeting", "planner.instructions")[
            "metadata"
        ]
        results["unrendered"] = prompts_for(config).config_for("supervisor.system")["metadata"]
        results["identity"] = (prompts.binding_id, prompts.release_id, prompts.workspace_id)
        with pytest.raises(ValueError):
            prompts.config_for()
        with pytest.raises(PromptRenderError) as kind:
            prompts.render_text("assistant.chat", {"customer": "ACME"})
        results["kind"] = kind.value.details["reason"]
        return {"log": ["x"]}

    builder = StateGraph(LogState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    world.bind(builder.compile(checkpointer=InMemorySaver())).invoke({"log": []}, thread("acc"))
    assert [message.type for message in results["compose"]] == ["system", "human", "ai"]
    assert results["compose"][0].content == "GREET"
    assert [message.content for message in results["messages"]] == [
        "CHAT v1 for ACME",
        "earlier",
        "reply",
    ]
    assert results["single"]["agenomic_rendered_hash"].startswith("sha256:")
    assert "agenomic_rendered_hash" not in results["pair"]
    assert results["pair"]["agenomic_prompt_refs"] == "prm_greet:1,prm_plan:1"
    assert "agenomic_rendered_hash" not in results["unrendered"]
    assert results["identity"][1:] == (world.releases["v1"], WORKSPACE)
    assert results["kind"] == "kind_mismatch"
