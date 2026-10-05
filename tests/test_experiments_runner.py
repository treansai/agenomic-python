from __future__ import annotations

import asyncio
import json
import pickle
import sys
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from experiment_fakes import (
    BASE,
    TOKEN,
    FakeRunnerServer,
    GraphState,
    agent_case,
    make_runner,
    runtime_digest,
    single_node,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt
from langgraph_world import World
from prompt_fakes import AGENT

import agenomic._transport as transport_module
import agenomic.experiments.tools as tools_module
from agenomic import Client
from agenomic.cli.__main__ import main
from agenomic.exceptions import ApiError
from agenomic.experiments import (
    EnvSecretResolver,
    IsolationViolation,
    SecretResolutionError,
    SecretValues,
    TrialContext,
    classify,
)
from agenomic.experiments.errors import error_code
from agenomic.experiments.isolation import NamespacedStore, jsonable
from agenomic.experiments.runner import (
    CallableEntryPoint,
    ExperimentRunner,
    GraphNodeEntryPoint,
    GraphTarget,
    RunnerEvaluator,
    local_assignment,
)
from agenomic.experiments.secrets import redact_message, replace_secrets
from agenomic.experiments.tools import FIXTURE_MISS_TEXT, logical_call_id
from agenomic.integrations.langgraph_binding import bind_langgraph, prompts_for

SECRET = "s3cr3t-runner-value-0123456789"


@pytest.fixture(autouse=True)
def fast_sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    async def fast(delay: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(transport_module, "_asleep", fast)
    monkeypatch.setattr(transport_module, "_sleep", lambda delay: None)
    monkeypatch.setattr(tools_module, "_asleep", fast)
    monkeypatch.setattr(tools_module, "_sleep", lambda delay: None)
    return


@pytest.fixture
def world() -> World:
    return World.create()


@pytest.fixture
def server(world: World) -> FakeRunnerServer:
    return FakeRunnerServer(world)


async def render_plan(
    ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    text = prompts_for(config).render_text("planner.instructions")
    return {"log": [text], "messages": [AIMessage(content=text)]}


@tool(description="Look up an order.")
def lookup_order(order_id: str) -> str:
    return f"real order {order_id}"


def tool_graph(model_calls: list[list[ToolCall]], *, handle_errors: bool = False) -> Any:
    def factory(ctx: TrialContext) -> Any:
        replies = iter(model_calls)

        def agent(state: dict[str, Any]) -> dict[str, Any]:
            calls = next(replies, [])
            return {"messages": [AIMessage(content="", tool_calls=calls)]}

        builder = StateGraph(GraphState)
        builder.add_node("agent", agent)
        builder.add_node(
            "tools", ToolNode(ctx.wrap_tools([lookup_order]), handle_tool_errors=handle_errors)
        )
        builder.add_edge(START, "agent")
        builder.add_edge("agent", "tools")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    return factory


def call(order_id: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(name="lookup_order", args={"order_id": order_id}, id=call_id)


def serve(runner: ExperimentRunner, **options: Any) -> int:
    return runner.serve(idle_timeout=0, **options)


def test_hello_declares_agents_entry_points_and_secret_names(
    world: World, server: FakeRunnerServer
) -> None:
    evaluator = RunnerEvaluator(lambda case, output: True, version=3)
    runner = make_runner(
        server,
        single_node(render_plan),
        entry_points={"planner": GraphNodeEntryPoint("plan", seed_as_node="__start__")},
        evaluators={"refund_ok": evaluator},
        live_tools=True,
        runner_options={"secrets": EnvSecretResolver(allow=["OPENAI_API_KEY"], environ={})},
    )
    assert serve(runner) == 0
    hello = server.hellos[0]
    agent = hello["agents"][0]
    assert agent["agent_id"] == AGENT
    assert agent["runtime_digests"] == [runtime_digest(world)]
    assert agent["levels"] == ["agent", "node"]
    assert agent["tool_modes"] == ["none", "mock", "recorded", "live"]
    assert agent["entry_points"] == [
        {"name": "planner", "kind": "graph_node", "node_path": "plan", "seed_as_node": "__start__"}
    ]
    assert agent["custom_evaluators"][0]["code_digest"] == evaluator.digest()
    assert hello["secret_ref_names"] == ["env:OPENAI_API_KEY"]
    assert TOKEN not in repr(runner)
    assert TOKEN.encode() not in b"".join(server.raw_bodies())


def test_concurrent_arms_isolated_in_runner(server: FakeRunnerServer) -> None:
    active: list[int] = [0]
    peak: list[int] = [0]
    savers: list[Any] = []

    async def node(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        text = prompts_for(config).render_text("planner.instructions")
        namespace = (*ctx.store_namespace, "memories")
        await ctx.store.aput(namespace, "seen", {"text": text})
        await asyncio.sleep(0.05)
        items = await ctx.store.asearch(namespace)
        savers.append(ctx.checkpointer)
        active[0] -= 1
        return {"log": [f"{text}|{len(items)}|{config['configurable']['thread_id']}"]}

    trials = {
        name: server.add_trial(name, agent_case(), arm_key=f"arm_{name}", experiment_id="exp_iso")
        for name in ("v1", "v2")
    }
    runner = make_runner(
        server,
        single_node(node),
        output_adapter=lambda values: values["log"],
        runner_options={"max_concurrency": 2},
    )
    assert serve(runner) == 2
    assert peak[0] == 2
    assert len({id(saver) for saver in savers}) == 2
    for name, trial_id in trials.items():
        result = server.trials[trial_id].result
        assert result is not None
        assert result["outcome"] == "evaluated"
        text, count, thread = result["output"]["final"][0].split("|")
        assert text == f"PLAN {name}"
        assert count == "1"
        assert thread == f"exp:exp_iso:{trial_id}:a1"
        assert result["isolation"]["fresh_thread"] is True
        assert result["isolation"]["checkpointer"] == "per_trial_in_memory"
        assert (
            result["runner_view_digest"] == server.trials[trial_id].assignment["runner_view_digest"]
        )
        assert server.trials[trial_id].accepts == 1


def test_isolation_violation_detected(world: World, server: FakeRunnerServer) -> None:
    def own_saver(ctx: TrialContext) -> Any:
        builder = StateGraph(GraphState)
        builder.add_node("plan", lambda state: {"log": ["x"]})
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=InMemorySaver())

    first = server.add_trial("v1", agent_case())
    assert serve(make_runner(server, own_saver)) == 1
    assert server.trials[first].failures[0]["error_code"] == "isolation_violation"
    assert server.trials[first].failures[0]["error_class"] == "runner_configuration"

    async def escape(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        await ctx.store.aput(("other_trial",), "k", {"v": 1})
        return {}

    shared = InMemoryStore()
    second = server.add_trial("v1", agent_case("case-2"))
    assert serve(make_runner(server, single_node(escape), store=shared)) == 1
    failure = server.trials[second].failures[0]
    assert (failure["error_class"], failure["error_code"]) == (
        "runner_configuration",
        "isolation_violation",
    )
    assert shared.search(("other_trial",)) == []


def test_secrets_never_in_reports(world: World, server: FakeRunnerServer) -> None:
    resolver = EnvSecretResolver(allow=["SERVICE_KEY"], environ={"SERVICE_KEY": SECRET})

    async def leaky(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        value = ctx.secrets["env:SERVICE_KEY"]
        if ctx.case.case_id == "raise":
            raise RuntimeError(f"provider rejected key {value}")
        tools = {item.name: item for item in ctx.wrap_tools([lookup_order])}
        answer = await tools["lookup_order"].ainvoke({"order_id": value})
        return {"log": [f"key={value}", str(answer)]}

    ok = server.add_trial(
        "v1", agent_case("ok"), secret_refs=["env:SERVICE_KEY"], tools={"mode": "mock"}
    )
    failed = server.add_trial("v1", agent_case("raise"), secret_refs=["env:SERVICE_KEY"])
    runner = make_runner(
        server,
        single_node(leaky),
        output_adapter=lambda values: values["log"],
        runner_options={"secrets": resolver},
    )
    assert serve(runner) == 2
    for body in server.raw_bodies():
        assert SECRET.encode() not in body
        assert TOKEN.encode() not in body
    output = server.trials[ok].result["output"]["final"]
    assert output[0] == "key=[REDACTED]"
    error = server.trials[failed].result["error"]
    assert error["message"] == "provider rejected key [REDACTED]"
    assert server.trials[failed].result["outcome"] == "agent_error"
    assert "values=hidden" in repr(SecretValues({"env:SERVICE_KEY": SECRET}))
    assert SECRET not in repr(SecretValues({"env:SERVICE_KEY": SECRET}))
    with pytest.raises(TypeError):
        pickle.dumps(SecretValues({"env:SERVICE_KEY": SECRET}))


def test_unresolved_secret_is_runner_configuration(server: FakeRunnerServer) -> None:
    trial = server.add_trial("v1", agent_case(), secret_refs=["env:MISSING"])
    assert serve(make_runner(server, single_node(render_plan))) == 1
    assert server.trials[trial].failures[0]["error_code"] == "secret_unresolved"
    resolver = EnvSecretResolver(allow=["MISSING"], environ={})
    with pytest.raises(SecretResolutionError):
        resolver.resolve("env:MISSING")
    with pytest.raises(SecretResolutionError):
        resolver.resolve("vault:MISSING")
    with pytest.raises(ValueError):
        EnvSecretResolver(allow=["bad name"])


def test_tool_calls_mock_recorded_and_fixture_policies(server: FakeRunnerServer) -> None:
    mock = server.add_trial("v1", agent_case("mock"), tools={"mode": "mock"})
    hit = server.add_trial("v1", agent_case("hit"), tools={"mode": "recorded"})
    miss = server.add_trial("v1", agent_case("miss"), tools={"mode": "recorded"})
    soft = server.add_trial(
        "v1", agent_case("soft"), tools={"mode": "recorded", "on_fixture_miss": "tool_error"}
    )
    server.fixtures[("lookup_order", '{"order_id":"1234"}')] = {"status": "shipped"}
    plans = {
        "mock": [call("77")],
        "hit": [call("1234")],
        "miss": [call("9999")],
        "soft": [call("9999")],
    }

    def factory(ctx: TrialContext) -> Any:
        return tool_graph([plans[ctx.case.case_id]], handle_errors=ctx.case.case_id == "miss")(ctx)

    runner = make_runner(
        server, factory, output_adapter=lambda values: values["messages"][-1].content
    )
    assert serve(runner) == 4
    assert json.loads(server.trials[mock].result["output"]["final"]) == {
        "mock": "lookup_order",
        "arguments": {"order_id": "77"},
    }
    assert json.loads(server.trials[hit].result["output"]["final"]) == {"status": "shipped"}
    missed = server.trials[miss].result
    assert missed["outcome"] == "recorded_fixture_miss"
    assert missed["error"]["tool"] == "lookup_order"
    assert server.trials[soft].result["outcome"] == "evaluated"
    assert server.trials[soft].result["output"]["final"] == FIXTURE_MISS_TEXT
    calls = [json.loads(r.content) for r in server.requests if r.url.path.endswith("/tool-calls")]
    assert {item["logical_call_id"] for item in calls} == {"call_1"}
    assert all(item["attempt"] == 1 for item in calls)


def test_tool_call_in_progress_backoff_and_exhaustion(server: FakeRunnerServer) -> None:
    waited = server.add_trial("v1", agent_case("waited"), tools={"mode": "mock"})
    server.in_progress = 2
    runner = make_runner(server, tool_graph([[call("1")]]))
    assert serve(runner) == 1
    assert server.trials[waited].result["outcome"] == "evaluated"
    exhausted = server.add_trial("v1", agent_case("exhausted"), tools={"mode": "mock"})
    server.in_progress = 10
    assert serve(make_runner(server, tool_graph([[call("2")]]))) == 1
    failure = server.trials[exhausted].failures[0]
    assert (failure["error_class"], failure["error_code"]) == (
        "infrastructure",
        "tool_call_in_progress",
    )


def test_tool_call_id_reused_fails_runner_configuration(server: FakeRunnerServer) -> None:
    trial = server.add_trial("v1", agent_case(), tools={"mode": "mock"})
    runner = make_runner(server, tool_graph([[call("1", "call_x"), call("2", "call_x")]]))
    assert serve(runner) == 1
    failure = server.trials[trial].failures[0]
    assert (failure["error_class"], failure["error_code"]) == (
        "runner_configuration",
        "tool_call_id_reused",
    )


def test_tool_mode_none_refuses_tool_calls(server: FakeRunnerServer) -> None:
    trial = server.add_trial("v1", agent_case())
    assert serve(make_runner(server, tool_graph([[call("1")]], handle_errors=True))) == 1
    assert server.trials[trial].failures[0]["error_code"] == "tool_mode_none"
    assert not any(r.url.path.endswith("/tool-calls") for r in server.requests)


def test_logical_call_id_is_model_id_or_stable_hash() -> None:
    config = {"configurable": {"checkpoint_ns": "tools:abc", "__pregel_task_id": "abc"}}
    assert logical_call_id("call_ok-1", config, "t", {}) == "call_ok-1"
    hashed = logical_call_id("bad id with spaces", config, "t", {"a": 1.5, "b": [1]})
    assert hashed.startswith("call_")
    assert len(hashed) == 21
    assert hashed == logical_call_id(None, config, "t", {"b": [1], "a": 1.5})
    assert hashed != logical_call_id(None, config, "t", {"a": 2})
    assert hashed != logical_call_id(
        None, {"configurable": {"checkpoint_ns": "tools:def"}}, "t", {"a": 1.5, "b": [1]}
    )


def test_live_tool_runs_locally_once_and_reports(server: FakeRunnerServer) -> None:
    executed: list[str] = []

    @tool(description="Issue a refund.")
    def refund(order_id: str) -> str:
        executed.append(order_id)
        return f"refunded {order_id}"

    def factory(ctx: TrialContext) -> Any:
        async def node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            proxied = ctx.wrap_tools([refund])[0]
            first = await proxied.ainvoke(
                tool_call(name="refund", args={"order_id": "5"}, id="call_r")
            )
            again = await proxied.ainvoke(
                tool_call(name="refund", args={"order_id": "5"}, id="call_r")
            )
            return {"log": [first.content, again.content]}

        builder = StateGraph(GraphState)
        builder.add_node("plan", node)
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    trial = server.add_trial("v1", agent_case(), tools={"mode": "live"})
    server.drop_tool_call_responses = 1
    server.drop_report_responses = 1
    runner = make_runner(server, factory, live_tools=True, output_adapter=lambda v: v["log"])
    assert serve(runner) == 1
    assert executed == ["5"]
    assert server.trials[trial].result["output"]["final"] == ["refunded 5", "refunded 5"]
    reports = [json.loads(r.content) for r in server.requests if r.url.path.endswith("/report")]
    assert len(reports) == 2
    assert reports[0] == reports[1]
    assert reports[0]["value"] == "refunded 5"
    assert reports[0]["is_error"] is False


def test_budget_stop_reports_budget_stopped(server: FakeRunnerServer) -> None:
    async def two_calls(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        model = GenericFakeChatModel(messages=iter([AIMessage("one"), AIMessage("two")]))
        prompts = prompts_for(config)
        text = prompts.render_text("planner.instructions")
        first = await model.ainvoke(
            [SystemMessage(text)], prompts.config_for("planner.instructions")
        )
        second = await model.ainvoke([SystemMessage(text)], config)
        return {"log": [str(first.content), str(second.content)]}

    stopped = server.add_trial("v1", agent_case("stop"), limits={"max_model_calls_per_trial": 1})
    ok = server.add_trial("v2", agent_case("ok"))
    assert serve(make_runner(server, single_node(two_calls))) == 2
    result = server.trials[stopped].result
    assert result["outcome"] == "budget_stopped"
    assert result["error"] == {"code": "trial_budget_exceeded", "limit": "model_calls"}
    calls = server.trials[ok].result["model_calls"]
    assert len(calls) == 2
    assert calls[0]["slot_path"] == "planner.instructions"
    assert calls[0]["prompt_ref"] == "prm_plan:2"
    assert calls[0]["usage_source"] == "not_reported"
    assert calls[0]["input_tokens"] is None
    assert calls[1]["prompt_ref"] is None


def test_prompt_outside_manifest_refused_before_report(server: FakeRunnerServer) -> None:
    async def foreign(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        model = GenericFakeChatModel(messages=iter([AIMessage("x")]))
        prompts = prompts_for(config)
        borrowed = dict(prompts.config_for("planner.instructions"))
        borrowed["metadata"] = {
            **borrowed["metadata"],
            "agenomic_prompt_refs": "prm_plan:1",
        }
        await model.ainvoke("hi", borrowed)
        return {}

    trial = server.add_trial("v2", agent_case())
    assert serve(make_runner(server, single_node(foreign))) == 1
    failure = server.trials[trial].failures[0]
    assert (failure["error_class"], failure["error_code"]) == (
        "runner_configuration",
        "prompt_outside_manifest",
    )
    assert server.trials[trial].result is None


def test_agent_and_infrastructure_errors_are_separated(server: FakeRunnerServer) -> None:
    async def failing(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        if ctx.case.case_id == "agent":
            raise ValueError("the agent gave up")
        if ctx.case.case_id == "infra":
            raise httpx.ConnectError("provider unreachable")
        raise KeyError("hooked")

    agent = server.add_trial("v1", agent_case("agent"))
    infra = server.add_trial("v1", agent_case("infra"))
    hooked = server.add_trial("v1", agent_case("hooked"))
    runner = make_runner(
        server,
        single_node(failing),
        runner_options={
            "classify_error": lambda exc: "infrastructure" if isinstance(exc, KeyError) else None
        },
    )
    assert serve(runner) == 3
    result = server.trials[agent].result
    assert result["outcome"] == "agent_error"
    assert result["error"]["code"] == "agent_exception"
    assert result["error"]["type"] == "ValueError"
    assert result["error_class_source"] == "default_rules/v1"
    infra_failure = server.trials[infra].failures[0]
    assert (infra_failure["error_class"], infra_failure["error_code"]) == (
        "infrastructure",
        "connection_error",
    )
    assert server.trials[hooked].failures[0]["error_class"] == "infrastructure"


def test_error_classification_rules() -> None:
    class RateLimitError(Exception):
        status_code = 429

    class BadRequestError(Exception):
        status_code = 400

    assert classify(RateLimitError()) == ("infrastructure", "default_rules/v1")
    assert error_code(RateLimitError()) == "provider_rate_limited"
    assert classify(BadRequestError())[0] == "agent"
    assert classify(IsolationViolation("x"))[0] == "runner_configuration"
    assert classify(ApiError("registry_unavailable", 0, "down"))[0] == "infrastructure"
    assert error_code(ApiError("registry_unavailable", 0, "down")) == "agenomic_unavailable"
    assert classify(ApiError("prompt_render_error", 0, "missing"))[0] == "agent"
    assert error_code(ApiError("prompt_render_error", 0, "missing")) == "prompt_render_error"
    assert error_code(httpx.ReadTimeout("slow")) == "provider_timeout"
    assert classify(ApiError("server", 503, "x"))[0] == "infrastructure"
    assert error_code(ApiError("server", 503, "x")) == "provider_unavailable"
    with pytest.raises(ValueError):
        classify(ValueError(), lambda exc: "bogus")


def test_cancel_and_stop_release_the_trial(server: FakeRunnerServer) -> None:
    async def slow(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {}

    cancelled = server.add_trial("v1", agent_case("cancel"))
    stopped = server.add_trial("v1", agent_case("stop"))

    def flag(trial: Any) -> None:
        if trial.view["case"]["case_id"] == "cancel":
            trial.cancel_requested = True
        else:
            trial.stop_requested = True

    server.on_heartbeat = flag
    assert serve(make_runner(server, single_node(slow))) == 2
    assert server.trials[cancelled].releases == [
        {"lease_token": server.trials[cancelled].lease_token, "reason": "cancelled"}
    ]
    assert server.trials[stopped].releases[0]["reason"] == "stopped"
    assert server.trials[cancelled].result is None


def test_stale_lease_drops_the_trial_without_report(server: FakeRunnerServer) -> None:
    async def slow(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {}

    trial = server.add_trial("v1", agent_case())

    def expire(item: Any) -> None:
        item.status = "queued"

    server.on_heartbeat = expire
    assert serve(make_runner(server, single_node(slow))) == 1
    posted = [r.url.path.rsplit("/", 1)[-1] for r in server.requests]
    assert "result" not in posted
    assert "failure" not in posted
    assert "release" not in posted
    assert server.trials[trial].result is None


def test_assignment_checks_before_execution(world: World, server: FakeRunnerServer) -> None:
    tampered = server.add_trial("v1", agent_case("tampered"))
    view = server.trials[tampered].assignment["view"]
    ref = next(iter(view["prompts"]["prompts"]))
    view["prompts"]["prompts"][ref]["content"]["body"] = "INJECTED"
    other_runtime = server.add_trial("v1", agent_case("runtime"))
    server.trials[other_runtime].assignment["view"]["arm"]["runtime_digest"] = "blake3:" + "0" * 64
    other_thread = server.add_trial("v1", agent_case("thread"))
    server.trials[other_thread].assignment["view"]["binding"]["thread_key"] = (
        "thread:sha256:" + "0" * 64
    )
    other_manifest = server.add_trial("v1", agent_case("manifest"))
    server.trials[other_manifest].assignment["view"]["arm"]["prompt_manifest_digest"] = (
        "sha256:" + "1" * 64
    )
    built: list[str] = []

    def factory(ctx: TrialContext) -> Any:
        built.append(ctx.case.case_id)
        return single_node(render_plan)(ctx)

    assert serve(make_runner(server, factory)) == 4
    codes = {
        server.trials[trial].view["case"]["case_id"]: server.trials[trial].failures[0]["error_code"]
        for trial in (tampered, other_runtime, other_thread, other_manifest)
    }
    assert codes == {
        "tampered": "artifact_digest_mismatch",
        "runtime": "runtime_digest_mismatch",
        "thread": "binding_mismatch",
        "manifest": "artifact_digest_mismatch",
    }
    assert built == []


def test_node_entry_point_runs_only_the_node(world: World, server: FakeRunnerServer) -> None:
    def factory(ctx: TrialContext) -> Any:
        def intake(state: dict[str, Any]) -> dict[str, Any]:
            return {"log": ["intake ran"]}

        def plan(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            return {"log": [prompts_for(config).render_text("planner.instructions")]}

        def respond(state: dict[str, Any]) -> dict[str, Any]:
            return {"log": ["respond ran"]}

        builder = StateGraph(GraphState)
        for name, node in (("intake", intake), ("plan", plan), ("respond", respond)):
            builder.add_node(name, node)
        builder.add_edge(START, "intake")
        builder.add_edge("intake", "plan")
        builder.add_edge("plan", "respond")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    case = {
        "case_id": "node-1",
        "kind": "node_state",
        "input": {"context": {"locale": "en"}},
        "expected": None,
        "initial_state": {"log": ["seeded"]},
        "turns": [],
        "tags": [],
    }
    good = server.add_trial(
        "v2",
        case,
        level="node",
        entry_point={
            "name": "planner",
            "kind": "graph_node",
            "node_path": "plan",
            "seed_as_node": "intake",
            "slot_paths": [],
        },
    )
    wrong = server.add_trial(
        "v2",
        dict(case, case_id="node-2"),
        level="node",
        entry_point={
            "name": "start",
            "kind": "graph_node",
            "node_path": "plan",
            "seed_as_node": "__start__",
            "slot_paths": [],
        },
    )
    unknown = server.add_trial(
        "v2",
        dict(case, case_id="node-3"),
        level="node",
        entry_point={
            "name": "nope",
            "kind": "graph_node",
            "node_path": "plan",
            "seed_as_node": "intake",
            "slot_paths": [],
        },
    )
    runner = make_runner(
        server,
        factory,
        entry_points={
            "planner": GraphNodeEntryPoint("plan", seed_as_node="intake"),
            "start": GraphNodeEntryPoint("plan", seed_as_node="__start__"),
        },
    )
    assert serve(runner) == 3
    result = server.trials[good].result
    assert result["outcome"] == "evaluated"
    assert result["output"] == {
        "state_update": {"log": ["seeded", "PLAN v2"]},
        "extra_nodes_executed": [],
    }
    assert result["isolation"]["fork_fidelity"] == "none"
    failure = server.trials[wrong].failures[0]
    assert failure["error_code"] == "entry_point_not_next"
    assert server.trials[unknown].failures[0]["error_code"] == "entry_point_unavailable"


def test_callable_entry_point(server: FakeRunnerServer) -> None:
    def entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        text = prompts_for(config).render_text("planner.instructions")
        return {"plan": f"{state['goal']}:{text}"}

    case = {
        "case_id": "callable-1",
        "kind": "node_state",
        "input": {},
        "expected": None,
        "initial_state": {"goal": "refund"},
        "turns": [],
        "tags": [],
    }
    trial = server.add_trial(
        "v1",
        case,
        level="node",
        entry_point={
            "name": "plan_fn",
            "kind": "callable",
            "node_path": "plan_fn",
            "seed_as_node": None,
            "slot_paths": [],
        },
    )
    built: list[str] = []
    runner = make_runner(
        server,
        lambda ctx: built.append("graph"),
        entry_points={"plan_fn": CallableEntryPoint(entry)},
    )
    assert runner.hello_document()["agents"][0]["entry_points"][0]["kind"] == "callable"
    assert serve(runner) == 1
    assert server.trials[trial].result["output"]["state_update"] == {"plan": "refund:PLAN v1"}
    assert built == []


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="langgraph interrupt() in asyncio tasks needs Python 3.11 contextvar propagation",
)
def test_multi_turn_case_reuses_its_thread(server: FakeRunnerServer) -> None:
    def factory(ctx: TrialContext) -> Any:
        def ask(state: dict[str, Any]) -> dict[str, Any]:
            answer = interrupt("approve?")
            return {"log": [f"approved={answer}"]}

        builder = StateGraph(GraphState)
        builder.add_node("ask", ask)
        builder.add_edge(START, "ask")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    trial = server.add_trial("v1", agent_case(turns=[{"resume": True}]))
    assert serve(make_runner(server, factory, output_adapter=lambda v: v["log"])) == 1
    result = server.trials[trial].result
    assert [turn["interrupted"] for turn in result["turns"]] == [True, False]
    assert result["output"]["final"] == ["approved=True"]


def test_runner_evaluators_custom_and_judge(server: FakeRunnerServer, world: World) -> None:
    judge_prompt = {
        "prompt_id": "prm_judge",
        "version": 1,
        "prompt_kind": "text",
        "content_digest": None,
        "content": {
            "schema": "agenomic.prompt_content/v1",
            "template_format": "agenomic-fstring/v1",
            "renderer_version": "1",
            "kind": "text",
            "body": "Rate {output}",
            "variables": {"output": {"type": "json", "required": True}},
            "partials": {},
            "output_contract": None,
            "fragments": {},
        },
    }
    from agenomic.prompts.digest import prompt_digest

    judge_prompt["content_digest"] = prompt_digest(judge_prompt["content"])
    evaluator = RunnerEvaluator(lambda case, output: output["final"] == ["PLAN v1"], version=2)
    evaluators = [
        {
            "evaluator_id": "custom",
            "kind": "runner_custom",
            "version": 2,
            "config": {
                "name": "plan_ok",
                "version": 2,
                "code_digest": evaluator.digest(),
                "value_kind": "binary",
            },
        },
        {
            "evaluator_id": "judge_help",
            "kind": "model_judge",
            "version": 4,
            "config": {
                "model": {"provider": "fake", "model": "judge-1"},
                "rubric": {"scale": {"min": 1, "max": 5}},
            },
            "prompt": judge_prompt,
        },
        {
            "evaluator_id": "judge_off",
            "kind": "model_judge",
            "version": 1,
            "config": {},
            "prompt": None,
        },
    ]
    trial = server.add_trial("v1", agent_case(), runner_evaluators=evaluators)
    seen: list[Any] = []

    def judge_model(settings: Any) -> Any:
        seen.append(dict(settings))
        return GenericFakeChatModel(messages=iter([AIMessage('{"score": 4}')]))

    runner = make_runner(
        server,
        single_node(render_plan),
        evaluators={"plan_ok": evaluator},
        output_adapter=lambda values: values["log"],
        runner_options={"judge_model": judge_model},
    )
    assert serve(runner) == 1
    result = server.trials[trial].result
    assert result["runner_metrics"] == {
        "plan_ok": {"value": 1, "evaluator_ref": "runner_custom:plan_ok@2"}
    }
    judged, skipped = result["judge_scores"]
    assert judged["score"] == 4
    assert judged["parsed"] is True
    assert judged["judge_model"] == "judge-1"
    assert skipped["parsed"] is False
    assert skipped["score"] is None
    assert seen == [{"provider": "fake", "model": "judge-1"}]
    mismatch = server.add_trial(
        "v1",
        agent_case("bad"),
        runner_evaluators=[
            dict(
                evaluators[0],
                config=dict(evaluators[0]["config"], code_digest="sha256:" + "0" * 64),
            )
        ],
    )
    assert serve(runner) == 1
    assert server.trials[mismatch].failures[0]["error_code"] == "evaluator_unavailable"


def test_result_refusals_become_failures(server: FakeRunnerServer) -> None:
    secret = server.add_trial("v1", agent_case("secret"))
    server.refuse_result = ("experiment_result_secret_detected", 400, {})
    assert serve(make_runner(server, single_node(render_plan))) == 1
    assert server.trials[secret].failures[0]["error_code"] == "secret_in_report"
    invalid = server.add_trial("v1", agent_case("invalid"))
    server.refuse_result = ("experiment_result_invalid", 400, {"reason": "prompt_outside_manifest"})
    assert serve(make_runner(server, single_node(render_plan))) == 1
    assert server.trials[invalid].failures[0]["error_code"] == "prompt_outside_manifest"


def test_hello_required_and_stop_release_unstarted_claim(server: FakeRunnerServer) -> None:
    server.add_trial("v1", agent_case())
    runner = make_runner(server, single_node(render_plan))
    original = runner._hello
    calls: list[int] = []

    async def counting(http: Any) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            return {}
        return await original(http)

    runner._hello = counting
    server.on_claim = lambda trial: runner.stop()
    assert serve(runner) == 0
    trial = next(iter(server.trials.values()))
    assert trial.releases[0]["reason"] == "shutdown"
    assert len(calls) == 2


def test_runner_configuration_validation(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    target = GraphTarget(factory=lambda ctx: None, runtime_digest=runtime_digest(world))
    with pytest.raises(ValueError):
        ExperimentRunner(targets={})
    with pytest.raises(ValueError):
        ExperimentRunner(targets={"agent": target})
    with pytest.raises(TypeError):
        ExperimentRunner(targets={AGENT: object()})
    with pytest.raises(ValueError):
        ExperimentRunner(targets={AGENT: target}, token="agm_wrong")
    with pytest.raises(ValueError):
        ExperimentRunner(targets={AGENT: target}, max_concurrency=0)
    with pytest.raises(ValueError):
        ExperimentRunner(targets={AGENT: target}, workspace_id="WS")
    monkeypatch.delenv("AGENOMIC_RUNNER_TOKEN", raising=False)
    monkeypatch.delenv("AGENOMIC_ENDPOINT", raising=False)
    with pytest.raises(ValueError):
        ExperimentRunner(targets={AGENT: target}, base_url=BASE).serve()
    with pytest.raises(ValueError):
        ExperimentRunner(targets={AGENT: target}, token=TOKEN).serve()


def test_workspace_pin_refuses_other_workspace(world: World, server: FakeRunnerServer) -> None:
    trial = server.add_trial("v1", agent_case())
    runner = make_runner(
        server,
        single_node(render_plan),
        runner_options={"workspace_id": "5a5a5a5a-1111-4222-8333-444455556666"},
    )
    assert serve(runner) == 1
    assert server.trials[trial].failures[0]["error_code"] == "workspace_mismatch"


def test_run_trial_offline_harness(world: World) -> None:
    runner = ExperimentRunner(
        targets={
            AGENT: GraphTarget(
                factory=single_node(render_plan), runtime_digest=runtime_digest(world)
            )
        }
    )
    assignment = local_assignment(
        world.engine, agent_id=AGENT, release_id=world.releases["v2"], case=agent_case()
    )
    run = runner.run_trial(assignment)
    assert run.kind == "result"
    assert run.result is not None
    assert run.result["output"]["final"] == "PLAN v2"
    assert run.result["runtime"]["runtime_digest_source"] == "runner_declared"
    tools_assignment = local_assignment(
        world.engine,
        agent_id=AGENT,
        release_id=world.releases["v1"],
        case=agent_case(),
        tools={"mode": "mock"},
    )
    tool_runner = ExperimentRunner(
        targets={
            AGENT: GraphTarget(
                factory=tool_graph([[call("1")]]), runtime_digest=runtime_digest(world)
            )
        }
    )
    failed = tool_runner.run_trial(tools_assignment)
    assert (failed.kind, failed.error_code) == ("failure", "proxy_unavailable")


def test_namespaced_store_and_helpers() -> None:
    inner = InMemoryStore()
    store = NamespacedStore(inner, ("agenomic_exp", "e", "t", "a1"))
    store.put(("agenomic_exp", "e", "t", "a1", "m"), "k", {"v": 1})
    assert store.get(("agenomic_exp", "e", "t", "a1", "m"), "k").value == {"v": 1}
    assert store.list_namespaces(prefix=("agenomic_exp", "e", "t", "a1")) == [
        ("agenomic_exp", "e", "t", "a1", "m")
    ]
    for attempt in (
        lambda: store.get(("other",), "k"),
        lambda: store.search(("agenomic_exp",)),
        lambda: store.list_namespaces(),
    ):
        with pytest.raises(IsolationViolation):
            attempt()
    with pytest.raises(ValueError):
        NamespacedStore(store, ("x",))
    with pytest.raises(ValueError):
        NamespacedStore(inner, ())
    assert jsonable({"m": AIMessage("x"), "t": (1, 2)}) == {
        "m": {"role": "assistant", "content": "x"},
        "t": [1, 2],
    }
    assert replace_secrets({"k": ["a SECRET b"]}, ["SECRET"]) == {"k": ["a [REDACTED] b"]}
    assert redact_message("x" * 5000, []).encode() == b"x" * 2048
    assert "sk-" not in redact_message("key sk-" + "a" * 30, [])


def test_cli_experiment_serve(server: FakeRunnerServer, monkeypatch: pytest.MonkeyPatch) -> None:
    trial = server.add_trial("v1", agent_case())
    module = types.ModuleType("agenomic_test_runner_target")
    target = GraphTarget(
        factory=single_node(render_plan), runtime_digest=runtime_digest(server.world)
    )
    module.runner = ExperimentRunner(targets={AGENT: target}, transport=server.transport())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("AGENOMIC_RUNNER_TOKEN", TOKEN)
    monkeypatch.setenv("AGENOMIC_ENDPOINT", BASE)
    assert (
        main(
            [
                "experiment",
                "serve",
                "--target",
                f"{module.__name__}:runner",
                "--idle-timeout",
                "0",
                "--max-concurrency",
                "2",
            ]
        )
        == 0
    )
    assert server.trials[trial].accepts == 1
    assert main(["experiment", "serve", "--target", "missing_module_x:runner"]) == 2
    assert (
        main(
            [
                "experiment",
                "serve",
                "--target",
                f"{module.__name__}:runner",
                "--max-concurrency",
                "0",
            ]
        )
        == 2
    )
    module.other = 42
    assert main(["experiment", "serve", "--target", f"{module.__name__}:other"]) == 2
    monkeypatch.delenv("AGENOMIC_ENDPOINT")
    module.bare = ExperimentRunner(targets={AGENT: target})
    assert main(["experiment", "serve", "--target", f"{module.__name__}:bare"]) == 2


def test_sync_tool_paths_mock_and_live(server: FakeRunnerServer) -> None:
    executed: list[str] = []

    @tool(description="Issue a refund.")
    def refund(order_id: str) -> str:
        executed.append(order_id)
        return f"refunded {order_id}"

    def factory(ctx: TrialContext) -> Any:
        def node(state: dict[str, Any]) -> dict[str, Any]:
            lookup, issue = ctx.wrap_tools([lookup_order, refund])
            picked = issue if ctx.tool_mode == "live" else lookup
            first = picked.invoke(tool_call(name=picked.name, args={"order_id": "9"}, id="call_s"))
            return {"log": [str(first.content)]}

        builder = StateGraph(GraphState)
        builder.add_node("plan", node)
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    mock = server.add_trial("v1", agent_case("mock"), tools={"mode": "mock"})
    live = server.add_trial("v1", agent_case("live"), tools={"mode": "live"})
    server.in_progress = 1
    server.drop_report_responses = 1
    runner = make_runner(server, factory, live_tools=True, output_adapter=lambda v: v["log"])
    assert serve(runner) == 2
    assert json.loads(server.trials[mock].result["output"]["final"][0])["mock"] == "lookup_order"
    assert server.trials[live].result["output"]["final"] == ["refunded 9"]
    assert executed == ["9"]


def test_live_tool_error_is_reported_and_reraised(server: FakeRunnerServer) -> None:
    @tool(description="Always fails.")
    def explode(order_id: str) -> str:
        raise RuntimeError(f"cannot refund {order_id} with {SECRET}")

    resolver = EnvSecretResolver(allow=["SERVICE_KEY"], environ={"SERVICE_KEY": SECRET})

    async def node(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        proxied = ctx.wrap_tools([explode])[0]
        await proxied.ainvoke({"order_id": "3"})
        return {}

    trial = server.add_trial(
        "v1", agent_case(), tools={"mode": "live"}, secret_refs=["env:SERVICE_KEY"]
    )
    runner = make_runner(
        server, single_node(node), live_tools=True, runner_options={"secrets": resolver}
    )
    assert serve(runner) == 1
    report = next(json.loads(r.content) for r in server.requests if r.url.path.endswith("/report"))
    assert report["is_error"] is True
    assert report["value"] == {"error": "cannot refund 3 with [REDACTED]"}
    assert server.trials[trial].result["outcome"] == "agent_error"
    assert all(SECRET.encode() not in body for body in server.raw_bodies())


def test_store_seed_output_and_factory_errors(server: FakeRunnerServer) -> None:
    async def read_seed(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        item = await ctx.store.aget((*ctx.store_namespace, "profile"), "customer")
        if ctx.case.case_id == "opaque":
            return {"log": ["x"]}
        return {"log": [item.value["tier"] if item else "missing"]}

    seeds = [{"namespace": ["profile"], "key": "customer", "value": {"tier": "gold"}}]
    seeded = server.add_trial(
        "v1", agent_case("seeded", input={"messages": [], "store_seed": seeds})
    )
    invalid = server.add_trial("v1", agent_case("invalid", input={"store_seed": "nope"}))
    opaque = server.add_trial("v1", agent_case("opaque"))

    def output(values: Any) -> Any:
        return object() if values["log"] == ["x"] else values["log"]

    assert serve(make_runner(server, single_node(read_seed), output_adapter=output)) == 3
    assert server.trials[seeded].result["output"]["final"] == ["gold"]
    assert server.trials[invalid].failures[0]["error_code"] == "store_seed_invalid"
    assert server.trials[opaque].failures[0]["error_code"] == "output_not_serializable"
    bound = server.add_trial("v1", agent_case("bound"))

    def bound_factory(ctx: TrialContext) -> Any:
        graph = single_node(render_plan)(ctx)
        return bind_langgraph(graph, client=Client(), agent_id=AGENT, channel="production")

    assert serve(make_runner(server, bound_factory)) == 1
    assert server.trials[bound].failures[0]["error_code"] == "factory_returned_bound_graph"


def test_checkpointer_factory_cleanup_and_trial_end(server: FakeRunnerServer) -> None:
    savers: dict[str, InMemorySaver] = {}
    ended: list[str] = []

    def saver_for(thread_key: str) -> InMemorySaver:
        savers[thread_key] = InMemorySaver()
        return savers[thread_key]

    def on_end(ctx: TrialContext) -> None:
        ended.append(ctx.trial_id)
        raise RuntimeError("hook failures are logged, never raised")

    first = server.add_trial("v1", agent_case("first"))
    runner = make_runner(
        server, single_node(render_plan), checkpointer_factory=saver_for, on_trial_end=on_end
    )
    assert serve(runner) == 1
    result = server.trials[first].result
    assert result["isolation"]["checkpointer"] == "per_trial_factory"
    assert ended == [first]
    assert all(not saver.storage for saver in savers.values())
    kept = server.add_trial("v1", agent_case("kept"))
    keeper = make_runner(
        server, single_node(render_plan), checkpointer_factory=saver_for, keep_checkpoints=True
    )
    assert serve(keeper) == 1
    assert any(saver.storage for saver in savers.values())
    assert server.trials[kept].result["outcome"] == "evaluated"


def test_deadline_reported_as_agent_timeout(server: FakeRunnerServer) -> None:
    async def slow(
        ctx: TrialContext, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {}

    trial = server.add_trial("v1", agent_case())

    def expire(item: Any) -> None:
        item.deadline_exceeded = True

    server.on_heartbeat = expire
    assert serve(make_runner(server, single_node(slow))) == 1
    result = server.trials[trial].result
    assert result["outcome"] == "agent_timeout"
    assert result["error"] == {"code": "deadline_exceeded"}
