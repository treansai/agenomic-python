from __future__ import annotations

import asyncio
import json
import operator
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import pytest
from experiment_fakes import TOKEN, FakeRunnerServer, GraphState, make_runner
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt
from langgraph_world import World, thread
from prompt_fakes import AGENT
from typing_extensions import TypedDict

import agenomic._transport as transport_module
from agenomic.cli.__main__ import main
from agenomic.experiments import TrialAssignment, TrialContext
from agenomic.experiments.runner import GraphNodeEntryPoint, SnapshotRefusedError, snapshot_case
from agenomic.integrations.langgraph_binding import prompts_for

ENTRY = {
    "name": "planner",
    "kind": "graph_node",
    "node_path": "plan",
    "seed_as_node": "intake",
    "slot_paths": ["planner.instructions"],
}


@pytest.fixture(autouse=True)
def fast_sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    async def fast(delay: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(transport_module, "_asleep", fast)
    return


def intake(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [AIMessage(content="intake done", id="m-intake")], "log": ["intake"]}


def plan(state: dict[str, Any], config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
    text = prompts_for(config).render_text("planner.instructions")
    store.put(("plans",), str(config["configurable"]["thread_id"]), {"text": text})
    return {"messages": [AIMessage(content=text, id="m-plan")], "log": [text]}


def respond(state: dict[str, Any]) -> dict[str, Any]:
    return {"log": ["respond"]}


def build(checkpointer: Any, store: Any, state: Any = GraphState) -> Any:
    builder = StateGraph(state)
    builder.add_node("intake", intake)
    builder.add_node("plan", plan)
    builder.add_node("respond", respond)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "plan")
    builder.add_edge("plan", "respond")
    return builder.compile(checkpointer=checkpointer, store=store)


class Production:
    def __init__(self, world: World) -> None:
        self.world = world
        self.saver = InMemorySaver()
        self.store = InMemoryStore()
        self.graph = build(self.saver, self.store)
        self.managed = world.bind(self.graph)
        self.managed.invoke(
            {"messages": [HumanMessage(content="refund 1234", id="m-user")]}, thread("prod")
        )
        history = list(self.graph.get_state_history(thread("prod")))
        self.before_plan = next(item for item in history if item.next == ("plan",))

    def fingerprint(self) -> tuple[Any, ...]:
        prod = thread("prod")
        return (
            [item.config["configurable"]["checkpoint_id"] for item in self.saver.list(prod)],
            json.dumps(self.graph.get_state(prod).values, default=str, sort_keys=True),
            [(item.namespace, item.key, item.value) for item in self.store.search(("plans",))],
            self.world.engine.get_channel(AGENT, "production")["generation"],
            [
                self.world.engine.get_release(self.world.releases[name])["status"]
                for name in ("v1", "v2")
            ],
        )

    def snapshot(self) -> dict[str, Any]:
        return snapshot_case(
            self.graph,
            "prod",
            agent_id=AGENT,
            checkpoint_id=self.before_plan.config["configurable"]["checkpoint_id"],
            tags=["production_snapshot"],
        )


def trial_factory(seen: list[TrialContext], state: Any = GraphState) -> Any:
    def factory(ctx: TrialContext) -> Any:
        seen.append(ctx)
        return build(ctx.checkpointer, ctx.store, state)

    return factory


def runner_for(server: FakeRunnerServer, seen: list[TrialContext], state: Any = GraphState) -> Any:
    return make_runner(
        server,
        trial_factory(seen, state),
        entry_points={"planner": GraphNodeEntryPoint("plan", seed_as_node="intake")},
    )


def test_counterfactual_new_thread_parent_binding() -> None:
    world = World.create()
    production = Production(world)
    case = production.snapshot()
    production_binding = production.before_plan.metadata["agenomic_binding_id"]
    provenance = case["input"]["provenance"]
    assert provenance == {
        "source": "production_snapshot",
        "parent_binding_id": production_binding,
        "checkpoint_id": production.before_plan.config["configurable"]["checkpoint_id"],
    }
    assert case["kind"] == "node_state"
    assert case["initial_state"]["log"] == ["intake"]
    server = FakeRunnerServer(world)
    trial_id = server.add_trial("v2", case, level="node", entry_point=ENTRY)
    seen: list[TrialContext] = []
    assert runner_for(server, seen).serve(idle_timeout=0) == 1
    assignment = server.trials[trial_id].assignment
    binding = assignment["view"]["binding"]
    assert binding["parent_binding_id"] == production_binding
    assert binding["thread_key"] == f"exp:exp_local:{trial_id}:a1"
    assert seen[0].thread_key == binding["thread_key"]
    result = server.trials[trial_id].result
    assert result["outcome"] == "evaluated"
    assert result["isolation"]["fork_fidelity"] == "state_values_only"
    assert "pending_writes" in result["isolation"]["not_preserved"]
    update = result["output"]["state_update"]
    assert update["log"] == ["intake", "PLAN v2"]
    assert update["messages"][-1]["kwargs"]["content"] == "PLAN v2"
    assert seen[0].store.search(("plans",))[0].value == {"text": "PLAN v2"}


def test_source_thread_store_channel_unchanged() -> None:
    world = World.create()
    production = Production(world)
    before = production.fingerprint()
    case = production.snapshot()
    server = FakeRunnerServer(world)
    for release in ("v1", "v2"):
        server.add_trial(release, case, level="node", entry_point=ENTRY)
    seen: list[TrialContext] = []
    assert runner_for(server, seen).serve(idle_timeout=0) == 2
    assert production.fingerprint() == before
    assert all(ctx.checkpointer is not production.saver for ctx in seen)
    assert all(ctx.store is not production.store for ctx in seen)
    assert {item.key for item in production.store.search(("plans",))} == {"prod"}
    outputs = sorted(t.result["output"]["state_update"]["log"][-1] for t in server.trials.values())
    assert outputs == ["PLAN v1", "PLAN v2"]


def test_snapshot_refuses_parallel_step() -> None:
    class S(TypedDict, total=False):
        log: Annotated[list[str], operator.add]

    builder = StateGraph(S)
    builder.add_node("p1", lambda state: {"log": ["p1"]})
    builder.add_node("p2", lambda state: {"log": ["p2"]})
    builder.add_node("join", lambda state: {"log": ["join"]})
    builder.add_edge(START, "p1")
    builder.add_edge(START, "p2")
    builder.add_edge(["p1", "p2"], "join")
    graph = builder.compile(checkpointer=InMemorySaver())
    graph.invoke({"log": []}, thread("q"))
    after = next(item for item in graph.get_state_history(thread("q")) if item.next == ("join",))
    with pytest.raises(SnapshotRefusedError) as refused:
        snapshot_case(graph, "q", checkpoint_id=after.config["configurable"]["checkpoint_id"])
    assert refused.value.reason == "parallel_step"


def test_snapshot_refuses_pending_interrupt() -> None:
    class S(TypedDict, total=False):
        log: Annotated[list[str], operator.add]

    def ask(state: dict[str, Any]) -> dict[str, Any]:
        return {"log": [f"ask:{interrupt('approve?')}"]}

    builder = StateGraph(S)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    graph = builder.compile(checkpointer=InMemorySaver())
    graph.invoke({"log": []}, thread("i"))
    with pytest.raises(SnapshotRefusedError) as refused:
        snapshot_case(graph, "i")
    assert refused.value.reason == "pending_interrupt"


def keep_max(left: int | None, right: int | None) -> int:
    return max(left or 0, right or 0) + (1 if right is not None else 0)


class Scored(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    score: Annotated[int, keep_max]


def test_snapshot_refuses_non_identity_reducer() -> None:
    builder = StateGraph(Scored)
    builder.add_node("x", lambda state: {"log": ["x"], "score": 5})
    builder.add_node("y", lambda state: {"log": ["y"]})
    builder.add_edge(START, "x")
    builder.add_edge("x", "y")
    graph = builder.compile(checkpointer=InMemorySaver())
    graph.invoke({"log": []}, thread("r"))
    after = next(item for item in graph.get_state_history(thread("r")) if item.next == ("y",))
    with pytest.raises(SnapshotRefusedError) as refused:
        snapshot_case(graph, "r", checkpoint_id=after.config["configurable"]["checkpoint_id"])
    assert refused.value.reason == "non_identity_reducer"
    with pytest.raises(SnapshotRefusedError) as missing:
        snapshot_case(graph, "never-ran")
    assert missing.value.reason == "thread_not_found"
    with pytest.raises(ValueError):
        snapshot_case(builder.compile(), "r")


def test_snapshot_refuses_other_agent_and_bound_graph() -> None:
    world = World.create()
    production = Production(world)
    with pytest.raises(SnapshotRefusedError) as other:
        snapshot_case(production.graph, "prod", agent_id="3c4d5e6f-7a8b-4c9d-8e0f-1a2b3c4d5e6f")
    assert other.value.reason == "agent_mismatch"
    with pytest.raises(ValueError):
        snapshot_case(production.managed, "prod")


def doubling(left: list[str] | None, right: list[str] | None) -> list[str]:
    return [*(left or []), *(right or []), *(right or [])]


class Doubling(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], doubling]


def test_seed_mismatch_reports_fork_unsupported() -> None:
    world = World.create()
    case = Production(world).snapshot()
    server = FakeRunnerServer(world)
    trial_id = server.add_trial("v2", case, level="node", entry_point=ENTRY)
    assert runner_for(server, [], Doubling).serve(idle_timeout=0) == 1
    failure = server.trials[trial_id].failures[0]
    assert (failure["error_class"], failure["error_code"]) == (
        "runner_configuration",
        "fork_unsupported",
    )
    assert server.trials[trial_id].result is None


def test_redelivered_result_same_lease_same_digest() -> None:
    world = World.create()
    case = Production(world).snapshot()
    server = FakeRunnerServer(world)
    trial_id = server.add_trial("v2", case, level="node", entry_point=ENTRY)
    server.drop_result_responses = 1
    runner = runner_for(server, [])
    assert runner.serve(idle_timeout=0) == 1
    trial = server.trials[trial_id]
    assert (trial.accepts, trial.duplicates) == (1, 1)
    posts = [r.content for r in server.requests if r.url.path.endswith("/result")]
    assert len(posts) == 2
    assert posts[0] == posts[1]
    assert runner._outbox == {}
    assignment = TrialAssignment.model_validate(trial.assignment)
    http = runner._connect()

    async def report(token: str, result: dict[str, Any]) -> Any:
        target = assignment.model_copy(update={"lease_token": token})
        return await runner._report_result(http, target, result)

    accepted = trial.result
    assert asyncio.run(report(assignment.lease_token, accepted)) == {
        "accepted": True,
        "duplicate": True,
    }
    assert (
        asyncio.run(report(assignment.lease_token, {**accepted, "outcome": "agent_error"})) is None
    )
    assert asyncio.run(report("00000000-0000-4000-8000-000000000000", accepted)) is None
    assert (trial.accepts, trial.duplicates) == (1, 2)


def test_cli_experiment_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    production = Production(World.create())
    module = types.ModuleType("agenomic_test_snapshot_target")
    module.graph = production.graph
    module.factory = lambda: production.graph
    monkeypatch.setitem(sys.modules, module.__name__, module)
    out = tmp_path / "case.json"
    checkpoint = production.before_plan.config["configurable"]["checkpoint_id"]
    code = main(
        [
            "experiment",
            "snapshot",
            "--graph",
            f"{module.__name__}:graph",
            "--agent",
            AGENT,
            "--thread",
            "prod",
            "--checkpoint-id",
            checkpoint,
            "--case-id",
            "refund-cf-1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["case_id"] == "refund-cf-1"
    assert written["input"]["provenance"]["checkpoint_id"] == checkpoint
    assert (
        main(
            [
                "experiment",
                "snapshot",
                "--graph",
                f"{module.__name__}:factory",
                "--agent",
                AGENT,
                "--thread",
                "missing",
            ]
        )
        == 1
    )
    assert (
        main(["experiment", "snapshot", "--graph", "nope", "--agent", AGENT, "--thread", "prod"])
        == 2
    )
    assert (
        main(
            [
                "experiment",
                "snapshot",
                "--graph",
                f"{module.__name__}:graph",
                "--agent",
                "AGENT",
                "--thread",
                "prod",
            ]
        )
        == 2
    )
    assert TOKEN not in out.read_text(encoding="utf-8")
