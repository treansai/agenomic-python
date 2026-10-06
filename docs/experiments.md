# Prompt experiments

An experiment compares a baseline release of an agent with one to five
candidate releases, for example a candidate agent version that changes one
prompt, on a frozen dataset of cases. Agenomic Cloud plans the trials, leases
them to runners that you operate and computes paired statistics over the
results. Your agent code, your model credentials and your tools stay on your
machines: a runner receives one trial at a time, runs it in isolation with
the pinned prompts of its arm and reports a result document.

The Python SDK has three parts:

- `client.experiments` creates, checks, launches and reads experiments;
- `ExperimentRunner` (`agenomic.experiments`) is the runner that executes
  trials of a LangGraph agent;
- `agenomic-py experiment serve` runs a runner, and
  `agenomic-py experiment snapshot` freezes the state of a production thread
  into a counterfactual case.

```bash
pip install agenomic
pip install "agenomic[langgraph]"
```

`client.experiments` needs only the core package. The runner imports LangGraph
and LangChain, so it needs the `langgraph` extra; `import agenomic.experiments`
alone loads neither. The case and frozen spec formats
(`agenomic.experiment_case/v1`, `agenomic.experiment_spec/v1`) are defined by
agenomic-spec.

## How an experiment runs

1. A dataset version holds the cases. A case is an agent input
   (`agent_input`, with optional `turns` for a conversation) or the state of
   a graph just before one node (`node_state`).
2. You create a draft that names the agent, the baseline release, the
   candidate releases, the dataset version, the evaluators and the metrics.
3. The preflight freezes the spec, reports confounders, estimates and
   eligible runners, and returns the `spec_digest` of the frozen spec.
4. The launch cites that digest and acknowledges what the preflight asked
   for. Agenomic Cloud then queues one trial per case, arm and repetition.
5. Registered runners claim the trials. A trial carries a blinded view: the
   case, an opaque arm key, the arm's release and pinned prompts, and an
   execution binding that Agenomic Cloud created for that trial. The runner
   never learns which arm is the baseline.
6. The runner runs the trial on a fresh thread and reports the result.
   Agenomic Cloud evaluates it and, once the trials are done, computes the
   paired comparisons and the verdict.

Experiments need Agenomic Cloud and the `prompts.experiments` capability. In
local mode (`Client()` without `base_url`) every `client.experiments` call
raises `ApiError("cloud_required")`. A runner can still run a trial offline
for development ([Trying a target offline](#trying-a-target-offline)).

## Launching an experiment

```python
from agenomic import Client

client = Client(api_key="agm_...", base_url="https://agenomic.example")
draft = {
    "name": "Planner checks the order status",
    "agent_id": "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c",
    "baseline_release_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
    "candidate_release_ids": ["a2b3c4d5-e6f7-4a8b-9c0d-1e2f3a4b5c6d"],
    "dataset": {"dataset_id": "exds_01jag7k2m9p4r6t8v0w2x4y6z8", "version": 3},
    "repetitions": 3,
    "evaluators": [
        {
            "evaluator_id": "decision",
            "kind": "expected_match",
            "version": 1,
            "config": {"mode": "json_subset", "success_source": True},
        }
    ],
    "metrics": {
        "primary": {
            "metric_id": "task_success",
            "direction": "higher_is_better",
            "rule": "superiority",
            "margin": "0",
        },
        "guardrails": [],
        "reported": ["latency_ms"],
    },
}
experiment = client.experiments.create(draft)
experiment_id = experiment["experiment_id"]
preflight = client.experiments.preflight(
    experiment_id, expected_revision=experiment["revision"]
)
frozen = client.experiments.get(experiment_id)
spec = frozen["spec"]
launched = client.experiments.launch(
    experiment_id,
    expected_revision=frozen["revision"],
    spec_digest=preflight["spec_digest"],
    authorization={
        "paid_model_usage": {
            "acknowledged": True,
            "max_total_tokens": spec["budgets"]["max_total_tokens"],
            "max_cost_micros": spec["budgets"]["max_cost_micros"],
        },
        "tool_plan_hashes": preflight["tool_plan_hashes"],
        "acknowledged_confounders": [
            confounder["id"] for confounder in spec["confounders"]
        ],
    },
    idempotency_key="launch-planner-check-1",
)
print(launched["experiment"]["status"])

page = client.experiments.events(experiment_id, after=0)
for event in page["events"]:
    print(event["sequence"], event["kind"])
results = client.experiments.results(experiment_id)
print(results["interim"], results["verdict"])
```

- Creating, updating, preflighting, launching and cancelling need a
  maintainer or an owner, with a signed-in session or a `write` key. Every
  member can read.
- `update(experiment_id, draft, expected_revision=n)` and `preflight` send
  the revision you read as `If-Match`; a stale one raises
  `experiment_revision_conflict`. Updating a draft invalidates its
  preflight.
- `launch` cites the revision and the preflight's `spec_digest`.
  `authorization` acknowledges paid model usage with the budgets of the
  frozen spec, the tool plan hashes of the preflight and every confounder it
  found. The launch is idempotent on `idempotency_key` (`[A-Za-z0-9._:-]`, at
  most 128 characters): sending the same launch again returns the first one.
- An experiment with live tools is launched only from a signed-in owner
  session; an API key gets `experiment_live_tools_owner_session_required`.
- `create`, `update`, `get` and `launch` recompute the `spec_digest` of a
  frozen spec (sha256 of the canonical spec without its `identity` member)
  and raise `ApiError("experiment_spec_digest_mismatch")` on a difference.
- `events(experiment_id, after=n, limit=m)` returns up to `m` events (1 to
  500) after sequence `n`, with `next_after`. `results` returns the results
  document: paired comparisons per candidate and metric, per-arm summaries,
  the limitations and, once the experiment is complete, the verdict.
  `cancel(experiment_id, reason=...)` stops an experiment and keeps what was
  accepted.
- Every call has an `a*` twin (`acreate`, `apreflight`, `alaunch`, `aget`,
  `aresults`, `aevents`, ...). Server refusals are `ApiError` with the
  server's code.

Datasets, runner registration, progress, trials and evidence are not wrapped
by the SDK yet: use the web app or the HTTP API.

## Runner targets

A runner serves one or more agents. For each agent it holds a `GraphTarget`,
whose factory builds your graph for one trial. Write the graph builder so that
it takes its checkpointer and store, and use the same code in production and
in trials:

```python
import itertools
import operator
from typing import Annotated, Any

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agenomic.experiments import TrialContext
from agenomic.integrations import prompts_for


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    notes: Annotated[list[str], operator.add]


def build_graph(checkpointer: Any, store: Any) -> Any:
    model = GenericFakeChatModel(
        messages=itertools.cycle([AIMessage(content="refund")])
    )

    def intake(state: State) -> State:
        return {"notes": ["intake: request received"]}

    def planner(state: State, config: RunnableConfig) -> State:
        prompts = prompts_for(config)
        request = str(state["messages"][0].content)
        system = prompts.render_text(
            "planner.instructions", {"request": request}
        )
        reply = model.invoke(
            [SystemMessage(system), *state["messages"]],
            prompts.config_for("planner.instructions"),
        )
        return {"messages": [reply], "notes": [f"planner: {system}"]}

    builder = StateGraph(State)
    builder.add_node("intake", intake)
    builder.add_node("planner", planner)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "planner")
    return builder.compile(checkpointer=checkpointer, store=store)


def trial_graph(ctx: TrialContext) -> Any:
    return build_graph(ctx.checkpointer, ctx.store)
```

The model is a fake so that the examples of this page run offline; use your
own chat model.

- `factory(ctx)` builds a new graph for every trial, and may be a coroutine
  function. It compiles the graph with `ctx.checkpointer` and `ctx.store`
  (otherwise the trial fails with `isolation_violation`) and returns the
  compiled graph, not a `bind_langgraph` proxy
  (`factory_returned_bound_graph`): the runner binds it to the trial's
  execution binding itself.
- Nodes read prompts only through `prompts_for(config)`, as in production
  ([LangGraph managed prompts](integrations.md#langgraph-managed-prompts)),
  and pass `prompts.config_for(slot)` to the model call, so the result
  attributes each model call to its slot and prompt version. A model call
  whose prompt is outside the arm's manifest fails the trial
  (`prompt_outside_manifest`).
- `runtime_digest` declares the agent runtime deployed on this runner: the
  bundle hash of the agent bundle the releases were built from. A trial
  whose arm needs another runtime fails with `runtime_digest_mismatch`, so
  arms that differ only in prompts share one target. Agenomic Cloud cannot
  check this declaration, and every result says
  `"runtime_digest_source": "runner_declared"`.
- `input_adapter(case)` builds the graph input. The default passes
  `case.input` without its `context`, `store_seed` and `provenance` members;
  `case.input.context` goes to `config["configurable"]`.
- `output_adapter(values)` builds the reported `output.final` from the final
  state. The default is the content of the last message, or the whole state
  without a message list.
- `turn_adapter(text)` builds the input of a `{"user": text}` turn of a
  multi-turn case; the default adds one user message. A `{"resume": value}`
  turn resumes an interrupt. All turns run on the trial's one thread.
- `children` maps node paths to child agents, as in `bind_langgraph`.
- `entry_points`, `evaluators`, `live_tools`, `checkpointer_factory`,
  `keep_checkpoints`, `store` and `on_trial_end` are described below.

The trial context `ctx` carries `experiment_id`, `trial_id`, `attempt`,
`arm_key`, `agent_id`, `thread_key`, `case` (an `ExperimentCase`), `level`,
`seed` (an integer), `resource_suffix`, `store_namespace`, `tool_mode`,
`checkpointer`, `store`, `secrets` and `wrap_tools(tools)`.

## Trying a target offline

`local_assignment` builds a trial assignment for a release of the local
registry, and `ExperimentRunner.run_trial` runs it in-process: no Agenomic
Cloud, no runner token. This is a simulation for development; in Agenomic
Cloud the server creates every trial binding and leases the trial to a
registered runner.

```python
from agenomic import Client
from agenomic.experiments import (
    ExperimentRunner,
    GraphNodeEntryPoint,
    GraphTarget,
    local_assignment,
)

AGENT_ID = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
BODIES = [
    "Plan the next step for: {request}",
    "Plan the next step for: {request}. Check the order status first.",
]


def planner_content(body: str) -> dict:
    return {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": {"request": {"type": "string", "required": True}},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }


client = Client()
engine = client.prompts.local
client.prompts.create("prm_planner", name="Planner", kind="text")
releases = []
for number, body in enumerate(BODIES, start=1):
    client.prompts.publish(
        "prm_planner",
        planner_content(body),
        parent_version=None if number == 1 else number - 1,
        change_message=f"Version {number}",
    )
    releases.append(
        engine.create_release(
            AGENT_ID, {"planner.instructions": f"prm_planner:{number}"}
        )
    )
baseline, candidate = releases

target = GraphTarget(
    factory=trial_graph,
    runtime_digest=engine.get_release(baseline)["bundle_hash"],
    entry_points={
        "planner": GraphNodeEntryPoint("planner", seed_as_node="intake")
    },
)
runner = ExperimentRunner(targets={AGENT_ID: target})

case = {
    "case_id": "refund-1001",
    "kind": "agent_input",
    "input": {
        "messages": [{"role": "user", "content": "Refund order 1001"}]
    },
    "expected": {"decision": "refund"},
}
for release_id in (baseline, candidate):
    assignment = local_assignment(
        engine, agent_id=AGENT_ID, release_id=release_id, case=case
    )
    run = runner.run_trial(assignment)
    print(run.kind, run.result["outcome"], run.result["output"]["final"])
    print([call["prompt_ref"] for call in run.result["model_calls"]])
```

`run_trial` returns a `TrialRun` whose `kind` is `result` (with the result
document in `run.result`) or `failure` (with `error_class`, `error_code` and
`message`). `arun_trial` is its async twin. Pass `proxy=` to route tool calls
to a tool proxy of your own; without one, a tool call fails the trial.

## Serving trials

A workspace owner registers the runner in Agenomic Cloud, names the agents it
may serve and receives its runner token, `agr_` followed by 64 lowercase
hexadecimal characters, once. The token authenticates the runner routes only:
it cannot read experiments, and API keys and sessions cannot claim trials.

```python
import os

from agenomic.experiments import EnvSecretResolver

runner = ExperimentRunner(
    targets={
        AGENT_ID: GraphTarget(
            factory=trial_graph,
            runtime_digest=os.environ["AGENT_RUNTIME_DIGEST"],
            entry_points={
                "planner": GraphNodeEntryPoint(
                    "planner", seed_as_node="intake"
                )
            },
        )
    },
    base_url="https://agenomic.example",
    secrets=EnvSecretResolver(allow=["OPENAI_API_KEY"]),
    max_concurrency=4,
)
completed = runner.serve(idle_timeout=600)
print(completed)
```

- The token comes from `token=` or from `AGENOMIC_RUNNER_TOKEN`, never from a
  command-line flag, so it stays out of shell history. It is sent only as the
  bearer header. The endpoint comes from `base_url=` or `AGENOMIC_ENDPOINT`.
- `serve()` (or `await aserve()`) sends a hello, then runs `max_concurrency`
  workers (1 to 64, default 4). Each worker claims one trial at a time,
  executes it, heartbeats while it runs and reports it. `serve` returns the
  number of trials processed.
- `serve(max_trials=n)` stops after n claimed trials, and
  `serve(idle_timeout=s)` once no trial was claimed for s seconds. Without
  either it runs until `stop()`, which lets the trials in flight finish and
  releases a trial claimed afterwards with the reason `shutdown`.
- The hello declares, per agent, the runtime digest, the levels (`agent`,
  and `node` when the target has entry points), the tool modes, the entry
  points, the custom evaluators and the names of the secret refs the runner
  can resolve, never their values. The preflight uses it to find eligible
  runners. The runner sends it again every 60 seconds.
- A heartbeat runs at the interval of the assignment. When Agenomic Cloud
  cancels the experiment or a stop condition fires, the trial is cancelled
  and released (`cancelled` or `stopped`). A trial past its deadline is
  reported as `agent_timeout`. A trial whose lease is no longer current is
  dropped without a report.
- After the first hello, `serve` stops only when Agenomic refuses the
  runner's own credentials or configuration. A failed claim, a gateway
  outage or a failing trial is logged on the `agenomic.experiments` logger,
  and the runner goes on. The first hello fails fast.
- A runner token cannot read its workspace, so the first assignment pins it
  (or pass `workspace_id=`), and an assignment of another workspace fails
  with `workspace_mismatch`.

From the command line:

```bash
export AGENOMIC_ENDPOINT=https://agenomic.example
export AGENOMIC_RUNNER_TOKEN=agr_...
agenomic-py experiment serve --target support_agent.runner:runner
```

- `--target module:attribute` names an `ExperimentRunner`, or a function
  without arguments that returns one. The module is imported, so it must be
  importable, for example through `PYTHONPATH`.
- `--max-concurrency` (1 to 64) replaces the runner's concurrency;
  `--max-trials` and `--idle-timeout` (seconds) bound the run.
- The command prints `runner stopped after <n> trial(s)` and exits 0. It
  exits 1 when Agenomic refuses the runner (`error: <code>: <message>`) and 2
  when the target cannot be loaded or the token or the endpoint is missing.

## Isolation

Every trial runs in isolation, and the runner enforces it:

1. **Fresh thread.** The thread id is the trial binding's thread key,
   `exp:<experiment>:<trial>:a<attempt>`. It names neither the arm nor the
   case. A multi-turn case reuses it for its own turns only.
2. **Checkpointer per trial.** A new `InMemorySaver`, or your
   `checkpointer_factory(thread_key)`, whose thread is deleted after the
   trial unless `keep_checkpoints=True`.
3. **Store per trial.** A new `InMemoryStore`, or your `GraphTarget.store`
   wrapped in `NamespacedStore`, which refuses every operation outside
   `ctx.store_namespace` (`isolation_violation`). Prefix your store
   namespaces with `ctx.store_namespace`; the `case.input.store_seed` items
   (`{"namespace": [...], "key": ..., "value": {...}}`) are written under it
   before the trial. Nothing deletes what a trial wrote into your store
   (`on_trial_end` may), so `GraphTarget.store` must never be a production
   store.
4. **Nothing shared.** The factory builds a new graph for each trial, and
   concurrent trials run in separate asyncio tasks, so arms and repetitions
   share no graph, state or prompt object.
5. **Pinned prompts.** Prompts come only from the trial binding and the
   arm's prompts, verified against the arm's manifest digest. The runner
   never resolves a channel or an alias.
6. **Wrapped tools.** Tools reach the outside only through
   `ctx.wrap_tools(...)` ([Tools](#tools)). A call that a node makes
   directly, outside a wrapped tool, is not intercepted.
7. **Test resources.** Name external test resources with
   `ctx.resource_suffix`. `on_trial_end(ctx)` runs after every trial,
   failed ones included.

Before your factory runs, the runner checks that the assignment agrees with
itself: the view and the envelope name the same trial and attempt; the
binding is a thread binding with the trial's key, release and arm key; the
arm, the binding and the prompts name one manifest digest; the arm's runtime
digest is the target's; the trial's tool mode is one the target serves. A
mismatch is a `runner_configuration` failure and nothing runs.

Every result discloses the isolation it had:

```json
{
  "fresh_thread": true,
  "checkpointer": "per_trial_in_memory",
  "store": "per_trial_namespace",
  "tool_routing": "sdk_wrapped_tools",
  "fork_fidelity": "none",
  "not_preserved": [
    "pending_writes",
    "channel_versions",
    "checkpoint_history",
    "subgraph_checkpoints",
    "store_contents",
    "external_side_effects"
  ]
}
```

`checkpointer` is `per_trial_in_memory` or `per_trial_factory`, `store` is
always `per_trial_namespace`, and `fork_fidelity` is `none` for a case and
`state_values_only` for a [production snapshot](#counterfactuals-from-a-production-thread).

## Tools

Build your tool nodes from `ctx.wrap_tools(tools)`, which takes LangChain
tools or plain functions, for example `ToolNode(ctx.wrap_tools([lookup]))`.
Each call goes to the trial's tool proxy in Agenomic Cloud, which answers
according to the experiment's tool mode:

| Mode | A tool call |
| --- | --- |
| `none` | fails the trial (`tool_mode_none`), with no request |
| `mock` | returns the mock response of the experiment's tool configuration |
| `recorded` | returns the recorded response that matches the call |
| `live` | runs your tool on the runner, once Agenomic authorized the call |

- In `recorded` mode a miss ends the trial with the outcome
  `recorded_fixture_miss` when the spec says `"on_fixture_miss": "stop_trial"`,
  or returns the text `no recorded response is available for this call` to
  the agent with `"tool_error"`. Several matching recordings end it with
  `recorded_fixture_ambiguous`.
- A call that a Protect policy denies returns
  `this action was denied by policy` with the policy's explanation, and one
  that needs a human approval returns `this action requires human approval,
  which is not available inside experiments`. The agent sees both as tool
  output.
- Live tools run only on a runner that opts in with
  `GraphTarget(live_tools=True)`, and only in a trial whose mode is `live`.
  A live trial claimed by a runner without that opt-in fails with
  `tool_mode_unavailable` before anything runs, and an authorization for a
  trial that is not live fails it with `live_tools_disabled` without running
  the tool.
- A live call runs your tool once per call id. Retries resend the same
  report and never repeat the side effect, and two concurrent calls with
  one id fail the trial (`live_call_concurrent`).
- The call id is the model's tool call id, else a hash of the graph
  namespace, the task, the tool and its arguments. Arguments are sent as the
  model produced them, except resolved secret values, which become
  `[REDACTED]`, so such a call may not match a recording.
- Once a trial has ended (a fixture miss, a budget, a refused call), every
  later tool call fails before any request, even when a `ToolNode` with
  `handle_tool_errors` catches the error.

## Secrets

- A spec names secret refs such as `env:OPENAI_API_KEY`. The runner
  resolves them with the resolver passed as `secrets=`. `EnvSecretResolver`
  reads only the environment variables listed in `allow`; a resolver of
  your own implements `names()` and `resolve(ref)` (`SecretResolver`).
- A ref that is not allowed, or has no value, fails the trial with
  `secret_unresolved` before your factory runs.
- Values reach your code only as `ctx.secrets[ref]`, a read-only mapping
  that hides its values in `repr` and cannot be pickled. Read them in the
  factory, by closure. Never put them in graph state, `configurable` or run
  metadata, which LangGraph copies into checkpoints.
- Before a result leaves the runner, members named like `password`, `token`,
  `api_key`, `secret` or `authorization` are masked, then every resolved
  value, also in its quoted, JSON and percent-encoded forms, becomes
  `[REDACTED]`. Failure and error messages also have the secret patterns of
  managed prompts and any runner token replaced, and are cut to 2 KiB.
  Agenomic Cloud scans every result again and refuses one that matches a
  secret pattern; the trial then fails with `secret_in_report`.
- Records of the `agenomic.experiments` logger, tracebacks included, are
  redacted the same way. Records of other loggers (`httpx`, LangChain,
  provider SDKs) are not.

## Evaluators

Agenomic Cloud runs the deterministic evaluators of the spec when it accepts
a result. Two kinds run on the runner:

- **Custom evaluators.** Declare them on the target, for example
  `GraphTarget(..., evaluators={"refund_ok": RunnerEvaluator(check, 3)})`,
  where `check(case, output)` returns a number, a boolean or `None`, and
  `value_kind` is `binary` (the default) or `continuous`. The hello declares
  each with its `code_digest`: by default the sha256 of the function's
  source, or the `code_digest=` you pin. A trial whose spec names another
  version or digest fails with `evaluator_unavailable`. An evaluator that
  raises reports `None`.
- **Model judges** run only when the runner has `judge_model=`, a function
  from the judge's model settings to a LangChain chat model. The judge
  prompt is rendered with `input`, `output`, `expected` and `rubric`, and the
  reply must be JSON with an integer `score` within the rubric's scale.
  Without `judge_model`, and on any judge failure, the score is `null` with
  `"parsed": false`. A judge never changes the trial outcome.

## Budgets, outcomes and failures

- The view's `max_model_calls_per_trial` and `max_tokens_per_trial` are
  enforced by a callback: the first model call past a limit raises, and the
  trial is reported as `budget_stopped`. Tokens are counted only when the
  provider reports usage; a call without usage is reported with
  `"usage_source": "not_reported"` and null counts.
- A reported result has one of the outcomes `evaluated`, `agent_error`,
  `budget_stopped`, `recorded_fixture_miss`, `recorded_fixture_ambiguous`
  and `agent_timeout`.
- Other errors are reported as failures, never as results. Agenomic Cloud
  retries an `infrastructure` failure (a provider rate limit, timeout or
  outage, a connection error, Agenomic unavailable) and does not retry a
  `runner_configuration` failure (an assignment the runner cannot serve, an
  isolation violation, a missing secret).
- `classify` applies the default rules: a `RunnerConfigurationError` is
  `runner_configuration`; transport errors, provider errors named
  `APIConnectionError`, `APITimeoutError`, `RateLimitError` or
  `InternalServerError`, HTTP statuses 408, 409, 429 and 5xx, and
  `registry_unavailable` are `infrastructure`; everything else, a render
  error or a slot missing from a candidate's manifest included, is an agent
  error of the arm under test.
- `ExperimentRunner(classify_error=hook)` runs `hook(error)` first. It
  returns `"infrastructure"`, `"runner_configuration"`, `"agent"` or `None`
  for the default rules, and the result records
  `"error_class_source": "user_hook"` when it decided.

## Node experiments

A node experiment runs one node of your graph on `node_state` cases. Declare
the node on the target as an entry point, under the name the spec uses:
`entry_points={"planner": GraphNodeEntryPoint("planner",
seed_as_node="intake")}`.

- The runner writes `case.initial_state` once with
  `update_state(..., as_node=seed_as_node)` on the fresh thread, then
  requires that the entry node is the only next node; otherwise the trial
  fails with `entry_point_not_next` before any node runs. Choose as
  `seed_as_node` a node whose only successor is the entry node (the default
  `__start__` when the entry node comes first).
- The node runs with `interrupt_after` on itself. The output is
  `{"state_update": ..., "extra_nodes_executed": []}`, the channels whose
  value changed, in the serialized form of the case. When another node ran
  too, the trial fails with `entry_point_not_next`.
- A node case takes only `{"resume": value}` turns
  (`node_turn_unsupported`).
- The spec's entry point (name, kind, node path and `seed_as_node`) must
  match the runner's declaration (`entry_point_unavailable`). The runner
  does not check `seed_as_node` against your graph when it says hello; the
  trial-time check above does.
- `CallableEntryPoint(fn)` declares a function `fn(state, config)` that
  returns a state update. It runs alone in a one-node graph bound to the
  trial binding, so `prompts_for(config)` works, with `case.initial_state`
  as its input and no turns. It gets no trial context, so it cannot route
  tools through the trial proxy, and an `interrupt()` in it is an agent
  error (`interrupt_unsupported_in_callable`).

## Counterfactuals from a production thread

To see what a candidate prompt would have done at one point of a real
conversation, freeze the state of a production thread into a `node_state`
case, add the case to a dataset version and run a node experiment on it.

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agenomic.experiments import snapshot_case
from agenomic.integrations import bind_langgraph

engine.move_channel(AGENT_ID, "production", baseline, expected_generation=0)
production_graph = build_graph(InMemorySaver(), InMemoryStore())
production = bind_langgraph(
    production_graph, client=client, agent_id=AGENT_ID, channel="production"
)
thread = {"configurable": {"thread_id": "customer-7"}}
production.invoke({"messages": [HumanMessage("Refund order 1234")]}, thread)

before_planner = next(
    state
    for state in production_graph.get_state_history(thread)
    if state.next == ("planner",)
)
snapshot = snapshot_case(
    production_graph,
    "customer-7",
    agent_id=AGENT_ID,
    case_id="refund-1234-counterfactual",
    checkpoint_id=before_planner.config["configurable"]["checkpoint_id"],
)
print(snapshot["input"]["provenance"])
```

`create_release` and `move_channel` are calls of the local engine, a
simulation of the governed path: in Agenomic Cloud a channel move is a
session-only, approved action, and the SDK cannot make it.

- `snapshot_case` reads the thread through your own checkpointer and never
  writes it, its store or its release. Pass the compiled graph, not the
  `bind_langgraph` proxy. Nothing leaves your machine until you upload the
  case.
- Pass the `checkpoint_id` whose `next` is the entry node, and declare that
  checkpoint's writer as the entry point's `seed_as_node`. Without
  `checkpoint_id` the latest checkpoint is used: for a finished thread that
  is its end state, and the trial would run the node again on that state.
- It raises `SnapshotRefusedError`, with the reason in `error.reason`, for a
  thread without checkpoints (`thread_not_found`), a pending interrupt
  (`pending_interrupt`), a subgraph task in flight (`subgraph_in_flight`), a
  checkpoint without exactly one writer, such as the input checkpoint
  (`parallel_step`), a thread bound to another agent (`agent_mismatch`),
  values that cannot be serialized (`not_serializable`), and values that,
  written again, the channel reducers would change (`non_identity_reducer`)
  or that would schedule other nodes (`next_mismatch`).
- `initial_state` holds the channel values in LangChain's serialized form;
  at trial time only chat messages are rebuilt from it. `input.provenance`
  records `"source": "production_snapshot"`, the `parent_binding_id` of the
  thread's binding (`null` for a thread that `bind_langgraph` never bound)
  and the `checkpoint_id`.
- At trial time the case runs on a new thread whose binding records the
  production binding as its parent. The seeded state must equal the
  snapshot (`fork_unsupported` otherwise), and the result discloses
  `"fork_fidelity": "state_values_only"`: pending writes, checkpoint
  history, subgraph checkpoints and store contents are not carried over.

Run the snapshot against both releases offline, as a node trial:

```python
offline = ExperimentRunner(targets={AGENT_ID: target})
entry_point = {
    "name": "planner",
    "kind": "graph_node",
    "node_path": "planner",
    "seed_as_node": "intake",
    "slot_paths": ["planner.instructions"],
}
for release_id in (baseline, candidate):
    assignment = local_assignment(
        engine,
        agent_id=AGENT_ID,
        release_id=release_id,
        case=snapshot,
        level="node",
        entry_point=entry_point,
    )
    run = offline.run_trial(assignment)
    print(run.result["output"]["state_update"]["notes"][-1])
print(run.result["isolation"]["fork_fidelity"])
```

From the command line, `--graph` names the compiled graph with your
production checkpointer, or a function without arguments that returns it:

```bash
agenomic-py experiment snapshot --graph support_agent.graph:graph \
  --agent "$AGENT_ID" --thread customer-7 \
  --checkpoint-id "$CHECKPOINT_ID" --case-id refund-1234 --out case.json
```

It writes the case to `--out` (or standard output), prints
`wrote node_state case <case_id> from checkpoint <checkpoint_id>` on
standard error and exits 0. A refusal prints
`error: snapshot refused (<reason>): <message>`, writes nothing and exits 1;
a usage error exits 2. `examples/17_prompt_experiment_counterfactual.py` runs
the whole counterfactual offline and proves the production thread unchanged.

## Trial results

A result document (`agenomic.experiment_trial_result/v1`), trimmed from the
baseline trial of [Trying a target offline](#trying-a-target-offline):

```json
{
  "schema": "agenomic.experiment_trial_result/v1",
  "trial_id": "extr_01m47...",
  "attempt": 1,
  "outcome": "evaluated",
  "error": null,
  "output": {
    "final": "refund",
    "messages": [
      {"role": "user", "content": "Refund order 1001"},
      {"role": "assistant", "content": "refund"}
    ]
  },
  "turns": [{"index": 0, "interrupted": false, "latency_ms": 1}],
  "model_calls": [
    {
      "slot_path": "planner.instructions",
      "prompt_ref": "prm_planner:1",
      "content_digest": "sha256:...",
      "rendered_hash": "sha256:...",
      "protect_overlay_digest": null,
      "provider": "genericfakechatmodel",
      "model": null,
      "input_tokens": null,
      "output_tokens": null,
      "usage_source": "not_reported",
      "latency_ms": 0,
      "seed_applied": false
    }
  ],
  "measurements": {"wall_clock_ms": 2},
  "runner_metrics": {},
  "judge_scores": [],
  "artifacts": [],
  "runtime": {
    "runtime_digest": "blake3:...",
    "runtime_digest_source": "runner_declared",
    "sdk": "agenomic-python/<version>"
  },
  "runner_view_digest": "sha256:...",
  "isolation": {
    "fresh_thread": true,
    "checkpointer": "per_trial_in_memory",
    "store": "per_trial_namespace",
    "tool_routing": "sdk_wrapped_tools",
    "fork_fidelity": "none",
    "not_preserved": ["pending_writes", "...", "external_side_effects"]
  },
  "error_class_source": "default_rules/v1"
}
```

- `output` is `{"final", "messages"}` for an agent trial and
  `{"state_update", "extra_nodes_executed"}` for a node trial.
- `error` is `null` for `evaluated`, `{"code", "type", "message"}` for
  `agent_error`, `{"code": "trial_budget_exceeded", "limit"}` for
  `budget_stopped`, `{"code": "recorded_fixture_miss", "tool",
  "arguments_hash"}`, `{"code": "recorded_fixture_ambiguous", "tool"}` and
  `{"code": "deadline_exceeded"}` for `agent_timeout`.
- The result echoes `runner_view_digest` and is built once per lease.
  Retries resend the same bytes, so Agenomic Cloud accepts it once.

## Runner error codes

A failure carries its class and one of these codes:

- before anything runs: `assignment_invalid`, `agent_not_served`,
  `runtime_digest_mismatch`, `tool_mode_unavailable`, `binding_mismatch`,
  `workspace_mismatch`, `artifact_digest_mismatch`, `secret_unresolved`;
- isolation and state: `isolation_violation`,
  `factory_returned_bound_graph`, `store_seed_invalid`,
  `state_not_serializable`, `output_not_serializable`;
- node experiments: `entry_point_unavailable`, `entry_point_not_next`,
  `fork_unsupported`, `initial_state_invalid`, `node_turn_unsupported`;
- prompts and evaluators: `prompt_outside_manifest`,
  `evaluator_unavailable`;
- tools: `tool_mode_none`, `tool_arguments_not_json`, `proxy_unavailable`,
  `live_tools_disabled`, `live_call_concurrent`, and any other refusal of
  the tool proxy under its code without the `experiment_` prefix;
- reports: `secret_in_report`, or the reason of a refused result;
- infrastructure: `connection_error`, `provider_rate_limited`,
  `provider_timeout`, `provider_unavailable`, `provider_conflict`,
  `agenomic_unavailable`, `tool_call_in_progress`, `invalid_response`,
  `runner_internal_error`.

An `agent_error` result carries `agent_exception`, a provider code, the code
of a prompt error raised in the graph (`prompt_render_error`,
`slot_not_in_manifest`, ...) or `interrupt_unsupported_in_callable`.

## Limits

- Graphs run with `ainvoke`, so a heartbeat can cancel them. On Python 3.10
  LangGraph cannot run `interrupt()` there: cases that interrupt need
  Python 3.11 or later, and are agent errors on 3.10.
- A model call is attributed to the first slot of its `config_for`
  metadata.
- Agenomic Cloud cannot verify that the runner runs the code its runtime
  digest names.
- Calls made outside wrapped tools are not intercepted, and loggers other
  than `agenomic.experiments` are not redacted.

## Not in this release

These parts of experiments are not in this SDK release yet:

- dataset management, runner registration and revocation, progress, trial,
  case, evidence and label reads, held-out confirmation, trial requeue and
  the prompt playground, which the web app and the HTTP API provide;
- prompt-level batch tests: Agenomic Cloud refuses `prompt_variables` cases
  at launch, and the runner runs agent and node trials only;
- trial artifacts: results always carry `"artifacts": []`;
- the runner does not enforce `limits.max_output_bytes`;
- runners for other frameworks or languages: the runner serves LangGraph
  agents in Python only.
