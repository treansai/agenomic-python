# RMP benchmarks and the agent bridge

Agenomic Cloud can run five external benchmark suites against your agent from
an RMP session: τ²-bench, ToolSandbox, AppWorld, AgentDojo and MCP-Universe.
The suites and their native evaluators run inside isolated Agenomic runners;
your agent takes part through a **bridge** that you run in your own runtime.
Nothing here is local-first: every call needs a cloud client (`base_url`), and
`client.rmp.start()` still never launches anything.

## 1. Serve your agent

Implement `AgentTargetBridge`. Each turn you receive the benchmark transcript,
the benchmark's tool schemas and the task instructions; you return your agent's
next message. Tool calls are executed by the benchmark environment, never by
your integrations, and their results arrive on the next turn.

```python
from agenomic import Client
from agenomic.benchmarks import AgentTargetBridge, BridgeCapability, ToolCall, TurnReply, TurnRequest, serve_bridge

class MyBridge(AgentTargetBridge):
    capabilities = [BridgeCapability.MULTI_TURN, BridgeCapability.BENCHMARK_TOOLS]

    def handle_turn(self, turn: TurnRequest) -> TurnReply:
        result = my_agent.step(messages=turn.messages, tools=turn.tools, system=turn.instructions)
        return TurnReply(
            content=result.text,
            tool_calls=[ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in result.tool_calls],
            usage={"input_tokens": result.usage.input, "output_tokens": result.usage.output},
        )

client = Client(api_key="agm_...", base_url="https://cloud.agenomic.io")
serve_bridge(client, MyBridge(), agent="agent://acme/support", release_id="rel_2026_09")
```

Or from the shell:

```bash
agenomic-py benchmark serve --agent agent://acme/support --release rel_2026_09 \
  --bridge my_package.bridge:MyBridge --base-url https://cloud.agenomic.io --api-key agm_...
```

The bridge registers its capabilities (the catalogue's compatibility state is
derived from that registration, which expires two minutes after the last
heartbeat), long-polls pending turns for exactly this agent and release, and
answers them. Replies are idempotent: a retried delivery never double counts.
`--bridge fixture` serves a deterministic fixture that never calls a tool; it
exists for adapter tests and its results are never presented as a customer
evaluation.

## 2. Plan, preflight, launch

```python
session = client.rmp.start(agent="agent://acme/support", release_id="rel_2026_09", environment="staging")

cards = client.benchmarks.catalog(agent="agent://acme/support", release_id="rel_2026_09")
for entry in cards:
    print(entry["card"]["id"], entry["availability"]["state"], entry["agent_compatibility"]["state"])

plan = client.benchmarks.create_plan(
    session["session_id"],
    selections=[
        {
            "benchmark_id": "agentdojo",
            "benchmark_version": "v0.1.35",
            "scope": {"domains": ["workspace"], "injection_families": ["none", "important_instructions"]},
            "profile": "standard",
            "requirement": "required",
            "gates": [{"gate_id": "asr", "metric_id": "agentdojo.targeted_asr", "operator": "lte", "threshold": 0.1, "min_evaluable": 20}],
        },
        {
            "benchmark_id": "tau2-bench",
            "benchmark_version": "v0.2.0",
            "scope": {"domains": ["retail"]},
            "profile": "smoke",
            "requirement": "advisory",
            "auxiliary": {"user_simulator": "dummy_user"},
        },
    ],
)
plan = client.benchmarks.preflight(plan["plan_id"])
for check in plan["preflight"]["checks"]:
    print(check["status"], check["check_id"], check["message"], check.get("corrective_action"))
if plan["preflight"]["passed"]:
    launched = client.benchmarks.launch(plan["plan_id"])
```

Preflight checks permissions, the session, the release, quota, licences, the
bridge, secrets and network, and freezes the exact number of tasks, variants
and trials per benchmark; it never trims the sample to fit a budget. Launch
freezes the manifest (its blake3 hash is `plan["manifest_hash"]`), is
idempotent, and returns without waiting for the runs.

## 3. Follow runs and read results

```python
view = client.benchmarks.get_plan(plan["plan_id"])
print(view["overall_verdict"], view["overall_reason"])
for run in view["runs"]:
    detail = client.benchmarks.get_run(run["run_id"])
    print(run["benchmark_id"], run["status"], run["verdict"], run["counts"], run["normalized_metrics"])
```

Lifecycle (`queued`, `running`, `completed`, `failed`, `cancelled`) and
verdict (`pass`, `fail`, `inconclusive`, `not_applicable`) are separate. Native
scores stay verbatim in `native_summary`; `normalized_metrics` never turns a
missing value into zero; smoke profiles are never conclusive; a required
benchmark that is incomplete blocks a favourable verdict. `compare(plan,
baseline)` pairs tasks present in both plans and says when they are not
comparable.

## 4. Protect

Findings can become policy proposals bound to the plan's manifest:
`propose_policy({...})`, then `decide_policy(id, "reviewed")`, `"shadow"`,
`"approved"` and `"active"` (approval and activation must cite the exact
`manifest_hash`), `"rejected"` or `"rolled_back"`. An active
`deny_tools` policy is enforced by the relay before a tool call reaches the
benchmark environment; `block_promotion_on_verdict` refuses release promotion
while the latest launched plan has not passed.
