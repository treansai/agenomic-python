"""Agents Vault with LangGraph: the graph carries references, the executor holds the credential.

Optional commercial module of Agenomic Cloud/Enterprise (Agents Vault add-on).
Requires `pip install agenomic[langgraph]`.

The agent holds an authorization, never a credential. The graph state, and so
every checkpoint, carries only opaque references: a binding name, an
``action_id`` and a ``receipt_id``. The ``execute`` node calls
``client.tools.execute``; the ``summarize`` node re-reads the stored business
result by ``action_id`` instead of keeping it in the state.

Offline by default: executions are answered from replay fixtures and nothing
reaches the network. To run against a real workspace set

    AGENOMIC_BASE_URL=https://cloud.example
    AGENOMIC_VAULT_RUNTIME_TOKEN=vrt_...        (enrolled with client.vault.runtime_identities.issue)
    AGENOMIC_VAULT_BINDING=crm-read              (an active binding for the tool crm.get_customer)
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any, Optional, TypedDict

from agenomic import Client
from agenomic.vault import (
    ReplayFixture,
    ReplayOutcome,
    VaultApprovalRequired,
    VaultNotEntitled,
    VaultOutcomeUnknown,
    VaultPolicyDenied,
    VaultReplay,
)

TOOL = "crm.get_customer"
REFERENCE_KEYS = {"binding", "customer_id", "action_id", "receipt_id", "approval_id", "status"}


class VaultState(TypedDict, total=False):
    """Opaque references only: no secret, no credential, no business payload."""

    binding: str
    customer_id: str
    action_id: str
    receipt_id: Optional[str]
    approval_id: Optional[str]
    status: str


def plan(state: VaultState) -> VaultState:
    """Fix the logical action once: a resumed graph reuses the checkpointed ``action_id``."""
    return {"action_id": state.get("action_id") or str(uuid.uuid4()), "status": "planned"}


def make_execute(client: Client) -> Callable[[VaultState], VaultState]:
    def execute(state: VaultState) -> VaultState:
        try:
            out = client.tools.execute(
                tool=TOOL,
                binding=state["binding"],
                arguments={"id": state["customer_id"]},
                action_id=state["action_id"],
            )
        except VaultApprovalRequired as pending:
            return {"status": "approval_pending", "approval_id": pending.approval_id}
        except VaultOutcomeUnknown:
            return {"status": "outcome_unknown"}
        except VaultPolicyDenied:
            return {"status": "denied"}
        except VaultNotEntitled as locked:
            return {"status": "locked_upgrade" if locked.upgrade_hint else "locked"}
        return {"status": "succeeded", "receipt_id": out.receipt_id}

    return execute


def make_summarize(
    client: Client, on_result: Callable[[Any], None]
) -> Callable[[VaultState], VaultState]:
    def summarize(state: VaultState) -> VaultState:
        stored = client.tools.get_execution(state["action_id"])
        on_result(stored.result)
        return {"status": "done"}

    return summarize


def after_execute(state: VaultState) -> str:
    return "summarize" if state.get("status") == "succeeded" else "end"


def build_graph(client: Client, on_result: Callable[[Any], None], checkpointer: Any = None) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(VaultState)
    graph.add_node("plan", plan)
    graph.add_node("execute", make_execute(client))
    graph.add_node("summarize", make_summarize(client, on_result))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", after_execute, {"summarize": "summarize", "end": END})
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)


def offline_client() -> tuple[Client, VaultReplay]:
    """A client whose vault executions come from fixtures: a missing one is an error."""
    replay = VaultReplay(
        [
            _fixture("c_1", ReplayOutcome(result={"name": "Ada Lovelace"}, receipt_id="rcpt-1")),
            _fixture("c_2", ReplayOutcome(kind="approval_required", approval_id="apr-1")),
            _fixture("c_3", ReplayOutcome(kind="outcome_unknown", status_code=504)),
            _fixture("c_4", ReplayOutcome(kind="denied", reason_codes=["no_policy_bound"])),
        ]
    )
    return Client(vault_replay=replay), replay


def _fixture(customer_id: str, outcome: ReplayOutcome) -> ReplayFixture:
    return ReplayFixture(
        fixture_id=f"fx-{customer_id}",
        tool=TOOL,
        binding="crm-read",
        arguments={"id": customer_id},
        outcome=outcome,
    )


def run(client: Client, binding: str, customer_ids: list[str]) -> dict[str, str]:
    from langgraph.checkpoint.memory import InMemorySaver

    results: list[Any] = []
    saver = InMemorySaver()
    graph = build_graph(client, results.append, saver)
    outcomes: dict[str, str] = {}
    for customer_id in customer_ids:
        config = {"configurable": {"thread_id": f"crm-{customer_id}"}}
        state = graph.invoke({"binding": binding, "customer_id": customer_id}, config)
        assert set(state) <= REFERENCE_KEYS, "the state must carry references only"
        outcomes[customer_id] = state["status"]
        print(customer_id, {k: v for k, v in state.items() if k != "binding"})
    print("business results read in-process:", results)
    return outcomes


def main() -> int:
    try:
        import langgraph  # noqa: F401
    except ImportError as error:
        print(f"langgraph not installed: {error}")
        return 0
    base_url = os.environ.get("AGENOMIC_BASE_URL")
    token = os.environ.get("AGENOMIC_VAULT_RUNTIME_TOKEN")
    if base_url and token:
        binding = os.environ.get("AGENOMIC_VAULT_BINDING", "crm-read")
        run(Client(base_url=base_url, runtime_token=token), binding, ["c_1"])
        return 0
    client, replay = offline_client()
    outcomes = run(client, "crm-read", ["c_1", "c_2", "c_3", "c_4"])
    print("outcomes:", outcomes)
    print("served from fixtures:", [(c.fixture_id, c.action_id[:8]) for c in replay.calls])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
