"""Tests for workflow and multi-agent system manifests (spec v0.2, RFC 0009)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agenomic.types import SystemSpec, WorkflowSpec


def minimal_workflow(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "spec_version": "agenomic/v0.2",
        "workflow": {
            "id": "workflow://acme/pipeline",
            "name": "Pipeline",
            "domain": "claims",
            "criticality": "standard",
        },
        "steps": [
            {"id": "answer", "type": "agent", "agent": "agent://acme/bot"},
        ],
    }
    doc.update(overrides)
    return doc


def minimal_system(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "spec_version": "agenomic/v0.2",
        "system": {
            "id": "system://acme/orchestra",
            "name": "Orchestra",
            "domain": "claims",
            "criticality": "standard",
        },
        "agents": [{"role": "solo", "id": "agent://acme/bot"}],
        "orchestration": {"style": "pipeline", "entrypoint": "solo"},
    }
    doc.update(overrides)
    return doc


def test_minimal_workflow_validates() -> None:
    spec = WorkflowSpec.model_validate(minimal_workflow())
    assert spec.workflow.id == "workflow://acme/pipeline"
    assert spec.steps[0].type == "agent"


def test_workflow_full_step_vocabulary() -> None:
    doc = minimal_workflow(
        steps=[
            {"id": "extract", "type": "agent", "agent": "agent://acme/bot"},
            {
                "id": "route",
                "type": "tool",
                "tool": {"name": "rules", "protocol": "local"},
                "depends_on": ["extract"],
            },
            {
                "id": "review",
                "type": "human",
                "depends_on": ["route"],
                "when": "requires_human == true",
                "gate": {"role": "handler", "action": "review", "sla": "48h"},
            },
            {
                "id": "collect",
                "type": "loop",
                "depends_on": ["route"],
                "until": "complete == true",
                "max_iterations": 3,
                "body": [
                    {
                        "id": "await_docs",
                        "type": "wait",
                        "wait_for": {"signals": ["documents_received"], "timeout": "30d"},
                    },
                ],
            },
            {
                "id": "finalize",
                "type": "workflow",
                "uses": "workflows/finalize.yaml",
                "depends_on": ["collect"],
            },
        ],
    )
    spec = WorkflowSpec.model_validate(doc)
    assert [s.id for s in spec.steps] == ["extract", "route", "review", "collect", "finalize"]


def test_workflow_bad_id_rejected() -> None:
    doc = minimal_workflow()
    doc["workflow"]["id"] = "pipeline"
    with pytest.raises(ValidationError, match="workflow://org/name"):
        WorkflowSpec.model_validate(doc)


@pytest.mark.parametrize(
    ("step_type", "missing"),
    [
        ("agent", "agent"),
        ("tool", "tool"),
        ("human", "gate"),
        ("wait", "wait_for"),
        ("workflow", "uses"),
        ("loop", "body"),
    ],
)
def test_workflow_step_type_specific_required_fields(step_type: str, missing: str) -> None:
    doc = minimal_workflow(steps=[{"id": "s", "type": step_type}])
    with pytest.raises(ValidationError, match=missing):
        WorkflowSpec.model_validate(doc)


def test_workflow_duplicate_step_id_rejected() -> None:
    doc = minimal_workflow(
        steps=[
            {"id": "a", "type": "agent", "agent": "agent://acme/bot"},
            {"id": "a", "type": "agent", "agent": "agent://acme/bot"},
        ],
    )
    with pytest.raises(ValidationError, match="duplicate step id"):
        WorkflowSpec.model_validate(doc)


def test_workflow_unknown_dependency_rejected() -> None:
    doc = minimal_workflow(
        steps=[
            {"id": "a", "type": "agent", "agent": "agent://acme/bot", "depends_on": ["ghost"]},
        ],
    )
    with pytest.raises(ValidationError, match="unknown step"):
        WorkflowSpec.model_validate(doc)


def test_workflow_bad_duration_rejected() -> None:
    doc = minimal_workflow(
        steps=[
            {"id": "a", "type": "agent", "agent": "agent://acme/bot", "timeout": "soon"},
        ],
    )
    with pytest.raises(ValidationError, match="timeout"):
        WorkflowSpec.model_validate(doc)


def test_minimal_system_validates() -> None:
    spec = SystemSpec.model_validate(minimal_system())
    assert spec.system.id == "system://acme/orchestra"
    assert spec.agents[0].role == "solo"


def test_system_graph_with_edges_and_end() -> None:
    doc = minimal_system(
        agents=[
            {"role": "intake", "id": "agent://acme/intake"},
            {"role": "triage", "id": "agent://acme/triage"},
        ],
        orchestration={
            "style": "graph",
            "engine": {"kind": "temporal"},
            "entrypoint": "intake",
            "edges": [
                {"from": "intake", "to": "triage"},
                {"from": "triage", "to": "END", "when": "expertise_required == true"},
            ],
        },
    )
    spec = SystemSpec.model_validate(doc)
    assert spec.orchestration.edges[0].from_ == "intake"
    assert spec.orchestration.edges[1].to == "END"


def test_system_duplicate_role_rejected() -> None:
    doc = minimal_system(
        agents=[
            {"role": "solo", "id": "agent://acme/a"},
            {"role": "solo", "id": "agent://acme/b"},
        ],
    )
    with pytest.raises(ValidationError, match="duplicate member role"):
        SystemSpec.model_validate(doc)


def test_system_edge_to_undeclared_role_rejected() -> None:
    doc = minimal_system(
        orchestration={
            "style": "graph",
            "edges": [{"from": "solo", "to": "ghost"}],
        },
    )
    with pytest.raises(ValidationError, match="undeclared role"):
        SystemSpec.model_validate(doc)


def test_system_entrypoint_undeclared_role_rejected() -> None:
    doc = minimal_system(orchestration={"style": "pipeline", "entrypoint": "ghost"})
    with pytest.raises(ValidationError, match="undeclared role"):
        SystemSpec.model_validate(doc)


def test_system_shadowed_allowed_actions() -> None:
    doc = minimal_system(
        agents=[
            {
                "role": "solo",
                "id": "agent://acme/bot",
                "autonomy": {"allowed_actions": ["trigger_payment", "read_fnol"]},
            },
        ],
        forbidden_autonomy=["trigger_payment"],
    )
    spec = SystemSpec.model_validate(doc)
    assert spec.shadowed_allowed_actions() == {"solo": ["trigger_payment"]}


def test_system_round_trips_with_alias() -> None:
    doc = minimal_system(
        orchestration={
            "style": "graph",
            "edges": [{"from": "solo", "to": "END"}],
        },
    )
    spec = SystemSpec.model_validate(doc)
    dumped = spec.model_dump(by_alias=True)
    assert dumped["orchestration"]["edges"][0]["from"] == "solo"
    again = SystemSpec.model_validate(dumped)
    assert again.orchestration.edges[0].from_ == "solo"
