from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agenomic.prompts.digest import canonical_json_v1

__all__ = [
    "ASSIGNMENT_SCHEMA",
    "CASE_SCHEMA",
    "HELLO_SCHEMA",
    "RESULT_SCHEMA",
    "SPEC_SCHEMA",
    "VIEW_SCHEMA",
    "ExperimentCase",
    "RunnerEvaluatorView",
    "RunnerView",
    "TrialAssignment",
    "ViewArm",
    "ViewEntryPoint",
    "ViewIsolation",
    "ViewLimits",
    "ViewTools",
    "spec_digest",
]

SPEC_SCHEMA = "agenomic.experiment_spec/v1"
CASE_SCHEMA = "agenomic.experiment_case/v1"
HELLO_SCHEMA = "agenomic.experiment_runner_hello/v1"
ASSIGNMENT_SCHEMA = "agenomic.experiment_trial_assignment/v1"
VIEW_SCHEMA = "agenomic.experiment_runner_view/v1"
RESULT_SCHEMA = "agenomic.experiment_trial_result/v1"

ToolMode = Literal["none", "mock", "recorded", "live"]


def spec_digest(spec: Mapping[str, Any]) -> str:
    body = {key: value for key, value in spec.items() if key != "identity"}
    return "sha256:" + hashlib.sha256(canonical_json_v1(body).encode("utf-8")).hexdigest()


class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)


class ExperimentCase(_View):
    schema_: Optional[Literal["agenomic.experiment_case/v1"]] = Field(default=None, alias="schema")
    case_id: str
    kind: Literal["agent_input", "node_state", "prompt_variables"]
    input: Any = None
    expected: Any = None
    initial_state: Optional[dict[str, Any]] = None
    turns: list[dict[str, Any]] = []
    tags: list[str] = []

    @property
    def provenance(self) -> Optional[dict[str, Any]]:
        if isinstance(self.input, Mapping):
            value = self.input.get("provenance")
            if isinstance(value, Mapping):
                return dict(value)
        return None

    @property
    def context(self) -> dict[str, Any]:
        if isinstance(self.input, Mapping) and isinstance(self.input.get("context"), Mapping):
            return dict(self.input["context"])
        return {}

    @property
    def is_snapshot(self) -> bool:
        provenance = self.provenance
        return provenance is not None and provenance.get("source") == "production_snapshot"


class ViewArm(_View):
    arm_key: str
    release_id: str
    genome_version: Optional[str] = None
    prompt_manifest_digest: str
    runtime_digest: str


class ViewIsolation(_View):
    store_namespace: list[str]
    resource_suffix: str


class ViewTools(_View):
    mode: ToolMode = "none"
    tool_repetition: Optional[int] = None
    declared_tools: list[str] = []
    on_fixture_miss: Literal["stop_trial", "tool_error"] = "stop_trial"


class ViewLimits(_View):
    max_tokens_per_trial: Optional[int] = None
    max_model_calls_per_trial: Optional[int] = None
    max_output_bytes: Optional[int] = None


class ViewEntryPoint(_View):
    name: str
    kind: Literal["graph_node", "callable"]
    node_path: str
    seed_as_node: Optional[str] = None
    slot_paths: list[str] = []


class RunnerEvaluatorView(_View):
    evaluator_id: str
    kind: str
    version: int
    config: dict[str, Any] = {}
    prompt: Optional[dict[str, Any]] = None


class RunnerView(_View):
    schema_: Literal["agenomic.experiment_runner_view/v1"] = Field(alias="schema")
    experiment_id: str
    trial_id: str
    attempt: int
    level: Literal["agent", "node"]
    case: ExperimentCase
    arm: ViewArm
    binding: dict[str, Any]
    prompts: dict[str, Any]
    isolation: ViewIsolation
    seed: str
    tools: ViewTools = ViewTools()
    protect_overlay: Optional[dict[str, Any]] = None
    limits: ViewLimits = ViewLimits()
    runner_evaluators: list[RunnerEvaluatorView] = []
    secret_refs: list[str] = []
    entry_point: Optional[ViewEntryPoint] = None
    node_scope: Any = None


class TrialAssignment(_View):
    schema_: Literal["agenomic.experiment_trial_assignment/v1"] = Field(alias="schema")
    trial_id: str
    attempt: int
    lease_token: str
    lease_until: str
    deadline_at: str
    heartbeat_interval_seconds: Union[int, float] = 30
    runner_view_digest: str
    view: RunnerView

    def __repr__(self) -> str:
        return f"TrialAssignment(trial_id={self.trial_id!r}, attempt={self.attempt})"

    __str__ = __repr__
