from importlib import import_module
from typing import Any

from agenomic.experiments.errors import (
    ERROR_RULES,
    ErrorClass,
    InfrastructureError,
    IsolationViolation,
    LeaseLost,
    RecordedFixtureAmbiguous,
    RecordedFixtureMiss,
    RunnerConfigurationError,
    SecretResolutionError,
    TrialBudgetExceeded,
    classify,
)
from agenomic.experiments.models import (
    ExperimentCase,
    RunnerView,
    TrialAssignment,
    spec_digest,
)
from agenomic.experiments.resources import ExperimentsResource
from agenomic.experiments.secrets import EnvSecretResolver, SecretResolver, SecretValues

__all__ = [
    "ERROR_RULES",
    "EnvSecretResolver",
    "ErrorClass",
    "ExperimentCase",
    "ExperimentsResource",
    "InfrastructureError",
    "IsolationViolation",
    "LeaseLost",
    "RecordedFixtureAmbiguous",
    "RecordedFixtureMiss",
    "RunnerConfigurationError",
    "RunnerView",
    "SecretResolutionError",
    "SecretResolver",
    "SecretValues",
    "TrialAssignment",
    "TrialBudgetExceeded",
    "classify",
    "spec_digest",
]

_LAZY = {
    "CallableEntryPoint": "agenomic.experiments.runner",
    "ExperimentRunner": "agenomic.experiments.runner",
    "GraphNodeEntryPoint": "agenomic.experiments.runner",
    "GraphTarget": "agenomic.experiments.runner",
    "RunnerEvaluator": "agenomic.experiments.runner",
    "SnapshotRefusedError": "agenomic.experiments.runner",
    "TrialRun": "agenomic.experiments.runner",
    "local_assignment": "agenomic.experiments.runner",
    "snapshot_case": "agenomic.experiments.runner",
    "TrialContext": "agenomic.experiments.context",
    "NamespacedStore": "agenomic.experiments.isolation",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'agenomic.experiments' has no attribute {name!r}")
    return getattr(import_module(module), name)
