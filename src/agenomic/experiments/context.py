from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from agenomic.experiments.models import ExperimentCase, RunnerView
from agenomic.experiments.secrets import SecretValues

if TYPE_CHECKING:
    from agenomic.experiments.tools import ToolProxy

__all__ = ["TrialContext", "TrialState"]


@dataclass
class TrialState:
    proxy: Optional[ToolProxy]
    lease_token: str
    literals: list[str]
    terminal: Optional[BaseException] = None
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    model_call_count: int = 0
    tokens_used: int = 0
    live_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_terminal(self, error: BaseException) -> None:
        with self.lock:
            if self.terminal is None:
                self.terminal = error


class TrialContext:
    __slots__ = (
        "_state",
        "agent_id",
        "arm_key",
        "attempt",
        "case",
        "checkpointer",
        "experiment_id",
        "level",
        "resource_suffix",
        "secrets",
        "seed",
        "store",
        "store_namespace",
        "thread_key",
        "tool_mode",
        "trial_id",
        "view",
    )

    def __init__(
        self,
        *,
        view: RunnerView,
        agent_id: str,
        checkpointer: Any,
        store: Any,
        secrets: SecretValues,
        state: TrialState,
    ) -> None:
        self.view = view
        self.agent_id = agent_id
        self.experiment_id = view.experiment_id
        self.trial_id = view.trial_id
        self.attempt = view.attempt
        self.arm_key = view.arm.arm_key
        self.thread_key = str(view.binding["thread_key"])
        self.case: ExperimentCase = view.case
        self.level: Literal["agent", "node"] = view.level
        self.seed = int(view.seed)
        self.resource_suffix = view.isolation.resource_suffix
        self.store_namespace: tuple[str, ...] = tuple(view.isolation.store_namespace)
        self.tool_mode = view.tools.mode
        self.checkpointer = checkpointer
        self.store = store
        self.secrets = secrets
        self._state = state

    def __repr__(self) -> str:
        return (
            f"TrialContext(trial_id={self.trial_id!r}, attempt={self.attempt}, "
            f"arm_key={self.arm_key!r})"
        )

    def wrap_tools(self, tools: Sequence[Any]) -> list[Any]:
        from agenomic.experiments.tools import wrap_tools

        return wrap_tools(self, tools)
