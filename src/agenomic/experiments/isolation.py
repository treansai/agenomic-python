from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage
from langgraph.store.base import (
    BaseStore,
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from pydantic import BaseModel

from agenomic.experiments.errors import IsolationViolation, RunnerConfigurationError

__all__ = [
    "NOT_PRESERVED",
    "NamespacedStore",
    "check_graph",
    "isolation_record",
    "jsonable",
    "revive_state",
    "seed_store",
    "serialize_state",
]

NOT_PRESERVED = (
    "pending_writes",
    "channel_versions",
    "checkpoint_history",
    "subgraph_checkpoints",
    "store_contents",
    "external_side_effects",
)
_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


class NamespacedStore(BaseStore):
    def __init__(self, inner: BaseStore, prefix: Sequence[str]) -> None:
        if isinstance(inner, NamespacedStore):
            raise ValueError("the store is already namespaced")
        if not prefix or not all(isinstance(part, str) and part for part in prefix):
            raise ValueError("prefix is a non-empty sequence of non-empty strings")
        self._inner = inner
        self._prefix = tuple(prefix)
        self.supports_ttl = inner.supports_ttl
        self.ttl_config = inner.ttl_config

    @property
    def prefix(self) -> tuple[str, ...]:
        return self._prefix

    def _inside(self, namespace: Sequence[str]) -> bool:
        return tuple(namespace[: len(self._prefix)]) == self._prefix

    def _check(self, ops: list[Op]) -> None:
        for op in ops:
            if isinstance(op, (GetOp, PutOp)):
                allowed = self._inside(op.namespace)
            elif isinstance(op, SearchOp):
                allowed = self._inside(op.namespace_prefix)
            elif isinstance(op, ListNamespacesOp):
                allowed = any(
                    condition.match_type == "prefix" and self._inside(condition.path)
                    for condition in op.match_conditions or ()
                )
            else:
                allowed = False
            if not allowed:
                raise IsolationViolation(
                    "a store operation left the trial namespace", operation=type(op).__name__
                )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        checked = list(ops)
        self._check(checked)
        return self._inner.batch(checked)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        checked = list(ops)
        self._check(checked)
        return await self._inner.abatch(checked)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, BaseMessage):
        return {"role": _ROLES.get(value.type, value.type), "content": jsonable(value.content)}
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json"))
    raise RunnerConfigurationError(
        "output_not_serializable", f"a {type(value).__name__} value is not a JSON value"
    )


def _unserializable(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("lc") == 1 and value.get("type") == "not_implemented":
            return True
        return any(_unserializable(item) for item in value.values())
    if isinstance(value, list):
        return any(_unserializable(item) for item in value)
    return False


def serialize_state(values: Any) -> Any:
    serialized = dumpd(values)
    if _unserializable(serialized):
        raise RunnerConfigurationError(
            "state_not_serializable", "a channel value cannot be serialized"
        )
    return serialized


def revive_state(values: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return load(values, allowed_objects="messages")
        except (ValueError, TypeError, KeyError, NotImplementedError) as error:
            raise RunnerConfigurationError(
                "initial_state_invalid", "the case initial_state cannot be revived"
            ) from error


def seed_store(store: Any, namespace: Sequence[str], seeds: Any) -> int:
    if seeds is None:
        return 0
    if not isinstance(seeds, list):
        raise RunnerConfigurationError("store_seed_invalid", "store_seed must be a list")
    count = 0
    for item in seeds:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("key"), str)
            or not isinstance(item.get("value"), Mapping)
            or not isinstance(item.get("namespace", []), list)
        ):
            raise RunnerConfigurationError(
                "store_seed_invalid", "store_seed items are {namespace, key, value}"
            )
        store.put((*namespace, *item.get("namespace", [])), item["key"], dict(item["value"]))
        count += 1
    return count


def check_graph(graph: Any, checkpointer: Any, store: Any) -> None:
    from agenomic.integrations.langgraph_binding import ManagedGraph

    if isinstance(graph, ManagedGraph):
        raise RunnerConfigurationError(
            "factory_returned_bound_graph", "the factory must return the unbound compiled graph"
        )
    if getattr(graph, "checkpointer", None) is not checkpointer:
        raise IsolationViolation("the factory must compile the graph with ctx.checkpointer")
    graph_store = getattr(graph, "store", None)
    if graph_store is not None and graph_store is not store:
        raise IsolationViolation("the factory must compile the graph with ctx.store")


def isolation_record(*, checkpointer: str, store: str, fork_fidelity: str) -> dict[str, Any]:
    return {
        "fresh_thread": True,
        "checkpointer": checkpointer,
        "store": store,
        "tool_routing": "sdk_wrapped_tools",
        "fork_fidelity": fork_fidelity,
        "not_preserved": list(NOT_PRESERVED),
    }
