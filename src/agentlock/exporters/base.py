"""Base interface for trace envelope exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentlock.types.envelope import TraceEnvelope


class Exporter(ABC):
    """Sink for :class:`TraceEnvelope` instances.

    Implementations may be sync or async — the decorator drives both.
    Use as a context manager to ensure :meth:`close` runs on exit.

    Example:
        >>> class _Mem(Exporter):
        ...     def __init__(self) -> None: self.envs: list[TraceEnvelope] = []
        ...     def export(self, envelope: TraceEnvelope) -> None:
        ...         self.envs.append(envelope)
    """

    @abstractmethod
    def export(self, envelope: TraceEnvelope) -> Any:
        """Send `envelope` to the underlying sink. May be async."""

    def close(self) -> None:  # noqa: B027
        """Flush + release resources. Default is a no-op."""

    def __enter__(self) -> Exporter:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()
