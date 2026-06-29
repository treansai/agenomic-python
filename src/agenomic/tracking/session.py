"""Online tracking instrumentation.

``client.tracking.start(...)`` opens a :class:`TrackingSession` for a production
agent release. Events emitted on the session are streamed to Agenomic Cloud
(when the client has a ``base_url``) or buffered locally otherwise. The wire
format is the spec's snake_case ``tracking-event`` shape — a production-time
projection of the canonical trace / ATEP event model — so the cloud and CLI
engines (drift / loop / intent / harness detection) consume it unchanged.

Detection runs server-side or in the CLI; the SDK's job is faithful,
redaction-safe instrumentation.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import ulid

from agenomic.exceptions import CloudError

SPEC_VERSION = "agenomic/v0.3"

#: The runtime event vocabulary (a superset of the v0.3 event-type registry).
TRACKING_EVENT_TYPES = frozenset(
    {
        "agent.started",
        "agent.step.started",
        "agent.step.completed",
        "model.call.started",
        "model.call.completed",
        "tool.call.started",
        "tool.call.completed",
        "memory.read",
        "memory.write",
        "policy.evaluated",
        "intent.detected",
        "loop.detected",
        "drift.detected",
        "harness.violation",
        "alert.created",
        "agent.completed",
        "agent.failed",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrackingSession:
    """A live online-tracking session.

    In cloud mode each emit is POSTed; in local mode events are buffered and can
    be exported as JSONL for ``agenomic track report`` to analyze offline.
    """

    def __init__(self, client: Any, session_id: str, agent_id: str, environment: str) -> None:
        self._client = client
        self.session_id = session_id
        self.agent_id = agent_id
        self.environment = environment
        self._seq = 0
        self._events: list[dict[str, Any]] = []
        self._stopped = False

    @property
    def cloud(self) -> bool:
        return bool(getattr(self._client, "is_cloud", False))

    @property
    def events(self) -> list[dict[str, Any]]:
        """Events buffered in local mode (empty in cloud mode)."""
        return list(self._events)

    def _emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        if self._stopped:
            raise RuntimeError("tracking session already stopped")
        if event_type not in TRACKING_EVENT_TYPES:
            raise ValueError(f"unknown tracking event type: {event_type!r}")
        event: dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "event_id": ulid.new().str,
            "session_id": self.session_id,
            "timestamp": _now_iso(),
            "sequence_number": self._seq,
            "type": event_type,
            "agent_id": self.agent_id,
        }
        self._seq += 1
        for key, value in fields.items():
            if value is not None:
                event[key] = value
        if self.cloud:
            self._client._post(f"/v1/tracking/sessions/{self.session_id}/events", event)
        else:
            self._events.append(event)
        return event

    def event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Emit one runtime event with arbitrary snake_case wire fields."""
        return self._emit(event_type, **fields)

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[TrackingSession]:
        """Run a block as a workflow step.

        Emits ``agent.step.started`` on entry and ``agent.step.completed`` on
        success, or ``agent.failed`` if the block raises (then re-raises).
        """
        self._emit("agent.step.started", workflow_step_id=name)
        try:
            yield self
        except Exception:
            self._emit("agent.failed", workflow_step_id=name, metadata={"status": "error"})
            raise
        else:
            self._emit("agent.step.completed", workflow_step_id=name)

    def model_call(
        self,
        provider: str,
        model: str,
        *,
        temperature: Optional[float] = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        model_meta: dict[str, Any] = {"provider": provider, "model": model}
        if temperature is not None:
            model_meta["temperature"] = temperature
        return self._emit(
            "model.call.completed",
            model=model_meta,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def tool_call(
        self,
        tool: str,
        *,
        protocol: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        tool_meta: dict[str, Any] = {"name": tool}
        if protocol:
            tool_meta["protocol"] = protocol
        if permissions:
            tool_meta["permissions"] = permissions
        return self._emit(
            "tool.call.completed",
            tool=tool_meta,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def intent(self, value: str) -> dict[str, Any]:
        return self._emit("intent.detected", intent=value)

    def memory_write(
        self,
        *,
        schema_version: Optional[str] = None,
        output_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        metadata = {"schema_version": schema_version} if schema_version else None
        return self._emit("memory.write", output_hash=output_hash, metadata=metadata)

    def stop(self) -> None:
        """Finalize the session. Idempotent."""
        if self._stopped:
            return
        # Mark stopped only after a successful stop so a failed cloud POST stays
        # retryable and the remote session isn't orphaned.
        if self.cloud:
            self._client._post(f"/v1/tracking/sessions/{self.session_id}/stop", {})
        self._stopped = True

    def report(self) -> dict[str, Any]:
        """Fetch the tracking report (cloud mode only)."""
        if not self.cloud:
            raise RuntimeError(
                "report() requires cloud mode; in local mode export with to_jsonl() "
                "and run `agenomic track report`"
            )
        report: dict[str, Any] = self._client._get(
            f"/v1/tracking/sessions/{self.session_id}/report"
        )
        return report

    def to_jsonl(self) -> str:
        """Serialize buffered local events as JSONL (one event per line)."""
        return "".join(json.dumps(event) + "\n" for event in self._events)

    def __enter__(self) -> TrackingSession:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


class TrackingResource:
    """The ``client.tracking`` namespace."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def start(
        self,
        *,
        agent: str,
        release_id: Optional[str] = None,
        bundle_id: Optional[str] = None,
        genome_hash: Optional[str] = None,
        environment: str = "production",
        tracking_config: Optional[dict[str, Any]] = None,
    ) -> TrackingSession:
        """Start a new online-tracking session."""
        if getattr(self._client, "is_cloud", False):
            body: dict[str, Any] = {
                "spec_version": SPEC_VERSION,
                "agent_id": agent,
                "environment": environment,
            }
            if release_id:
                body["release_id"] = release_id
            if bundle_id:
                body["bundle_id"] = bundle_id
            if genome_hash:
                body["genome_hash"] = genome_hash
            if tracking_config:
                body["tracking_config"] = tracking_config
            response = self._client._post("/v1/tracking/sessions", body)
            session = response.get("session", response) if isinstance(response, dict) else {}
            session_id = session.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise CloudError("tracking start response did not include a session_id")
        else:
            session_id = ulid.new().str
        return TrackingSession(self._client, session_id, agent, environment)
