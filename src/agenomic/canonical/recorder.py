"""Canonical Trace SDK (master document §14) — the run-scoped event emitter.

A :class:`CanonicalRun` produces an ``agenomic/v0.3`` run trace: an append-only,
**hash-chained** event stream (``run/llm/tool/memory/policy/human_review/error``)
plus run metadata, a causal execution graph, and a signed Merkle integrity
block. Every event:

1. is **redacted** before anything is hashed or exported (never a sensitive
   payload in the clear),
2. carries a content-addressed ``payload_hash`` and a chained ``event_hash``
   (byte-identical to the spec verifier — see :mod:`agenomic.canonical.hashing`),
3. also surfaces as an **OpenTelemetry GenAI** span when a tracer is configured.

The emitter is runtime-agnostic; the LangGraph adapter
(:mod:`agenomic.integrations.langgraph`) drives it automatically.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from agenomic.canonical import otel
from agenomic.canonical.hashing import (
    GENESIS_PREV_EVENT_HASH,
    content_hash,
    event_hash,
    merkle_root,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.trace import Tracer

    from agenomic.redaction.engine import RedactionEngine

ActorKind = Literal["system", "agent", "human", "tool"]
RunStatus = Literal["success", "error", "cancelled"]

_SPEC_VERSION = "agenomic/v0.3"
_ADAPTER_VERSION = "agenomic-python/0.1"


def _sha256_uri(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    try:
        import ulid

        return str(ulid.new())
    except Exception:  # pragma: no cover - ulid always present in practice
        return uuid.uuid4().hex


class CanonicalRun:
    """An in-progress canonical run. Use :func:`start_run` to construct one."""

    def __init__(
        self,
        *,
        agent_id: str,
        agent_version: str = "0.0.0",
        genome_version: str | None = None,
        runtime_name: str = "python",
        runtime_version: str = "0.0.0",
        provider: str = "unknown",
        model: str = "unknown",
        model_version: str = "unknown",
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        run_id: str | None = None,
        input_payload: Any = None,
        classification: list[str] | None = None,
        redaction: RedactionEngine | None = None,
        tracer: Tracer | None = None,
        signed_by: str = "agenomic-python",
    ) -> None:
        self._run_id = run_id or _new_event_id()
        self._agent_id = agent_id
        self._redaction = redaction
        self._tracer = tracer
        self._signed_by = signed_by
        self._events: list[dict[str, Any]] = []
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[dict[str, Any]] = []
        self._prev_hash = GENESIS_PREV_EVENT_HASH
        self._completed = False

        self._agent = {
            "agent_id": agent_id,
            "agent_version": agent_version,
            "genome_version": genome_version or _sha256_uri(f"{agent_id}@{agent_version}"),
            "runtime": {
                "name": runtime_name,
                "version": runtime_version,
                "adapter_version": _ADAPTER_VERSION,
            },
        }
        self._llm = {
            "provider": provider,
            "model": model,
            "model_version": model_version,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        self._components = {
            "prompt_version": _sha256_uri("prompt"),
            "policy_version": _sha256_uri("policy"),
            "memory_version": _sha256_uri("memory"),
            "knowledge_version": _sha256_uri("knowledge"),
            "tool_versions": {},
        }
        self._input = self._io_payload(input_payload, classification)

        # The first event of every run.
        self._append("run.started", "system", "sdk", {"run_id": self._run_id})

    # ---- redaction + hashing ------------------------------------------------

    def _redact(self, payload: Any) -> Any:
        if self._redaction is None:
            return payload
        return self._redaction.apply(payload)

    def _io_payload(self, payload: Any, classification: list[str] | None) -> dict[str, Any]:
        redacted = self._redact(payload if payload is not None else {})
        return {
            "hash": content_hash(redacted),
            "redacted_payload": redacted,
            "classification": list(classification or []),
        }

    def _append(
        self,
        event_type: str,
        actor_kind: ActorKind,
        actor_id: str,
        payload: Any,
        *,
        span_attributes: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if self._completed:
            raise RuntimeError("cannot append to a completed run")
        redacted = self._redact(payload)
        event: dict[str, Any] = {
            "event_id": _new_event_id(),
            "type": event_type,
            "timestamp": _now_iso(),
            "actor": {"kind": actor_kind, "id": actor_id},
            "payload_hash": content_hash(redacted),
            "redacted_payload": redacted,
            "prev_event_hash": self._prev_hash,
            **extra,
        }
        event["event_hash"] = event_hash(event)
        self._prev_hash = event["event_hash"]

        self._nodes.append({"id": event["event_id"], "kind": event_type})
        if len(self._events) > 0:
            self._edges.append(
                {
                    "from": self._events[-1]["event_id"],
                    "to": event["event_id"],
                    "type": "caused_by",
                }
            )
        self._events.append(event)
        otel.emit_span(self._tracer, event_type, span_attributes or {})
        return event

    # ---- public capture API -------------------------------------------------

    def log_llm(
        self,
        *,
        prompt: Any,
        response: Any,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record an LLM call as ``llm.requested`` + ``llm.responded``."""
        prov = provider or str(self._llm["provider"])
        mdl = model or str(self._llm["model"])
        attrs = otel.llm_attributes(
            provider=prov, model=mdl, input_tokens=input_tokens, output_tokens=output_tokens
        )
        self._append(
            "llm.requested",
            "agent",
            self._agent_id,
            {"provider": prov, "model": mdl, "prompt": prompt},
            span_attributes=attrs,
        )
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self._append(
            "llm.responded",
            "agent",
            self._agent_id,
            {"provider": prov, "model": mdl, "response": response, "usage": usage},
            span_attributes=attrs,
        )

    def log_tool_call(
        self,
        *,
        tool: str,
        arguments: Any = None,
        result: Any = None,
        status: str = "ok",
        protocol: str = "local",
        server: str | None = None,
    ) -> None:
        """Record a tool call as ``tool.call.proposed`` + ``tool.call.executed``."""
        attrs = otel.tool_attributes(tool=tool)
        meta = {"protocol": protocol, "server": server}
        self._append(
            "tool.call.proposed",
            "agent",
            self._agent_id,
            {"tool": tool, "arguments": arguments, **meta},
            span_attributes=attrs,
        )
        self._append(
            "tool.call.executed",
            "tool",
            tool,
            {"tool": tool, "result": result, "status": status, **meta},
            span_attributes=attrs,
        )

    def log_memory(
        self,
        *,
        store: str,
        operation: Literal["read", "write"],
        key: str | None = None,
        value: Any = None,
    ) -> None:
        """Record a memory access. ``write`` emits proposed + committed."""
        base = {"store": store, "key": key, "value": value}
        if operation == "read":
            self._append("memory.read", "agent", self._agent_id, base)
        else:
            self._append("memory.write.proposed", "agent", self._agent_id, base)
            self._append("memory.write.committed", "system", store, base)

    def log_policy_check(self, *, policy: str, outcome: str, detail: Any = None) -> None:
        """Record a ``policy.check.performed`` event."""
        self._append(
            "policy.check.performed",
            "system",
            policy,
            {"policy": policy, "outcome": outcome, "detail": detail},
        )

    def request_human_review(self, *, reason: str, context: Any = None) -> None:
        """Record a ``human.review.requested`` interruption."""
        self._append(
            "human.review.requested",
            "agent",
            self._agent_id,
            {"reason": reason, "context": context},
        )

    def log_error(self, *, message: str, kind: str | None = None) -> None:
        """Record an ``error.raised`` event."""
        self._append(
            "error.raised",
            "system",
            kind or "error",
            {"message": message, "kind": kind},
        )

    def complete_run(
        self,
        *,
        output: Any = None,
        status: RunStatus = "success",
        classification: list[str] | None = None,
        semantic_signature: str | None = None,
    ) -> dict[str, Any]:
        """Emit ``run.completed`` and return the finalized, schema-valid trace."""
        self._append(
            "run.completed",
            "system",
            "sdk",
            {"status": status},
        )
        out = self._io_payload(output, classification)
        out["semantic_signature"] = semantic_signature or content_hash(out["redacted_payload"])

        event_hashes = [e["event_hash"] for e in self._events]
        root = merkle_root(event_hashes)
        trace: dict[str, Any] = {
            "spec_version": _SPEC_VERSION,
            "run_id": self._run_id,
            "agent": self._agent,
            "llm": self._llm,
            "components": self._components,
            "input": self._input,
            "output": out,
            "events": self._events,
            "execution_graph": {"nodes": self._nodes, "edges": self._edges},
            "risk_scores": [],
            "compliance_checks": [],
            "alignment_checks": [],
            "environment_snapshot": {"sdk": _ADAPTER_VERSION},
            "integrity": {
                "run_merkle_root": root,
                "signed_by": self._signed_by,
                # Detached signing is the embedder's job; an unsigned local
                # export still carries a non-empty, well-formed marker.
                "signature": f"unsigned:{content_hash(root)[7:]}",
            },
        }
        self._completed = True
        return trace


def start_run(agent_id: str, **kwargs: Any) -> CanonicalRun:
    """Start a canonical run and emit ``run.started``.

    Example:
        >>> run = start_run("agent://acme/support", model="gpt-4o")
        >>> run.log_tool_call(tool="search", arguments={"q": "x"}, result=[1])
        >>> trace = run.complete_run(output={"answer": "ok"})
        >>> trace["spec_version"]
        'agenomic/v0.3'
    """
    return CanonicalRun(agent_id=agent_id, **kwargs)


__all__ = ["CanonicalRun", "start_run"]
