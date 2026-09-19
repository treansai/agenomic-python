"""LangChain / LangGraph live tracking.

``TrackingCallbackHandler`` mirrors every LangChain run (graph nodes, model
calls, tools, retrievers) into an online :class:`~agenomic.tracking.TrackingSession`
with the runtime hierarchy the Agenomic live view renders: one turn per root
run, one span per node/model/tool run with ``span_id``/``parent_span_id``,
timing, token usage and content hashes. Raw prompts, arguments and completions
never leave the process.

This module imports ``langchain_core`` at import time; ``agenomic.integrations``
does not import it. Install with ``pip install agenomic[langgraph]``.

Example:
    >>> from agenomic import Client
    >>> from agenomic.integrations.langchain import TrackingCallbackHandler
    >>> session = Client().tracking.start(agent="agent://acme/support")
    >>> handler = TrackingCallbackHandler(session)
    >>> # graph.ainvoke(state, config={"callbacks": [handler]})  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.tracking.session import TrackingSession

try:
    from langchain_core.callbacks import AsyncCallbackHandler
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "langchain-core not installed. Install with: pip install agenomic[langgraph]"
    ) from e

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

logger = logging.getLogger("agenomic.integrations.langchain")

NODE_TAG_PREFIX = "graph:step:"
TURN_TITLE_MAX_CHARS = 120


def _hash(data: Any) -> str:
    try:
        blob = canonical_cbor(data if isinstance(data, dict) else {"v": data})
    except Exception:
        blob = canonical_cbor({"repr": repr(data)})
    return blake3_hex(blob)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _Dispatcher:
    """One worker thread per session so emits stay ordered and never block the event loop."""

    def __init__(self, session: TrackingSession) -> None:
        self._session = session
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        self._failures = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=f"agenomic-tracking-{session.session_id}", daemon=True
        )
        self._thread.start()

    def submit(self, event_type: str, fields: dict[str, Any]) -> None:
        if self._closed:
            self._drop(event_type, "emitter closed with the session")
            return
        self._queue.put((event_type, fields))

    def _drop(self, event_type: str, reason: object) -> None:
        self._failures += 1
        if self._failures <= 3:
            logger.warning(
                "tracking event %s dropped for session %s: %s",
                event_type,
                self._session.session_id,
                reason,
            )

    def flush(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty (or ``timeout`` elapses).

        Returns ``False`` when the timeout elapsed first, or when any event has
        been dropped for this session, so a caller can tell delivery from mere
        drainage. Drops are permanent, so this stays ``False`` afterwards.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return self._failures == 0

    def close(self, timeout: float | None = None) -> bool:
        """Refuse new events, drain, then stop the worker. ``timeout`` is the total."""
        deadline = None if timeout is None else time.monotonic() + timeout
        self._closed = True
        drained = self.flush(timeout)
        self._queue.put(None)
        self._thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        if self._failures:
            logger.warning(
                "tracking session %s: %d event(s) dropped",
                self._session.session_id,
                self._failures,
            )
        return drained

    @property
    def failures(self) -> int:
        return self._failures

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            event_type, fields = item
            try:
                self._session.event(event_type, **fields)
            # Anything the session raises is the caller's telemetry, not their
            # run: one bad event must never take the worker down with it.
            except Exception as exc:
                self._drop(event_type, exc)
            finally:
                self._queue.task_done()


_dispatchers: dict[str, _Dispatcher] = {}
_dispatchers_lock = threading.Lock()


def _dispatcher_for(session: TrackingSession) -> _Dispatcher:
    with _dispatchers_lock:
        dispatcher = _dispatchers.get(session.session_id)
        if dispatcher is not None:
            return dispatcher
        dispatcher = _Dispatcher(session)
        if session.stopped:
            # No stop() left to run a teardown, so never register a worker the
            # session can no longer reclaim.
            dispatcher.close(0.0)
            return dispatcher
        _dispatchers[session.session_id] = dispatcher

        def _teardown() -> None:
            shutdown(session)

        session.on_stop(_teardown)
        return dispatcher


def flush(session: TrackingSession, timeout: float | None = 5.0) -> bool:
    """Wait until every event queued for ``session`` has been sent.

    ``TrackingSession.stop`` drains and tears the emitter down on its own, so
    call this only to checkpoint mid-session. Returns ``False`` when ``timeout``
    elapsed first or when any event has been dropped; read
    :func:`dropped_events` for the count.

    Example:
        >>> from agenomic import Client
        >>> session = Client().tracking.start(agent="agent://acme/support")
        >>> flush(session)
        True
    """
    with _dispatchers_lock:
        dispatcher = _dispatchers.get(session.session_id)
    return True if dispatcher is None else dispatcher.flush(timeout)


def shutdown(session: TrackingSession, timeout: float | None = 5.0) -> bool:
    """Drain ``session``'s emitter, stop its worker thread and forget it.

    Registered on :meth:`TrackingSession.on_stop` the first time a handler is
    built, so ``session.stop()`` already calls it. Idempotent.

    Example:
        >>> from agenomic import Client
        >>> session = Client().tracking.start(agent="agent://acme/support")
        >>> shutdown(session)
        True
    """
    with _dispatchers_lock:
        dispatcher = _dispatchers.pop(session.session_id, None)
    return True if dispatcher is None else dispatcher.close(timeout)


def dropped_events(session: TrackingSession) -> int:
    """How many events failed to reach the cloud for ``session`` so far.

    Read it before ``stop()``: teardown forgets the emitter, and a session with
    no emitter reports ``0``. Drops are also logged, the first three
    individually and a total at shutdown, so they are never silent.

    Example:
        >>> from agenomic import Client
        >>> session = Client().tracking.start(agent="agent://acme/support")
        >>> dropped_events(session)
        0
    """
    with _dispatchers_lock:
        dispatcher = _dispatchers.get(session.session_id)
    return 0 if dispatcher is None else dispatcher.failures


@dataclass
class _Run:
    span_id: str
    parent_span_id: str | None
    turn_id: str
    started_perf: float
    started_at: str
    emitted: bool
    kind: str
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class TrackingCallbackHandler(AsyncCallbackHandler):
    """Mirror LangChain runs into a live tracking session.

    Pass one instance per request in the runnable config
    (``config={"callbacks": [handler]}``); LangChain propagates it to every
    child run, so subgraphs, nodes, chat models and tools are all observed
    without touching the graph. Emission is queued to a background worker and
    a tracking failure never interrupts the run.

    ``capture_turn_title`` sends the first ``TURN_TITLE_MAX_CHARS`` characters
    of the human message as ``turn_title``. It is off by default because it
    is raw user content.

    Example:
        >>> from agenomic import Client
        >>> session = Client().tracking.start(agent="agent://acme/support")
        >>> handler = TrackingCallbackHandler(session, capture_turn_title=True)
        >>> handler.raise_error
        False
    """

    raise_error = False
    run_inline = False

    def __init__(self, session: TrackingSession, *, capture_turn_title: bool = False) -> None:
        self._session = session
        self._capture_turn_title = capture_turn_title
        self._dispatcher = _dispatcher_for(session)
        self._runs: dict[UUID, _Run] = {}

    def _open(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        kind: str,
        emitted: bool,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> _Run:
        parent = self._runs.get(parent_run_id) if parent_run_id is not None else None
        if parent is None:
            turn_id = str(run_id)
            parent_span_id = None
        else:
            turn_id = parent.turn_id
            parent_span_id = parent.span_id if parent.emitted else parent.parent_span_id
        run = _Run(
            span_id=str(run_id),
            parent_span_id=parent_span_id,
            turn_id=turn_id,
            started_perf=time.perf_counter(),
            started_at=_now_iso(),
            emitted=emitted,
            kind=kind,
            name=name,
            meta=dict(meta or {}),
        )
        self._runs[run_id] = run
        return run

    def _close(self, run_id: UUID) -> _Run | None:
        return self._runs.pop(run_id, None)

    def _span_fields(self, run: _Run) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "span_id": run.span_id,
            "trace_id": run.turn_id,
            "turn_id": run.turn_id,
            "start_time": run.started_at,
            "timestamp": run.started_at,
        }
        if run.parent_span_id is not None:
            fields["parent_span_id"] = run.parent_span_id
        if run.name:
            fields["name"] = run.name
        return fields

    def _end_fields(self, run: _Run) -> dict[str, Any]:
        ended_at = _now_iso()
        fields = self._span_fields(run)
        fields["timestamp"] = ended_at
        fields["end_time"] = ended_at
        fields["duration_ms"] = int((time.perf_counter() - run.started_perf) * 1000)
        return fields

    def _emit(self, event_type: str, fields: dict[str, Any]) -> None:
        self._dispatcher.submit(event_type, fields)

    @staticmethod
    def _error_fields(error: BaseException) -> dict[str, Any]:
        return {"error": {"type": type(error).__name__}, "status": "error"}

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        is_root = parent_run_id is None or parent_run_id not in self._runs
        is_node = any(tag.startswith(NODE_TAG_PREFIX) for tag in tags or [])
        kind = "turn" if is_root else "step" if is_node else "internal"
        run = self._open(run_id, parent_run_id, kind=kind, emitted=kind == "step", name=name)
        if kind == "turn":
            fields: dict[str, Any] = {
                "turn_id": run.turn_id,
                "trace_id": run.turn_id,
                "timestamp": run.started_at,
                "start_time": run.started_at,
            }
            title = self._turn_title(inputs) if self._capture_turn_title else None
            if title:
                fields["turn_title"] = title
            self._emit("turn.started", fields)
        elif kind == "step":
            fields = self._span_fields(run)
            fields["workflow_step_id"] = name or run.span_id
            if metadata and metadata.get("langgraph_node"):
                fields["metadata"] = {"langgraph_node": metadata["langgraph_node"]}
            self._emit("agent.step.started", fields)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        if run.kind == "turn":
            fields = self._end_fields(run)
            fields.pop("span_id", None)
            self._emit("turn.completed", fields)
        elif run.kind == "step":
            fields = self._end_fields(run)
            fields["workflow_step_id"] = run.name or run.span_id
            self._emit("agent.step.completed", fields)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        if run.kind == "turn":
            fields = self._end_fields(run)
            fields.pop("span_id", None)
            fields.update(self._error_fields(error))
            self._emit("turn.failed", fields)
        elif run.kind == "step":
            fields = self._end_fields(run)
            fields["workflow_step_id"] = run.name or run.span_id
            fields.update(self._error_fields(error))
            self._emit("agent.step.failed", fields)

    @staticmethod
    def _turn_title(inputs: Any) -> str | None:
        messages = inputs.get("messages") if isinstance(inputs, dict) else None
        if not isinstance(messages, list) or not messages:
            return None
        last = messages[-1]
        text = getattr(last, "content", None)
        if text is None and isinstance(last, dict):
            text = last.get("content")
        if not isinstance(text, str) or not text.strip():
            return None
        text = " ".join(text.split())
        return text[:TURN_TITLE_MAX_CHARS]

    def _model_start(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        metadata: dict[str, Any] | None,
        invocation_params: dict[str, Any] | None,
        prompt: Any,
    ) -> None:
        params = invocation_params or {}
        meta = metadata or {}
        model = (
            meta.get("ls_model_name")
            or params.get("model")
            or params.get("model_name")
            or "unknown"
        )
        provider = meta.get("ls_provider") or params.get("_type") or "unknown"
        model_meta: dict[str, Any] = {"provider": str(provider), "model": str(model)}
        temperature = params.get("temperature", meta.get("ls_temperature"))
        if isinstance(temperature, (int, float)):
            model_meta["temperature"] = temperature
        run = self._open(
            run_id, parent_run_id, kind="model", emitted=True, name=str(model), meta=model_meta
        )
        fields = self._span_fields(run)
        fields["model"] = model_meta
        fields["input_hash"] = _hash(prompt)
        self._emit("model.call.started", fields)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        prompt = [[_message_dump(m) for m in batch] for batch in messages]
        self._model_start(run_id, parent_run_id, metadata, kwargs.get("invocation_params"), prompt)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_start(run_id, parent_run_id, metadata, kwargs.get("invocation_params"), prompts)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        llm_output = response.llm_output or {}
        model_meta = dict(run.meta)
        if llm_output.get("model_name"):
            model_meta["model"] = str(llm_output["model_name"])
        fields = self._end_fields(run)
        fields["name"] = model_meta["model"]
        fields["model"] = model_meta
        fields["output_hash"] = _hash(_result_dump(response))
        usage = _usage(response)
        if usage:
            fields["usage"] = usage
        fields["status"] = "ok"
        self._emit("model.call.completed", fields)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        fields = self._end_fields(run)
        fields["model"] = dict(run.meta)
        fields.update(self._error_fields(error))
        self._emit("model.call.failed", fields)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        tool_meta: dict[str, Any] = {"name": str(name)}
        run = self._open(
            run_id, parent_run_id, kind="tool", emitted=True, name=str(name), meta=tool_meta
        )
        fields = self._span_fields(run)
        fields["tool"] = tool_meta
        fields["input_hash"] = _hash(inputs if inputs is not None else input_str)
        self._emit("tool.call.started", fields)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        fields = self._end_fields(run)
        fields["tool"] = dict(run.meta)
        fields["output_hash"] = _hash(_tool_output_dump(output))
        fields["status"] = "ok"
        self._emit("tool.call.completed", fields)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        fields = self._end_fields(run)
        fields["tool"] = dict(run.meta)
        fields.update(self._error_fields(error))
        self._emit("tool.call.failed", fields)

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "retriever"
        run = self._open(run_id, parent_run_id, kind="retrieval", emitted=True, name=str(name))
        fields = self._span_fields(run)
        fields["category"] = "retrieval"
        fields["input_hash"] = _hash(query)
        self._emit("retrieval.started", fields)

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        fields = self._end_fields(run)
        fields["category"] = "retrieval"
        fields["output_hash"] = _hash([getattr(d, "page_content", repr(d)) for d in documents])
        fields["retrieval"] = {"documents": len(documents)}
        fields["status"] = "ok"
        self._emit("retrieval.completed", fields)

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._close(run_id)
        if run is None:
            return
        fields = self._end_fields(run)
        fields["category"] = "retrieval"
        fields.update(self._error_fields(error))
        self._emit("retrieval.failed", fields)


def _message_dump(message: Any) -> Any:
    dump: Callable[[], Any] | None = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return repr(message)
    return message if isinstance(message, (str, dict)) else repr(message)


def _result_dump(response: Any) -> Any:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return repr(response)
    return repr(response)


def _tool_output_dump(output: Any) -> Any:
    if isinstance(output, (str, int, float, bool, dict, list)) or output is None:
        return output
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return repr(output)
    return repr(output)


def _usage(response: Any) -> dict[str, Any] | None:
    usage: dict[str, Any] = {}
    generations = getattr(response, "generations", None) or []
    for batch in generations:
        # One batch is one request, and providers attach that request's usage to
        # every choice, so count the first choice that carries it and stop:
        # summing the batch would multiply the tokens by n.
        for generation in batch:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if isinstance(meta, dict):
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    if isinstance(meta.get(key), int):
                        usage[key] = usage.get(key, 0) + meta[key]
                break
    if not usage:
        token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        aliases = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }
        for key, alias in aliases.items():
            value = token_usage.get(key, token_usage.get(alias))
            if isinstance(value, int):
                usage[key] = value
    if not usage:
        return None
    if "total_tokens" not in usage and "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["scope"] = "self"
    usage["source"] = "runtime"
    return usage
