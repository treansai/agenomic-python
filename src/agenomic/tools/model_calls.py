"""Model call admission: ask the gateway before the runtime calls a model provider.

This is cooperative. The gateway decides on what the runtime declares (provider,
model, token and cost estimates, data classification, destination host); it never
sees the prompt and never makes or observes the provider call. A runtime that
skips :meth:`ModelCallsResource.authorize`, declares other values than it uses,
or can reach a provider directly is not stopped. The report carries the
runtime's own usage figures.

Cloud mode only, on protect runs: the routes are
``POST /v1/tool-execution/runs/{run_id}/model-calls/authorize`` and ``/report``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from agenomic.tools.models import (
    ProtectDecision,
    SignedPermit,
    ToolApprovalPending,
    ToolCallDenied,
    ToolExecutionError,
    ToolInvocationPhase,
)
from agenomic.tools.resources import (
    ToolsResource,
    _approval_status,
    _decision_envelope,
    _parse_authorize,
    asend_request,
    send_request,
)

logger = logging.getLogger("agenomic.tools.model_calls")

ModelCallDecisionKind = Literal["allow", "pending", "denied"]
ModelToolChoice = Literal["auto", "none", "required", "tool", "unspecified"]
_APPROVAL_GRANTED = ("approved", "consumed")


class ModelCost(BaseModel):
    """Estimated cost of a call in integer minor units.

    Example:
        >>> ModelCost(amount_minor=12, currency="EUR").model_dump()
        {'amount_minor': 12, 'currency': 'EUR'}
    """

    amount_minor: int
    currency: str


class ModelUsage(BaseModel):
    """Token usage the runtime observed.

    Example:
        >>> ModelUsage(input_tokens=10, output_tokens=5).output_tokens
        5
    """

    input_tokens: int
    output_tokens: int


class ModelCallRequest(BaseModel):
    """What the runtime declares about a model call it is about to make.

    The gateway rejects unknown fields, so prompt text cannot be sent. ``tools``
    are the tool names offered to the model; ``data_classification`` labels are
    declared, never inferred (empty means ``unclassified``). The same instance
    is used to authorize, resume and report: the approval is bound to the
    declared parameters.

    Example:
        >>> request = ModelCallRequest(provider="openai", model="gpt-5",
        ...     max_output_tokens=512, estimated_input_tokens=300, logical_call_id="c1")
        >>> request.tool_id
        'model:openai/gpt-5'
        >>> sorted(request.to_body())
        ['attempt', 'estimated_input_tokens', 'logical_call_id', 'max_output_tokens', 'model', 'provider', 'repetition']
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    max_output_tokens: int
    estimated_input_tokens: int
    logical_call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    repetition: int = 1
    attempt: int = 1
    endpoint_host: Optional[str] = None
    estimated_cost: Optional[ModelCost] = None
    data_classification: list[str] = Field(default_factory=list)
    tool_choice: ModelToolChoice = "unspecified"
    tool_choice_name: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    stream: bool = False
    parent_call_id: Optional[str] = None
    deadline_ms: Optional[int] = None

    @property
    def tool_id(self) -> str:
        """The id policies, restrictions and the kill switch address: ``model:{provider}/{model}``."""
        return f"model:{self.provider}/{self.model}"

    def to_body(self) -> dict[str, object]:
        """Authorize body: only declared fields, defaults the gateway applies are omitted."""
        body: dict[str, object] = {
            "repetition": self.repetition,
            "logical_call_id": self.logical_call_id,
            "attempt": self.attempt,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
        }
        if self.endpoint_host:
            body["endpoint_host"] = self.endpoint_host
        if self.estimated_cost is not None:
            body["estimated_cost"] = self.estimated_cost.model_dump()
        if self.data_classification:
            body["data_classification"] = list(self.data_classification)
        if self.tool_choice != "unspecified":
            body["tool_choice"] = self.tool_choice
        if self.tool_choice_name:
            body["tool_choice_name"] = self.tool_choice_name
        if self.tools:
            body["tools"] = list(self.tools)
        if self.stream:
            body["stream"] = True
        if self.parent_call_id:
            body["parent_call_id"] = self.parent_call_id
        if self.deadline_ms is not None:
            body["deadline_ms"] = self.deadline_ms
        return body


class ModelCallDecision(BaseModel):
    """The gateway answer to an authorization.

    ``decision`` is the wire value: ``allow`` (200, with the signed ``permit``),
    ``pending`` (202, a policy required approval or paused: ``approval_id``) or
    ``denied`` (403). The policy outcome (``allow``, ``deny``,
    ``require_approval``) is in ``protect.outcome``.

    Example:
        >>> decision = ModelCallDecision.model_validate({
        ...     "decision": "pending", "tool": "model:openai/gpt-5", "record_id": "r",
        ...     "approval_id": "apr_1", "protect": None, "phase": "approval_pending"})
        >>> (decision.requires_approval, decision.allowed, decision.denied)
        (True, False, False)
    """

    model_config = ConfigDict(extra="allow")

    decision: ModelCallDecisionKind
    tool: str
    record_id: str
    approval_id: Optional[str] = None
    permit: Optional[SignedPermit] = None
    protect: Optional[ProtectDecision] = None
    phase: ToolInvocationPhase

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def requires_approval(self) -> bool:
        return self.decision == "pending"

    @property
    def denied(self) -> bool:
        return self.decision == "denied"


class ModelCallReportResult(BaseModel):
    """Answer to a report.

    Example:
        >>> ModelCallReportResult(record_id="r").record_id
        'r'
    """

    record_id: str


def usage_from_response(response: object) -> Optional[ModelUsage]:
    """Token usage of a provider response, or ``None`` when it carries none.

    Reads ``usage`` as an attribute or a key and accepts both vocabularies
    (``prompt_tokens``/``completion_tokens`` and ``input_tokens``/``output_tokens``).

    Example:
        >>> usage_from_response({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        ModelUsage(input_tokens=7, output_tokens=3)
        >>> usage_from_response({"id": "x"}) is None
        True
    """
    usage = _field(response, "usage")
    if usage is None:
        return None
    input_tokens = _field(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _field(usage, "prompt_tokens")
    output_tokens = _field(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _field(usage, "completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _field(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


class ModelCallHandle:
    """What the body of :meth:`ModelCallsResource.guard` sees.

    Record the observed usage with :meth:`set_usage` or :meth:`set_usage_from`
    before the block ends; set :attr:`is_error` when the provider answered with
    an error that did not raise.

    Example:
        >>> handle = ModelCallHandle(
        ...     ModelCallRequest(provider="openai", model="m", max_output_tokens=1,
        ...         estimated_input_tokens=1),
        ...     ModelCallDecision.model_validate({"decision": "allow", "tool": "model:openai/m",
        ...         "record_id": "r", "phase": "dispatching"}))
        >>> handle.set_usage(input_tokens=4, output_tokens=2)
        >>> handle.usage
        ModelUsage(input_tokens=4, output_tokens=2)
    """

    def __init__(self, request: ModelCallRequest, decision: ModelCallDecision) -> None:
        self.request = request
        self.decision = decision
        self.usage: Optional[ModelUsage] = None
        self.is_error = False
        self.report: Optional[ModelCallReportResult] = None

    @property
    def record_id(self) -> str:
        return self.decision.record_id

    def set_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        self.usage = ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def set_usage_from(self, response: object) -> None:
        """Take the usage from a provider response; leaves it unset when there is none."""
        found = usage_from_response(response)
        if found is not None:
            self.usage = found


def _decision_from(body: Mapping[str, object], request: ModelCallRequest) -> ModelCallDecision:
    try:
        return ModelCallDecision.model_validate(dict(body))
    except ValueError as error:
        raise ToolExecutionError(
            "invalid_response",
            f"model-calls/authorize answered for {request.tool_id} with an unreadable decision",
            0,
        ) from error


def _require_dispatch(decision: ModelCallDecision) -> ModelCallDecision:
    if decision.allowed and decision.permit is None:
        raise ToolExecutionError(
            "invalid_response",
            f"model-calls/authorize allowed {decision.tool} without a permit; refusing to call the model",
            0,
        )
    return decision


def _refuse(decision: ModelCallDecision) -> Exception:
    envelope = _decision_envelope(decision.model_dump(mode="json"), "denied")
    return ToolCallDenied(decision.tool, envelope)


def _hold(decision: ModelCallDecision) -> Exception:
    envelope = _decision_envelope(decision.model_dump(mode="json"), "pending")
    return ToolApprovalPending(
        decision.tool, decision.approval_id or "", decision.record_id, envelope
    )


def _report_body(
    request: ModelCallRequest,
    decision: ModelCallDecision,
    usage: Optional[ModelUsage],
    is_error: bool,
    duration_ms: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "repetition": request.repetition,
        "logical_call_id": request.logical_call_id,
        "attempt": request.attempt,
        "provider": request.provider,
        "model": request.model,
        "is_error": is_error,
        "duration_ms": duration_ms,
    }
    if usage is not None:
        body["usage"] = usage.model_dump()
    if decision.permit is not None:
        body["permit"] = decision.permit.model_dump()
    return body


def _check_granted(decision: ModelCallDecision, status: str, deadline: float) -> bool:
    if status in _APPROVAL_GRANTED:
        return True
    if status != "pending":
        raise ToolCallDenied(
            decision.tool,
            _decision_envelope(decision.model_dump(mode="json"), "denied"),
            code=status or "policy_denied",
        )
    if time.monotonic() >= deadline:
        raise ToolExecutionError(
            "approval_timeout",
            f"approval for {decision.tool} was still pending after the timeout",
            0,
        )
    return False


class ModelCallsResource:
    """The ``client.tools.model_calls`` namespace (cloud mode, protect runs).

    ``authorize`` returns the typed decision for all three outcomes. ``guard``
    is the helper for a runtime: it authorizes, raises
    :class:`~agenomic.tools.ToolCallDenied` or
    :class:`~agenomic.tools.ToolApprovalPending` before the provider is
    called, and reports the usage on exit. A 409 on an identity or on another
    unresolved attempt raises :class:`~agenomic.tools.ToolCallUnresolved` and
    is never retried.

    Example:
        >>> from agenomic import Client
        >>> Client().tools.model_calls.report_path("run")
        '/v1/tool-execution/runs/run/model-calls/report'
    """

    def __init__(self, tools: ToolsResource) -> None:
        self._tools = tools

    @staticmethod
    def authorize_path(run_id: str) -> str:
        return f"/v1/tool-execution/runs/{run_id}/model-calls/authorize"

    @staticmethod
    def report_path(run_id: str) -> str:
        return f"/v1/tool-execution/runs/{run_id}/model-calls/report"

    def _require_cloud(self) -> None:
        if not self._tools.is_cloud:
            raise ToolExecutionError(
                "cloud_required", "model call admission is decided by the Agenomic Cloud gateway", 0
            )

    def authorize(self, run_id: str, request: ModelCallRequest) -> ModelCallDecision:
        """Ask whether the runtime may call this model; the provider is not called here.

        Returns the decision for ``allow`` (with the permit :meth:`report`
        needs), ``pending`` (``approval_id``) and ``denied``. Re-posting the same
        request after a reviewer decided resumes a pending call.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.model_calls.authorize("run", ModelCallRequest(
            ...     provider="openai", model="m", max_output_tokens=1, estimated_input_tokens=1))
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: cloud_required: model call admission is decided by the Agenomic Cloud gateway
        """
        self._require_cloud()
        path = self.authorize_path(run_id)
        response = send_request(self._tools._client, "POST", path, request.to_body())
        body = _parse_authorize(request.tool_id, "POST", path, response)
        return _require_dispatch(_decision_from(body, request))

    async def aauthorize(self, run_id: str, request: ModelCallRequest) -> ModelCallDecision:
        """Async counterpart of :meth:`authorize`."""
        self._require_cloud()
        path = self.authorize_path(run_id)
        response = await asend_request(self._tools._client, "POST", path, request.to_body())
        body = _parse_authorize(request.tool_id, "POST", path, response)
        return _require_dispatch(_decision_from(body, request))

    def report(
        self,
        run_id: str,
        request: ModelCallRequest,
        decision: ModelCallDecision,
        *,
        usage: Optional[ModelUsage] = None,
        is_error: bool = False,
        duration_ms: int = 0,
    ) -> ModelCallReportResult:
        """Settle an allowed call with the usage the runtime observed (single use permit).

        The gateway answers 403 ``permit_invalid`` without a valid permit and
        409 for a second report or another provider or model; both raise
        :class:`~agenomic.tools.ToolExecutionError`. The usage is the runtime's
        own word, and the budgets count the requested tokens, not these.
        """
        self._require_cloud()
        body = _report_body(request, decision, usage, is_error, duration_ms)
        return ModelCallReportResult.model_validate(
            self._tools._request("POST", self.report_path(run_id), body)
        )

    async def areport(
        self,
        run_id: str,
        request: ModelCallRequest,
        decision: ModelCallDecision,
        *,
        usage: Optional[ModelUsage] = None,
        is_error: bool = False,
        duration_ms: int = 0,
    ) -> ModelCallReportResult:
        """Async counterpart of :meth:`report`."""
        self._require_cloud()
        body = _report_body(request, decision, usage, is_error, duration_ms)
        return ModelCallReportResult.model_validate(
            await self._tools._arequest("POST", self.report_path(run_id), body)
        )

    def wait_for_approval(
        self,
        run_id: str,
        request: ModelCallRequest,
        decision: ModelCallDecision,
        *,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> ModelCallDecision:
        """Wait for the approval of a pending call, then re-post the identical request.

        ``approved`` and ``consumed`` both re-post once. A model call has no
        stored result: if the first approval was already consumed the re-post
        lands on a 409 and raises :class:`~agenomic.tools.ToolCallUnresolved`.
        ``rejected``, ``expired`` or any other terminal status raises
        :class:`~agenomic.tools.ToolCallDenied` with that status as ``code``;
        ``approval_timeout`` is raised when it is still pending after
        ``timeout`` seconds.
        """
        if not decision.approval_id:
            raise ToolExecutionError("approval_unknown", "the decision carries no approval", 0)
        deadline = time.monotonic() + timeout
        while True:
            record = self._tools._request("GET", f"/v1/protect/approvals/{decision.approval_id}")
            if _check_granted(decision, _approval_status(record), deadline):
                break
            time.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))
        return self._resumed(self.authorize(run_id, request))

    async def await_for_approval(
        self,
        run_id: str,
        request: ModelCallRequest,
        decision: ModelCallDecision,
        *,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> ModelCallDecision:
        """Async counterpart of :meth:`wait_for_approval`."""
        if not decision.approval_id:
            raise ToolExecutionError("approval_unknown", "the decision carries no approval", 0)
        deadline = time.monotonic() + timeout
        while True:
            record = await self._tools._arequest(
                "GET", f"/v1/protect/approvals/{decision.approval_id}"
            )
            if _check_granted(decision, _approval_status(record), deadline):
                break
            await asyncio.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))
        return self._resumed(await self.aauthorize(run_id, request))

    @staticmethod
    def _resumed(decision: ModelCallDecision) -> ModelCallDecision:
        if decision.denied:
            raise _refuse(decision)
        if decision.requires_approval:
            raise _hold(decision)
        return decision

    def admit(
        self,
        run_id: str,
        request: ModelCallRequest,
        *,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> ModelCallDecision:
        """Authorize and return only an allowed decision.

        ``denied`` raises :class:`~agenomic.tools.ToolCallDenied`; ``pending``
        raises :class:`~agenomic.tools.ToolApprovalPending` unless ``wait`` is
        true, in which case it waits for the reviewer and resumes.
        """
        decision = self.authorize(run_id, request)
        if decision.denied:
            raise _refuse(decision)
        if decision.requires_approval:
            if not wait:
                raise _hold(decision)
            decision = self.wait_for_approval(
                run_id, request, decision, poll_interval=poll_interval, timeout=timeout
            )
        return decision

    async def aadmit(
        self,
        run_id: str,
        request: ModelCallRequest,
        *,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> ModelCallDecision:
        """Async counterpart of :meth:`admit`."""
        decision = await self.aauthorize(run_id, request)
        if decision.denied:
            raise _refuse(decision)
        if decision.requires_approval:
            if not wait:
                raise _hold(decision)
            decision = await self.await_for_approval(
                run_id, request, decision, poll_interval=poll_interval, timeout=timeout
            )
        return decision

    @contextmanager
    def guard(
        self,
        run_id: str,
        request: ModelCallRequest,
        *,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> Iterator[ModelCallHandle]:
        """Wrap one provider call: authorize on entry, report the usage on exit.

        The body runs only for an allowed call. When it raises, the call is
        reported with ``is_error=True`` and the exception propagates (a failed
        report is logged, not raised, so it never hides the original error).
        When it ends normally a failed report raises. The duration is measured
        around the body. This is cooperative: nothing stops a provider call
        made outside this block.

        Example:
            >>> from agenomic import Client
            >>> request = ModelCallRequest(provider="openai", model="m",
            ...     max_output_tokens=1, estimated_input_tokens=1)
            >>> with Client().tools.model_calls.guard("run", request):
            ...     pass
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: cloud_required: model call admission is decided by the Agenomic Cloud gateway
        """
        decision = self.admit(
            run_id, request, wait=wait, poll_interval=poll_interval, timeout=timeout
        )
        handle = ModelCallHandle(request, decision)
        started = time.monotonic()
        try:
            yield handle
        except BaseException:
            handle.is_error = True
            try:
                self._finish(run_id, handle, started)
            except ToolExecutionError:
                logger.warning("model call %s was not reported after an error", handle.record_id)
            raise
        self._finish(run_id, handle, started)

    def _finish(self, run_id: str, handle: ModelCallHandle, started: float) -> None:
        handle.report = self.report(
            run_id,
            handle.request,
            handle.decision,
            usage=handle.usage,
            is_error=handle.is_error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    @asynccontextmanager
    async def aguard(
        self,
        run_id: str,
        request: ModelCallRequest,
        *,
        wait: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> AsyncIterator[ModelCallHandle]:
        """Async counterpart of :meth:`guard`."""
        decision = await self.aadmit(
            run_id, request, wait=wait, poll_interval=poll_interval, timeout=timeout
        )
        handle = ModelCallHandle(request, decision)
        started = time.monotonic()
        try:
            yield handle
        except BaseException:
            handle.is_error = True
            try:
                await self._afinish(run_id, handle, started)
            except ToolExecutionError:
                logger.warning("model call %s was not reported after an error", handle.record_id)
            raise
        await self._afinish(run_id, handle, started)

    async def _afinish(self, run_id: str, handle: ModelCallHandle, started: float) -> None:
        handle.report = await self.areport(
            run_id,
            handle.request,
            handle.decision,
            usage=handle.usage,
            is_error=handle.is_error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )


__all__ = [
    "ModelCallDecision",
    "ModelCallHandle",
    "ModelCallReportResult",
    "ModelCallRequest",
    "ModelCallsResource",
    "ModelCost",
    "ModelToolChoice",
    "ModelUsage",
    "usage_from_response",
]
