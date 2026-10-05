from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import PrivateAttr
from typing_extensions import override

from agenomic._transport import aapi_request, api_request, segment
from agenomic.canonical.hashing import canonical_json
from agenomic.exceptions import ApiError
from agenomic.experiments.errors import (
    InfrastructureError,
    LeaseLost,
    RecordedFixtureAmbiguous,
    RecordedFixtureMiss,
    RunnerConfigurationError,
)
from agenomic.experiments.isolation import jsonable
from agenomic.experiments.secrets import redact_message, redact_outbound, replace_secrets

if TYPE_CHECKING:
    from agenomic.experiments.context import TrialContext

__all__ = [
    "APPROVAL_TEXT",
    "FIXTURE_MISS_TEXT",
    "HttpToolProxy",
    "ToolProxy",
    "logical_call_id",
    "wrap_tools",
]

FIXTURE_MISS_TEXT = "no recorded response is available for this call"
APPROVAL_TEXT = "this action requires human approval, which is not available inside experiments"
DENIED_TEXT = "this action was denied by policy"
IN_PROGRESS_DELAYS = (0.5, 1.0, 2.0)
_CALL_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}", re.ASCII)
_CURRENT: contextvars.ContextVar[Optional[tuple[Optional[str], Optional[RunnableConfig]]]] = (
    contextvars.ContextVar("agenomic_experiment_tool_call", default=None)
)

_sleep = time.sleep
_asleep = asyncio.sleep


class ToolProxy(Protocol):
    def call(self, body: Mapping[str, Any]) -> dict[str, Any]: ...

    async def acall(self, body: Mapping[str, Any]) -> dict[str, Any]: ...

    def report(self, logical_call_id: str, body: Mapping[str, Any]) -> dict[str, Any]: ...

    async def areport(self, logical_call_id: str, body: Mapping[str, Any]) -> dict[str, Any]: ...


class HttpToolProxy:
    def __init__(self, http: Any, trial_id: str) -> None:
        self._http = http
        self._base = f"/v1/experiment-runner/trials/{segment(trial_id)}/tool-calls"

    def call(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return dict(api_request(self._http, "POST", self._base, body, retry=True).body)

    async def acall(self, body: Mapping[str, Any]) -> dict[str, Any]:
        response = await aapi_request(self._http, "POST", self._base, body, retry=True)
        return dict(response.body)

    def report(self, logical_call_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        path = f"{self._base}/{segment(logical_call_id)}/report"
        return dict(api_request(self._http, "POST", path, body, retry=True).body)

    async def areport(self, logical_call_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        path = f"{self._base}/{segment(logical_call_id)}/report"
        response = await aapi_request(self._http, "POST", path, body, retry=True)
        return dict(response.body)


def logical_call_id(
    tool_call_id: Optional[str],
    config: Optional[Mapping[str, Any]],
    tool: str,
    arguments: Any,
) -> str:
    if isinstance(tool_call_id, str) and _CALL_ID.fullmatch(tool_call_id):
        return tool_call_id
    configurable = (config or {}).get("configurable") or {}
    namespace = str(configurable.get("checkpoint_ns") or "")
    task_id = str(configurable.get("__pregel_task_id") or "")
    material = f"{namespace}|{task_id}|{tool}|{canonical_json(arguments)}"
    return "call_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _tool_call_id(input: Any) -> Optional[str]:
    if isinstance(input, Mapping) and input.get("type") == "tool_call":
        value = input.get("id")
        return value if isinstance(value, str) else None
    return None


def _native(result: Any) -> Any:
    if isinstance(result, Mapping) and isinstance(result.get("agenomic"), Mapping):
        return result.get("result")
    return result


def _json_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return dict(json.loads(json.dumps(dict(arguments), allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise RunnerConfigurationError(
            "tool_arguments_not_json", "tool arguments must be JSON values"
        ) from error


class _ProxiedTool(BaseTool):
    _inner: BaseTool = PrivateAttr()
    _context: Any = PrivateAttr()

    def attach(self, inner: BaseTool, context: TrialContext) -> _ProxiedTool:
        self._inner = inner
        self._context = context
        return self

    @override
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        token = _CURRENT.set((_tool_call_id(input), config))
        try:
            return super().invoke(input, config, **kwargs)
        finally:
            _CURRENT.reset(token)

    @override
    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        token = _CURRENT.set((_tool_call_id(input), config))
        try:
            return await super().ainvoke(input, config, **kwargs)
        finally:
            _CURRENT.reset(token)

    def _fail(self, error: BaseException) -> BaseException:
        self._context._state.mark_terminal(error)
        return error

    def _prepare(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        context = self._context
        if context.tool_mode == "none":
            raise self._fail(
                RunnerConfigurationError(
                    "tool_mode_none", "the experiment declares no tools, but a tool was called"
                )
            )
        raw = dict(kwargs) if kwargs or not args else {"input": args[0]}
        arguments = _json_arguments(raw)
        current = _CURRENT.get() or (None, None)
        logical = logical_call_id(current[0], current[1], self.name, arguments)
        state = context._state
        body = {
            "lease_token": state.lease_token,
            "logical_call_id": logical,
            "attempt": 1,
            "tool": self.name,
            "arguments": replace_secrets(arguments, state.literals),
            "parent_call_id": None,
        }
        return logical, arguments, body

    def _refusal(self, error: ApiError) -> BaseException:
        code = error.code
        if code == "experiment_lease_stale":
            return self._fail(LeaseLost())
        if code == "registry_unavailable" or error.status >= 500:
            return self._fail(InfrastructureError("proxy_unavailable", error.message))
        return self._fail(RunnerConfigurationError(code.removeprefix("experiment_"), error.message))

    def _outcome(self, response: Mapping[str, Any]) -> tuple[str, Any]:
        status = response.get("status")
        if status == "ok":
            return "value", _native(response.get("result"))
        if status == "denied":
            decision = response.get("decision")
            explanation = (
                decision.get("safe_explanation") if isinstance(decision, Mapping) else None
            )
            return "value", f"{DENIED_TEXT}: {explanation}" if explanation else DENIED_TEXT
        if status == "approval_required":
            return "value", APPROVAL_TEXT
        if status == "recorded_fixture_miss":
            policy = response.get("on_fixture_miss") or self._context.view.tools.on_fixture_miss
            if policy == "tool_error":
                return "value", FIXTURE_MISS_TEXT
            raise self._fail(RecordedFixtureMiss(self.name, response.get("arguments_hash")))
        if status == "recorded_fixture_ambiguous":
            raise self._fail(RecordedFixtureAmbiguous(self.name))
        if status == "authorized":
            return "authorized", response
        raise self._fail(
            InfrastructureError("invalid_response", f"unknown tool call status {status!r}")
        )

    def _report_body(
        self, value: Any, is_error: bool, started: float, permit: Any
    ) -> dict[str, Any]:
        state = self._context._state
        return {
            "lease_token": state.lease_token,
            "attempt": 1,
            "value": redact_outbound(jsonable(value), state.literals),
            "is_error": is_error,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "permit": permit,
        }

    def _check_report(self, response: Mapping[str, Any]) -> None:
        if response.get("recorded") is not True:
            raise self._fail(
                InfrastructureError("invalid_response", "the tool report was not recorded")
            )

    @override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        logical, arguments, body = self._prepare(args, kwargs)
        proxy = self._context._state.proxy
        if proxy is None:
            raise self._fail(
                RunnerConfigurationError("proxy_unavailable", "this trial has no tool proxy")
            )
        response: Optional[dict[str, Any]] = None
        for delay in (*IN_PROGRESS_DELAYS, None):
            try:
                response = proxy.call(body)
                break
            except ApiError as error:
                if error.code != "experiment_tool_call_in_progress":
                    raise self._refusal(error) from error
                if delay is None:
                    raise self._fail(
                        InfrastructureError("tool_call_in_progress", error.message)
                    ) from error
                _sleep(delay)
        assert response is not None
        kind, value = self._outcome(response)
        if kind == "value":
            return value
        return self._run_live(logical, arguments, value, proxy)

    @override
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        logical, arguments, body = self._prepare(args, kwargs)
        proxy = self._context._state.proxy
        if proxy is None:
            raise self._fail(
                RunnerConfigurationError("proxy_unavailable", "this trial has no tool proxy")
            )
        response: Optional[dict[str, Any]] = None
        for delay in (*IN_PROGRESS_DELAYS, None):
            try:
                response = await proxy.acall(body)
                break
            except ApiError as error:
                if error.code != "experiment_tool_call_in_progress":
                    raise self._refusal(error) from error
                if delay is None:
                    raise self._fail(
                        InfrastructureError("tool_call_in_progress", error.message)
                    ) from error
                await _asleep(delay)
        assert response is not None
        kind, value = self._outcome(response)
        if kind == "value":
            return value
        return await self._arun_live(logical, arguments, value, proxy)

    def _live_cached(self, logical: str) -> Optional[dict[str, Any]]:
        state = self._context._state
        with state.lock:
            return cast(Optional[dict[str, Any]], state.live_reports.get(logical))

    def _remember_live(
        self, logical: str, report: dict[str, Any], value: Any, error: Optional[BaseException]
    ) -> None:
        state = self._context._state
        with state.lock:
            state.live_reports[logical] = {"body": report, "value": value, "error": error}

    def _run_live(
        self, logical: str, arguments: dict[str, Any], decision: Mapping[str, Any], proxy: ToolProxy
    ) -> Any:
        cached = self._live_cached(logical)
        if cached is None:
            started = time.monotonic()
            error: Optional[BaseException] = None
            try:
                value = self._inner.invoke(arguments)
            except Exception as raised:
                error, value = (
                    raised,
                    {"error": redact_message(str(raised), self._context._state.literals)},
                )
            report = self._report_body(value, error is not None, started, decision.get("permit"))
            self._remember_live(logical, report, value, error)
            cached = {"body": report, "value": value, "error": error}
        try:
            self._check_report(proxy.report(logical, cached["body"]))
        except ApiError as refused:
            raise self._refusal(refused) from refused
        if cached["error"] is not None:
            raise cached["error"]
        return cached["value"]

    async def _arun_live(
        self, logical: str, arguments: dict[str, Any], decision: Mapping[str, Any], proxy: ToolProxy
    ) -> Any:
        cached = self._live_cached(logical)
        if cached is None:
            started = time.monotonic()
            error: Optional[BaseException] = None
            try:
                value = await self._inner.ainvoke(arguments)
            except Exception as raised:
                error, value = (
                    raised,
                    {"error": redact_message(str(raised), self._context._state.literals)},
                )
            report = self._report_body(value, error is not None, started, decision.get("permit"))
            self._remember_live(logical, report, value, error)
            cached = {"body": report, "value": value, "error": error}
        try:
            self._check_report(await proxy.areport(logical, cached["body"]))
        except ApiError as refused:
            raise self._refusal(refused) from refused
        if cached["error"] is not None:
            raise cached["error"]
        return cached["value"]


def _as_tool(tool: Any) -> BaseTool:
    if isinstance(tool, BaseTool):
        return tool
    if callable(tool):
        return StructuredTool.from_function(tool)
    raise TypeError("wrap_tools takes LangChain tools or plain functions")


def wrap_tools(context: TrialContext, tools: Sequence[Any]) -> list[BaseTool]:
    wrapped: list[BaseTool] = []
    for item in tools:
        inner = _as_tool(item)
        proxied = _ProxiedTool(
            name=inner.name,
            description=inner.description,
            args_schema=inner.args_schema,
            return_direct=inner.return_direct,
        )
        wrapped.append(proxied.attach(inner, context))
    return wrapped
