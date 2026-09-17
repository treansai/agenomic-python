"""``client.tools``: profiles, fixtures, scenarios, runs and the invoke path.

Cloud mode (``base_url`` set) talks to the Tool Gateway. Local mode runs the
in-process engine of :mod:`agenomic.tools.local`: same statuses, same
refusals, no network. Neither mode falls back to the other.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional, Union

import httpx

from agenomic.tools.local import LocalToolEngine
from agenomic.tools.models import (
    TOOL_EXECUTION_SCHEMA_VERSION,
    ProtectDecision,
    ToolApprovalPending,
    ToolCallDenied,
    ToolCallError,
    ToolCallResult,
    ToolExecutionError,
    ToolProvenance,
)

_IDEMPOTENCY_NAMESPACE = uuid.UUID("6b1f7a2e-9c44-4d0e-8f1a-2e7c0d9b5a31")
_APPROVAL_GRANTED = ("approved", "consumed")

LocalFunction = Callable[..., Any]
BeforeAction = Callable[[dict[str, Any]], Any]


def _unwrap_run(response: Mapping[str, Any]) -> dict[str, Any]:
    run = response.get("run")
    return dict(run) if isinstance(run, dict) else {}


def _config_body(
    config: Optional[Mapping[str, Any]],
    config_text: Optional[str],
    repetitions: int,
) -> dict[str, Any]:
    if (config is None) == (config_text is None):
        raise ValueError("provide exactly one of config or config_text")
    body: dict[str, Any] = {"repetitions": repetitions}
    if config is not None:
        body["config"] = dict(config)
    else:
        body["config_text"] = config_text
    return body


def _config_value(
    config: Optional[Mapping[str, Any]], config_text: Optional[str]
) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    try:
        parsed = json.loads(config_text or "")
    except ValueError as error:
        raise ToolExecutionError(
            "tool_execution_config_invalid",
            "local mode accepts JSON configuration text; pass config= for a mapping",
            400,
        ) from error
    if not isinstance(parsed, dict):
        raise ToolExecutionError(
            "tool_execution_config_invalid", "configuration must be an object", 400
        )
    return parsed


def _identity(
    tool: str,
    arguments: Mapping[str, Any],
    *,
    logical_call_id: str,
    repetition: int,
    attempt: int,
    parent_call_id: Optional[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "repetition": repetition,
        "logical_call_id": logical_call_id,
        "attempt": attempt,
        "tool": tool,
        "arguments": dict(arguments),
    }
    if parent_call_id:
        body["parent_call_id"] = parent_call_id
    return body


def _error_from_response(method: str, path: str, response: httpx.Response) -> ToolExecutionError:
    code, message = "http_error", f"{method} {path} returned {response.status_code}"
    try:
        error = response.json().get("error")
        if isinstance(error, dict):
            code = str(error.get("code", code))
            message = str(error.get("message", message))
    except (ValueError, AttributeError):
        pass
    return ToolExecutionError(code, message, response.status_code)


def _parse_response(method: str, path: str, response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise _error_from_response(method, path, response)
    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError as error:
        raise ToolExecutionError(
            "invalid_response", f"{method} {path} returned a non-JSON body", response.status_code
        ) from error
    return data if isinstance(data, dict) else {"data": data}


def _headers(idempotency_key: Optional[str]) -> dict[str, str]:
    return {"Idempotency-Key": idempotency_key} if idempotency_key else {}


def send_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
) -> httpx.Response:
    """Perform one cloud request; transport failures become ``transport_error``."""
    try:
        with client._http() as http:
            response: httpx.Response = http.request(
                method, path, json=body, headers=_headers(idempotency_key)
            )
            return response
    except httpx.HTTPError as exc:
        raise ToolExecutionError("transport_error", f"{method} {path} failed: {exc}", 0) from exc


async def asend_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
) -> httpx.Response:
    """Async counterpart of :func:`send_request`."""
    try:
        async with client._ahttp() as http:
            response: httpx.Response = await http.request(
                method, path, json=body, headers=_headers(idempotency_key)
            )
            return response
    except httpx.HTTPError as exc:
        raise ToolExecutionError("transport_error", f"{method} {path} failed: {exc}", 0) from exc


def typed_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
) -> dict[str, Any]:
    """Cloud request whose refusals raise :class:`ToolExecutionError` with the server code."""
    return _parse_response(
        method, path, send_request(client, method, path, body, idempotency_key=idempotency_key)
    )


async def atyped_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
) -> dict[str, Any]:
    """Async counterpart of :func:`typed_request`."""
    return _parse_response(
        method,
        path,
        await asend_request(client, method, path, body, idempotency_key=idempotency_key),
    )


def _parse_invoke(tool: str, method: str, path: str, response: httpx.Response) -> ToolCallResult:
    if response.status_code == 403:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("agenomic"), Mapping):
            raise ToolCallDenied(tool, ToolCallResult.from_response(body))
    return ToolCallResult.from_response(_parse_response(method, path, response))


def _parse_authorize(method: str, path: str, response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 403:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and ("decision" in body or "protect" in body):
            return body
    return _parse_response(method, path, response)


class ToolsResource:
    """The ``client.tools`` namespace.

    Example:
        >>> from agenomic import Client
        >>> tools = Client().tools                     # local mode, no network
        >>> tools.schema_version
        'agenomic.tool_execution/v1'
        >>> report = tools.validate(config={"schema_version": tools.schema_version, "mode": "mock"})
        >>> report["config"]["mode"]
        'mock'
    """

    schema_version = TOOL_EXECUTION_SCHEMA_VERSION

    def __init__(self, client: Any) -> None:
        self._client = client
        self._local: Optional[LocalToolEngine] = None

    # ── transport ─────────────────────────────────────────────────────

    @property
    def is_cloud(self) -> bool:
        """True when the client targets Agenomic Cloud.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.is_cloud
            False
        """
        return bool(getattr(self._client, "is_cloud", False))

    @property
    def local(self) -> LocalToolEngine:
        """The in-process engine (local mode only; never used in cloud mode).

        Example:
            >>> from agenomic import Client
            >>> Client().tools.local.adapters()["adapters"][0]["kind"]
            'local'
        """
        if self.is_cloud:
            raise ToolExecutionError(
                "cloud_required",
                "this client targets Agenomic Cloud; the local engine is not used",
                0,
            )
        if self._local is None:
            self._local = LocalToolEngine()
        return self._local

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        return typed_request(self._client, method, path, body, idempotency_key=idempotency_key)

    async def _arequest(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        return await atyped_request(
            self._client, method, path, body, idempotency_key=idempotency_key
        )

    # ── profiles and variables ────────────────────────────────────────

    def create_profile(
        self, *, name: str, environment: str = "dev", allowed_env: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Create an execution profile with its variable allowlist.

        Example:
            >>> from agenomic import Client
            >>> created = Client().tools.create_profile(name="replay-staging", allowed_env=["CRM_API_TOKEN"])
            >>> created["profile"]["name"], created["variables"][0]["available"]
            ('replay-staging', False)
        """
        body = {"name": name, "environment": environment, "allowed_env": list(allowed_env or [])}
        if self.is_cloud:
            return self._request("POST", "/v1/tool-execution/profiles", body)
        return self.local.create_profile(name, environment, list(allowed_env or []))

    def list_profiles(self) -> list[dict[str, Any]]:
        """List the execution profiles of the workspace.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.list_profiles()
            []
        """
        if self.is_cloud:
            return list(self._request("GET", "/v1/tool-execution/profiles").get("profiles", []))
        return self.local.list_profiles()

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        """Fetch one profile with the availability of its variables.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> pid = tools.create_profile(name="p", allowed_env=["TOKEN"])["profile"]["id"]
            >>> [(v["name"], v["available"]) for v in tools.get_profile(pid)["variables"]]
            [('TOKEN', False)]
        """
        if self.is_cloud:
            return self._request("GET", f"/v1/tool-execution/profiles/{profile_id}")
        return self.local.get_profile(profile_id)

    def set_variable(self, profile_id: str, name: str, value: str) -> dict[str, Any]:
        """Store a variable value write-only: it is never returned by any read.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> pid = tools.create_profile(name="p", allowed_env=["TOKEN"])["profile"]["id"]
            >>> tools.set_variable(pid, "TOKEN", "secret")["available"]
            True
            >>> "secret" in str(tools.get_profile(pid))
            False
        """
        if self.is_cloud:
            return self._request(
                "PUT",
                f"/v1/tool-execution/profiles/{profile_id}/variables/{name}",
                {"value": value},
            )
        return self.local.set_variable(profile_id, name, value)

    def variable_status(self, profile_id: str) -> list[dict[str, Any]]:
        """Availability per allowed variable. Never carries values.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> pid = tools.create_profile(name="p", allowed_env=["TOKEN"])["profile"]["id"]
            >>> [v["available"] for v in tools.variable_status(pid)]
            [False]
        """
        return list(self.get_profile(profile_id).get("variables", []))

    # ── contracts, fixtures, scenarios ────────────────────────────────

    def create_contract(
        self,
        *,
        name: str,
        version: int,
        input_schema: Optional[Mapping[str, Any]] = None,
        output_schema: Optional[Mapping[str, Any]] = None,
        effect: str = "unknown",
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        """Register a tool contract ``name@version`` with its effect class.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.create_contract(name="crm.get_customer", version=1, effect="read")["contract"]["effect"]
            'read'
        """
        body: dict[str, Any] = {"name": name, "version": version, "effect": effect}
        if input_schema is not None:
            body["input_schema"] = dict(input_schema)
        if output_schema is not None:
            body["output_schema"] = dict(output_schema)
        if description:
            body["description"] = description
        if self.is_cloud:
            return self._request("POST", "/v1/tool-execution/contracts", body)
        return self.local.create_contract(body)

    def create_fixture_set(
        self, *, name: str, version: int, fixtures: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Author a fixture set for the ``recorded`` strategy (unapproved until approved).

        ``request.arguments_hash`` may be omitted; it is computed from the
        canonical arguments.

        Example:
            >>> from agenomic import Client
            >>> created = Client().tools.create_fixture_set(name="docs", version=1, fixtures=[{
            ...     "fixture_id": "fx-1",
            ...     "request": {"tool": "documents.extract", "arguments": {"doc": "a"}},
            ...     "outcome": {"kind": "structured", "data": {"text": "hello"}},
            ...     "fidelity": "recorded_response", "origin": "authored"}])
            >>> created["fixture_set"]["approved"]
            False
        """
        body = {"name": name, "version": version, "fixtures": [dict(f) for f in fixtures]}
        if self.is_cloud:
            return self._request("POST", "/v1/tool-execution/fixture-sets", body)
        return self.local.create_fixture_set(name, version, [dict(f) for f in fixtures])

    def approve_fixture_set(self, fixture_set_id: str) -> dict[str, Any]:
        """Approve a fixture set so runs may reference it.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> sid = tools.create_fixture_set(name="docs", version=1, fixtures=[])["fixture_set"]["id"]
            >>> tools.approve_fixture_set(sid)["fixture_set"]["approved"]
            True
        """
        if self.is_cloud:
            return self._request(
                "POST", f"/v1/tool-execution/fixture-sets/{fixture_set_id}/approve", {}
            )
        return self.local.approve_fixture_set(fixture_set_id)

    def create_scenario(
        self,
        *,
        name: str,
        version: int,
        entities: Mapping[str, Any],
        tools: Mapping[str, Any],
        initial_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Register a stateful scenario definition (executed by the cloud engine).

        Example:
            >>> from agenomic import Client
            >>> Client().tools.create_scenario(name="tickets", version=1, entities={}, tools={})["scenario"]["name"]
            'tickets'
        """
        body = {
            "name": name,
            "version": version,
            "entities": dict(entities),
            "tools": dict(tools),
            "initial_state": dict(initial_state or {}),
        }
        if self.is_cloud:
            return self._request("POST", "/v1/tool-execution/scenarios", body)
        return self.local.create_scenario(body)

    def adapters(self) -> dict[str, Any]:
        """Capability matrix of the adapters available in this mode.

        Example:
            >>> from agenomic import Client
            >>> [a["kind"] for a in Client().tools.adapters()["adapters"]]
            ['local']
        """
        if self.is_cloud:
            return self._request("GET", "/v1/tool-execution/adapters")
        return self.local.adapters()

    # ── configuration ─────────────────────────────────────────────────

    def validate(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_text: Optional[str] = None,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        """Structural validation only: no store lookups, no network.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.validate(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "safety": {"allow_implicit_fallback": True}})
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: tool_execution_config_invalid: safety.allow_implicit_fallback must be false: a missing mock never falls back to a live call
        """
        if self.is_cloud:
            return self._request(
                "POST",
                "/v1/tool-execution/validate",
                _config_body(config, config_text, repetitions),
            )
        _config_body(config, config_text, repetitions)
        return self.local.validate(_config_value(config, config_text), repetitions)

    def preflight(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_text: Optional[str] = None,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        """Immutable execution plan: live vs mock tools, destinations, missing
        variables, possible effects and the ``plan_hash`` an approval binds to.

        Example:
            >>> from agenomic import Client
            >>> plan = Client().tools.preflight(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
            >>> plan["runnable"], plan["plan"]["mock_tools"]
            (True, ['email.send'])
        """
        if self.is_cloud:
            return self._request(
                "POST",
                "/v1/tool-execution/preflight",
                _config_body(config, config_text, repetitions),
            )
        _config_body(config, config_text, repetitions)
        return self.local.preflight(_config_value(config, config_text), repetitions)

    def test_mock(
        self,
        *,
        tool: str,
        binding: Mapping[str, Any],
        arguments: Optional[Mapping[str, Any]] = None,
        occurrence: int = 1,
        seed: int = 0,
        prior_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one mock binding in isolation; nothing is persisted and no
        live adapter is touched.

        Example:
            >>> from agenomic import Client
            >>> out = Client().tools.test_mock(tool="weather.now", binding={"mode": "mock",
            ...     "strategy": "rules", "rules": [{"id": "paris", "when": {"args_match": {"city": "Paris"}},
            ...     "then": {"kind": "structured", "data": {"temp_c": 18}}}]}, arguments={"city": "Paris"})
            >>> out["result"], out["agenomic"]["provenance"]["rule_id"]
            ({'temp_c': 18}, 'paris')
        """
        body: dict[str, Any] = {
            "tool": tool,
            "binding": dict(binding),
            "arguments": dict(arguments or {}),
            "occurrence": occurrence,
            "seed": seed,
        }
        if prior_state is not None:
            body["prior_state"] = dict(prior_state)
        if self.is_cloud:
            return self._request("POST", "/v1/tool-execution/mock/test", body)
        return self.local.mock_outcome(tool, dict(binding), dict(arguments or {}), occurrence)

    def test_connection(
        self,
        *,
        profile: str,
        tool: str,
        binding: Mapping[str, Any],
        allowed_env: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Read-only connection test of a live binding (cloud only: it opens a network connection).

        Example:
            >>> from agenomic import Client
            >>> Client().tools.test_connection(profile="p", tool="t", binding={"mode": "live", "adapter": "http"})
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: cloud_required: connection tests open a network connection and need Agenomic Cloud
        """
        if not self.is_cloud:
            raise ToolExecutionError(
                "cloud_required",
                "connection tests open a network connection and need Agenomic Cloud",
                0,
            )
        body: dict[str, Any] = {"profile": profile, "tool": tool, "binding": dict(binding)}
        if allowed_env is not None:
            body["allowed_env"] = list(allowed_env)
        return self._request("POST", "/v1/tool-execution/connection/test", body)

    # ── runs ──────────────────────────────────────────────────────────

    def create_run(
        self,
        *,
        name: str = "tool run",
        config: Optional[Mapping[str, Any]] = None,
        config_text: Optional[str] = None,
        repetitions: int = 1,
        replay_job_id: Optional[str] = None,
        rmp_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a run from a runnable plan; ``planned`` when approval is required, else ``approved``.

        Example:
            >>> from agenomic import Client
            >>> run = Client().tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {}})
            >>> run["status"]
            'approved'
        """
        body = _config_body(config, config_text, repetitions)
        body["name"] = name
        if replay_job_id:
            body["replay_job_id"] = replay_job_id
        if rmp_session_id:
            body["rmp_session_id"] = rmp_session_id
        if self.is_cloud:
            return _unwrap_run(self._request("POST", "/v1/tool-execution/runs", body))
        links: dict[str, Any] = {
            k: body[k] for k in ("replay_job_id", "rmp_session_id") if k in body
        }
        return self.local.create_run(name, _config_value(config, config_text), repetitions, links)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch one run with its frozen plan and approval.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.get_run(run["id"])["plan_hash"] == run["plan_hash"]
            True
        """
        if self.is_cloud:
            return _unwrap_run(self._request("GET", f"/v1/tool-execution/runs/{run_id}"))
        return self.local.get_run(run_id)

    def approve_run(self, run_id: str, *, plan_hash: str) -> dict[str, Any]:
        """Approve the exact plan version; a different hash is refused.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.approve_run(run["id"], plan_hash="sha256:other")
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: run_not_active: run ... is approved
        """
        if self.is_cloud:
            return _unwrap_run(
                self._request(
                    "POST", f"/v1/tool-execution/runs/{run_id}/approve", {"plan_hash": plan_hash}
                )
            )
        return self.local.approve_run(run_id, plan_hash)

    def start_run(self, run_id: str) -> dict[str, Any]:
        """Start an approved run.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.start_run(run["id"])["status"]
            'running'
        """
        if self.is_cloud:
            return _unwrap_run(self._request("POST", f"/v1/tool-execution/runs/{run_id}/start", {}))
        return self.local.start_run(run_id)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Cancel a run: new calls are refused from now on.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.cancel_run(run["id"])["status"]
            'cancelled'
        """
        if self.is_cloud:
            return _unwrap_run(
                self._request("POST", f"/v1/tool-execution/runs/{run_id}/cancel", {})
            )
        return self.local.cancel_run(run_id)

    def complete_run(
        self, run_id: str, *, failed: bool = False, error_message: Optional[str] = None
    ) -> dict[str, Any]:
        """Mark a running run completed (or failed).

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> _ = tools.start_run(run["id"])
            >>> tools.complete_run(run["id"])["status"]
            'completed'
        """
        body: dict[str, Any] = {"failed": failed}
        if error_message:
            body["error_message"] = error_message
        if self.is_cloud:
            return _unwrap_run(
                self._request("POST", f"/v1/tool-execution/runs/{run_id}/complete", body)
            )
        return self.local.complete_run(run_id, failed, error_message)

    def report(self, run_id: str) -> dict[str, Any]:
        """Report ventilated by provenance and status, with the real-call warning.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.report(run["id"])["report"]["has_real_calls"]
            False
        """
        if self.is_cloud:
            return self._request("GET", f"/v1/tool-execution/runs/{run_id}/report")
        return self.local.report(run_id)

    def export(self, run_id: str) -> dict[str, Any]:
        """Portable export of a run: plan, invocations and report.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> tools.export(run["id"])["export_version"]
            'agenomic.tool_run_export/v1'
        """
        if self.is_cloud:
            return self._request("GET", f"/v1/tool-execution/runs/{run_id}/export")
        return self.local.export(run_id)

    # ── per-call path (sync and async) ────────────────────────────────

    def _invoke_body(
        self,
        run_id: str,
        tool: str,
        arguments: Optional[Mapping[str, Any]],
        logical_call_id: Optional[str],
        repetition: int,
        attempt: int,
        parent_call_id: Optional[str],
        deadline_ms: Optional[int],
    ) -> tuple[dict[str, Any], str]:
        call_id = logical_call_id or f"call_{uuid.uuid4().hex}"
        body = _identity(
            tool,
            arguments or {},
            logical_call_id=call_id,
            repetition=repetition,
            attempt=attempt,
            parent_call_id=parent_call_id,
        )
        if deadline_ms is not None:
            body["deadline_ms"] = deadline_ms
        key = str(uuid.uuid5(_IDEMPOTENCY_NAMESPACE, f"{run_id}:{repetition}:{call_id}"))
        return body, key

    def invoke(
        self,
        run_id: str,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        logical_call_id: Optional[str] = None,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
        deadline_ms: Optional[int] = None,
    ) -> ToolCallResult:
        """Route one tool call (Tool Gateway in cloud mode, local engine otherwise).

        A Protect run answers 202 for a call waiting for approval: the result
        comes back with ``status == "pending"`` and an ``approval_id``. A 403
        carrying the invoke envelope raises :class:`ToolCallDenied`.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> out = tools.invoke(run["id"], "email.send", {"to": "a@example.test"})
            >>> out.result, out.source
            ({'delivered': True}, 'static')
        """
        body, key = self._invoke_body(
            run_id,
            tool,
            arguments,
            logical_call_id,
            repetition,
            attempt,
            parent_call_id,
            deadline_ms,
        )
        if self.is_cloud:
            path = f"/v1/tool-execution/runs/{run_id}/invoke"
            response = send_request(self._client, "POST", path, body, idempotency_key=key)
            return _parse_invoke(tool, "POST", path, response)
        return ToolCallResult.from_response(self.local.invoke(run_id, body))

    async def ainvoke(
        self,
        run_id: str,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        logical_call_id: Optional[str] = None,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
        deadline_ms: Optional[int] = None,
    ) -> ToolCallResult:
        """Async counterpart of :meth:`invoke` for asyncio runtimes.

        Example:
            >>> import asyncio
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> asyncio.run(tools.ainvoke(run["id"], "email.send")).result
            {'delivered': True}
        """
        body, key = self._invoke_body(
            run_id,
            tool,
            arguments,
            logical_call_id,
            repetition,
            attempt,
            parent_call_id,
            deadline_ms,
        )
        if self.is_cloud:
            path = f"/v1/tool-execution/runs/{run_id}/invoke"
            response = await asend_request(self._client, "POST", path, body, idempotency_key=key)
            return _parse_invoke(tool, "POST", path, response)
        return ToolCallResult.from_response(self.local.invoke(run_id, body))

    def authorize_local(
        self,
        run_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        logical_call_id: str,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask whether a local function may run for this call, before running it.

        Returns ``{"decision": "local", "record_id": ...}`` once the budget is
        reserved and a pending record exists, or ``{"decision": "gateway"}``
        when the run binds the tool to a mock or a non-local adapter. A
        Protect run may answer ``pending`` (with ``approval_id``) or
        ``denied`` (with ``protect``), and a ``local`` decision then carries
        the signed ``permit`` that :meth:`report_local` must present. Any
        refusal (inactive run, denied write, exhausted budget) raises before
        anything executes.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> tools.authorize_local(run["id"], "email.send", {}, logical_call_id="c1")
            {'decision': 'gateway'}
        """
        body = _identity(
            tool,
            arguments,
            logical_call_id=logical_call_id,
            repetition=repetition,
            attempt=attempt,
            parent_call_id=parent_call_id,
        )
        if self.is_cloud:
            path = f"/v1/tool-execution/runs/{run_id}/local/authorize"
            return _parse_authorize("POST", path, send_request(self._client, "POST", path, body))
        return self.local.authorize_local(run_id, body)

    async def aauthorize_local(
        self,
        run_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        logical_call_id: str,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Async counterpart of :meth:`authorize_local`.

        Example:
            >>> import asyncio
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> asyncio.run(tools.aauthorize_local(run["id"], "email.send", {}, logical_call_id="c1"))
            {'decision': 'gateway'}
        """
        body = _identity(
            tool,
            arguments,
            logical_call_id=logical_call_id,
            repetition=repetition,
            attempt=attempt,
            parent_call_id=parent_call_id,
        )
        if self.is_cloud:
            path = f"/v1/tool-execution/runs/{run_id}/local/authorize"
            return _parse_authorize(
                "POST", path, await asend_request(self._client, "POST", path, body)
            )
        return self.local.authorize_local(run_id, body)

    def _report_body(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        result: Any,
        logical_call_id: str,
        is_error: bool,
        duration_ms: int,
        repetition: int,
        attempt: int,
        parent_call_id: Optional[str],
        permit: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        body = _identity(
            tool,
            arguments,
            logical_call_id=logical_call_id,
            repetition=repetition,
            attempt=attempt,
            parent_call_id=parent_call_id,
        )
        body.update({"result": result, "is_error": is_error, "duration_ms": duration_ms})
        if permit is not None:
            body["permit"] = dict(permit)
        return body

    def report_local(
        self,
        run_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        logical_call_id: str,
        is_error: bool = False,
        duration_ms: int = 0,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
        permit: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Settle a call previously accepted by :meth:`authorize_local`.

        ``permit`` is the signed permit returned by the authorize step of a
        Protect run; it is sent verbatim and the gateway refuses the report
        without it.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1", "mode": "mock"})
            >>> _ = tools.start_run(run["id"])
            >>> tools.report_local(run["id"], "echo", {}, {"ok": True}, logical_call_id="c1")
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: live_call_denied: runtime-local call was not authorized; call authorize_local before executing
        """
        body = self._report_body(
            tool,
            arguments,
            result,
            logical_call_id,
            is_error,
            duration_ms,
            repetition,
            attempt,
            parent_call_id,
            permit,
        )
        if self.is_cloud:
            response = self._request("POST", f"/v1/tool-execution/runs/{run_id}/report-local", body)
        else:
            response = self.local.report_local(run_id, body, result, is_error, duration_ms)
        return str(response.get("record_id", ""))

    async def areport_local(
        self,
        run_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        logical_call_id: str,
        is_error: bool = False,
        duration_ms: int = 0,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
        permit: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Async counterpart of :meth:`report_local`.

        Example:
            >>> from agenomic import Client
            >>> callable(Client().tools.areport_local)
            True
        """
        body = self._report_body(
            tool,
            arguments,
            result,
            logical_call_id,
            is_error,
            duration_ms,
            repetition,
            attempt,
            parent_call_id,
            permit,
        )
        if self.is_cloud:
            response = await self._arequest(
                "POST", f"/v1/tool-execution/runs/{run_id}/report-local", body
            )
        else:
            response = self.local.report_local(run_id, body, result, is_error, duration_ms)
        return str(response.get("record_id", ""))

    def router(
        self,
        run_id: str,
        *,
        repetition: int = 1,
        local_functions: Optional[Mapping[str, LocalFunction]] = None,
        raise_on_error: bool = True,
        before_action: Optional[BeforeAction] = None,
    ) -> ToolRouter:
        """A synchronous router bound to one run.

        ``before_action`` receives the identity dict of every call (tool,
        arguments, logical_call_id, repetition, attempt, parent_call_id)
        before any request is sent; raising aborts the call, its return value
        is ignored. A coroutine hook cannot run here and raises
        ``ToolExecutionError("invalid_hook", ...)``; pass it to
        :meth:`arouter` instead.

        Example:
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> tools.router(run["id"]).call("email.send", {"to": "a@example.test"})
            {'delivered': True}
        """
        return ToolRouter(
            self,
            run_id,
            repetition=repetition,
            local_functions=local_functions,
            raise_on_error=raise_on_error,
            before_action=before_action,
        )

    def arouter(
        self,
        run_id: str,
        *,
        repetition: int = 1,
        local_functions: Optional[Mapping[str, LocalFunction]] = None,
        raise_on_error: bool = True,
        before_action: Optional[BeforeAction] = None,
    ) -> AsyncToolRouter:
        """An asyncio router bound to one run; local functions may be coroutines.

        ``before_action`` may be a coroutine function: an awaitable return is
        awaited before the call is admitted.

        Example:
            >>> import asyncio
            >>> from agenomic import Client
            >>> tools = Client().tools
            >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
            ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
            ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
            >>> _ = tools.start_run(run["id"])
            >>> asyncio.run(tools.arouter(run["id"]).call("email.send"))
            {'delivered': True}
        """
        return AsyncToolRouter(
            self,
            run_id,
            repetition=repetition,
            local_functions=local_functions,
            raise_on_error=raise_on_error,
            before_action=before_action,
        )


def _local_envelope(value: Any, is_error: bool, record_id: str, duration: int) -> ToolCallResult:
    return ToolCallResult(
        result={"error": value} if is_error else value,
        record_id=record_id,
        status="error" if is_error else "success",
        provenance=ToolProvenance(source="runtime_local", fidelity="live", binding_mode="live"),
        external_state="confirmed",
        duration_ms=duration,
    )


def _record_id_of(tool: str, decision: Mapping[str, Any]) -> str:
    record_id = str(decision.get("record_id") or "")
    if not record_id:
        raise ToolExecutionError(
            "invalid_response",
            f"local/authorize accepted {tool} without a record_id; refusing to execute",
            0,
        )
    return record_id


def _decision_envelope(
    decision: Mapping[str, Any], status: Literal["pending", "denied"]
) -> ToolCallResult:
    protect = decision.get("protect")
    approval_id = decision.get("approval_id")
    return ToolCallResult(
        record_id=str(decision.get("record_id") or ""),
        status=status,
        provenance=ToolProvenance(source="unrouted", binding_mode="live"),
        external_state="none",
        protect=ProtectDecision.model_validate(protect) if isinstance(protect, Mapping) else None,
        approval_id=str(approval_id) if approval_id else None,
        decision=str(decision.get("decision")) if decision.get("decision") is not None else None,
    )


def _approval_status(record: Mapping[str, Any]) -> str:
    return str(record.get("status") or "")


@dataclass(frozen=True)
class _PendingCall:
    tool: str
    arguments: dict[str, Any]
    logical_call_id: str
    attempt: int
    parent_call_id: Optional[str]
    envelope: ToolCallResult


class _RouterBase:
    def __init__(
        self,
        tools: ToolsResource,
        run_id: str,
        *,
        repetition: int,
        local_functions: Optional[Mapping[str, LocalFunction]],
        raise_on_error: bool,
        before_action: Optional[BeforeAction] = None,
    ) -> None:
        self._tools = tools
        self.run_id = run_id
        self.repetition = repetition
        self._local = dict(local_functions or {})
        self._raise = raise_on_error
        self._before_action = before_action
        self._sequence = 0
        self._pending: dict[str, _PendingCall] = {}
        self.calls: list[ToolCallResult] = []

    def _next_call_id(self, tool: str) -> str:
        self._sequence += 1
        return f"{tool}#{self._sequence}"

    def _prepare(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]],
        logical_call_id: Optional[str],
        attempt: int,
        parent_call_id: Optional[str],
    ) -> tuple[dict[str, Any], str, Any]:
        args = dict(arguments or {})
        call_id = logical_call_id or self._next_call_id(tool)
        if self._before_action is None:
            return args, call_id, None
        return (
            args,
            call_id,
            self._before_action(
                _identity(
                    tool,
                    args,
                    logical_call_id=call_id,
                    repetition=self.repetition,
                    attempt=attempt,
                    parent_call_id=parent_call_id,
                )
            ),
        )

    def _begin(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]],
        logical_call_id: Optional[str],
        attempt: int,
        parent_call_id: Optional[str],
    ) -> tuple[dict[str, Any], str]:
        args, call_id, outcome = self._prepare(
            tool, arguments, logical_call_id, attempt, parent_call_id
        )
        if inspect.isawaitable(outcome):
            if inspect.iscoroutine(outcome):
                outcome.close()
            raise ToolExecutionError(
                "invalid_hook",
                "before_action returned an awaitable on the synchronous router; "
                f"{tool} was not admitted because the hook could not run",
                0,
            )
        return args, call_id

    async def _abegin(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]],
        logical_call_id: Optional[str],
        attempt: int,
        parent_call_id: Optional[str],
    ) -> tuple[dict[str, Any], str]:
        args, call_id, outcome = self._prepare(
            tool, arguments, logical_call_id, attempt, parent_call_id
        )
        if inspect.isawaitable(outcome):
            await outcome
        return args, call_id

    def _hold(
        self,
        tool: str,
        args: dict[str, Any],
        call_id: str,
        attempt: int,
        parent_call_id: Optional[str],
        envelope: ToolCallResult,
    ) -> ToolApprovalPending:
        self.calls.append(envelope)
        approval_id = envelope.approval_id or ""
        if approval_id:
            self._pending[approval_id] = _PendingCall(
                tool, dict(args), call_id, attempt, parent_call_id, envelope
            )
        return ToolApprovalPending(tool, approval_id, envelope.record_id, envelope)

    def _refuse(self, tool: str, envelope: ToolCallResult) -> ToolCallDenied:
        self.calls.append(envelope)
        return ToolCallDenied(tool, envelope)

    def _pending_call(self, approval: Union[ToolApprovalPending, str]) -> tuple[str, _PendingCall]:
        approval_id = (
            approval.approval_id if isinstance(approval, ToolApprovalPending) else str(approval)
        )
        pending = self._pending.get(approval_id)
        if pending is None:
            raise ToolExecutionError(
                "approval_unknown",
                f"this router holds no pending call for approval {approval_id!r}",
                0,
            )
        return approval_id, pending

    @staticmethod
    def _replay_error(
        approval_id: str, pending: _PendingCall, status: str, error: ToolExecutionError
    ) -> ToolExecutionError:
        if status == "consumed" and error.status == 409:
            return ToolExecutionError(
                "conflict",
                f"approval {approval_id} is consumed and {pending.tool} already executed; "
                "the gateway refused to replay its result",
                409,
            )
        return error

    def _granted(self, pending: _PendingCall, status: str, deadline: float) -> bool:
        if status in _APPROVAL_GRANTED:
            return True
        if status != "pending":
            envelope = pending.envelope.model_copy(update={"status": "denied"})
            self.calls.append(envelope)
            raise ToolCallDenied(pending.tool, envelope, code=status or "policy_denied")
        if time.monotonic() >= deadline:
            raise ToolExecutionError(
                "approval_timeout",
                f"approval for {pending.tool} was still pending after the timeout",
                0,
            )
        return False

    @property
    def has_real_calls(self) -> bool:
        return any(c.is_real for c in self.calls)

    def summary(self) -> dict[str, Any]:
        """Counts by source plus the real-call and unreported flags.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.router("run").summary()
            {'calls': 0, 'by_source': {}, 'has_real_calls': False, 'unreported': 0}
        """
        by_source: dict[str, int] = {}
        for c in self.calls:
            by_source[c.source] = by_source.get(c.source, 0) + 1
        return {
            "calls": len(self.calls),
            "by_source": by_source,
            "has_real_calls": self.has_real_calls,
            "unreported": sum(1 for c in self.calls if not c.reported),
        }

    def _finish(self, tool: str, envelope: ToolCallResult) -> Any:
        self.calls.append(envelope)
        if self._raise and not envelope.ok:
            raise ToolCallError(tool, envelope)
        return envelope.result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(run_id={self.run_id!r}, calls={len(self.calls)})"


class ToolRouter(_RouterBase):
    """Synchronous runtime-side wrapper around one run.

    ``call`` returns the tool's native result so existing agent code keeps its
    shape. Tools listed in ``local_functions`` run in this process only after
    the engine authorized them; every other tool is routed to the engine.

    Example:
        >>> from agenomic import Client
        >>> tools = Client().tools
        >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
        ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
        ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
        >>> _ = tools.start_run(run["id"])
        >>> router = tools.router(run["id"])
        >>> router.call("email.send", {"to": "a@example.test"})
        {'delivered': True}
        >>> router.summary()["by_source"]
        {'static': 1}
    """

    def call(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        parent_call_id: Optional[str] = None,
        attempt: int = 1,
        logical_call_id: Optional[str] = None,
    ) -> Any:
        """Route one call and return the tool's native result.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.router("missing").call("email.send")
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: not_found: tool run not found
        """
        args, call_id = self._begin(tool, arguments, logical_call_id, attempt, parent_call_id)
        if tool in self._local:
            decision = self._tools.authorize_local(
                self.run_id,
                tool,
                args,
                logical_call_id=call_id,
                repetition=self.repetition,
                attempt=attempt,
                parent_call_id=parent_call_id,
            )
            kind = decision.get("decision")
            if kind == "local":
                record_id = _record_id_of(tool, decision)
                started = time.monotonic()
                try:
                    value = self._local[tool](**args)
                    is_error = False
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
                    value = {"code": type(exc).__name__, "message": str(exc)}
                    is_error = True
                duration = int((time.monotonic() - started) * 1000)
                envelope = _local_envelope(value, is_error, record_id, duration)
                try:
                    self._tools.report_local(
                        self.run_id,
                        tool,
                        args,
                        value,
                        is_error=is_error,
                        duration_ms=duration,
                        logical_call_id=call_id,
                        repetition=self.repetition,
                        attempt=attempt,
                        parent_call_id=parent_call_id,
                        permit=decision.get("permit"),
                    )
                except Exception:
                    envelope.reported = False
                    envelope.external_state = "indeterminate"
                    self.calls.append(envelope)
                    raise
                return self._finish(tool, envelope)
            if kind == "pending":
                pending = _decision_envelope(decision, "pending")
                raise self._hold(tool, args, call_id, attempt, parent_call_id, pending)
            if kind != "gateway":
                raise self._refuse(tool, _decision_envelope(decision, "denied"))
        try:
            envelope = self._tools.invoke(
                self.run_id,
                tool,
                args,
                logical_call_id=call_id,
                repetition=self.repetition,
                attempt=attempt,
                parent_call_id=parent_call_id,
            )
        except ToolCallDenied as denied:
            raise self._refuse(tool, denied.envelope) from None
        if envelope.status == "pending":
            raise self._hold(tool, args, call_id, attempt, parent_call_id, envelope)
        if envelope.status == "denied":
            raise self._refuse(tool, envelope)
        return self._finish(tool, envelope)

    def resume(
        self,
        approval: Union[ToolApprovalPending, str],
        *,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> Any:
        """Wait for an approval, then re-issue the identical call once granted.

        Polls ``GET /v1/protect/approvals/{id}``. ``approved`` and ``consumed``
        both re-issue the identical identity once, with the original
        Idempotency-Key: a consumed approval already executed, so the replay
        recovers its result. A 409 on that replay raises
        :class:`ToolExecutionError` with code ``conflict``. ``rejected``,
        ``expired`` or any other terminal status raises
        :class:`ToolCallDenied` with that status as ``code``;
        ``approval_timeout`` is raised when the approval is still pending
        after ``timeout`` seconds. The re-issued call keeps the same tool,
        arguments, logical_call_id, attempt and parent, so the gateway
        resumes the pending claim exactly once.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.router("run").resume("apr_unknown")
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: approval_unknown: this router holds no pending call for approval 'apr_unknown'
        """
        approval_id, pending = self._pending_call(approval)
        deadline = time.monotonic() + timeout
        status = ""
        while True:
            record = self._tools._request("GET", f"/v1/protect/approvals/{approval_id}")
            status = _approval_status(record)
            if self._granted(pending, status, deadline):
                break
            time.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))
        del self._pending[approval_id]
        try:
            return self.call(
                pending.tool,
                pending.arguments,
                parent_call_id=pending.parent_call_id,
                attempt=pending.attempt,
                logical_call_id=pending.logical_call_id,
            )
        except ToolExecutionError as error:
            raise self._replay_error(approval_id, pending, status, error) from None

    def wrap(self, tool: str) -> Callable[..., Any]:
        """Return a callable ``fn(**arguments)`` routed through this run.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.router("run").wrap("email.send").__name__
            'email_send'
        """

        def routed(**arguments: Any) -> Any:
            return self.call(tool, arguments)

        routed.__name__ = tool.replace(".", "_")
        return routed


class AsyncToolRouter(_RouterBase):
    """asyncio runtime-side wrapper around one run.

    Local functions may be plain callables or coroutine functions; the
    per-call I/O awaits :meth:`ToolsResource.ainvoke` and its siblings so
    the event loop is never blocked by a gateway round-trip.

    Example:
        >>> import asyncio
        >>> from agenomic import Client
        >>> tools = Client().tools
        >>> run = tools.create_run(config={"schema_version": "agenomic.tool_execution/v1",
        ...     "mode": "mock", "bindings": {"email.send": {"mode": "mock", "strategy": "static",
        ...     "response": {"kind": "structured", "data": {"delivered": True}}}}})
        >>> _ = tools.start_run(run["id"])
        >>> asyncio.run(tools.arouter(run["id"]).call("email.send"))
        {'delivered': True}
    """

    async def call(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        parent_call_id: Optional[str] = None,
        attempt: int = 1,
        logical_call_id: Optional[str] = None,
    ) -> Any:
        """Route one call and return the tool's native result.

        Example:
            >>> import asyncio
            >>> from agenomic import Client
            >>> asyncio.run(Client().tools.arouter("missing").call("email.send"))
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: not_found: tool run not found
        """
        args, call_id = await self._abegin(
            tool, arguments, logical_call_id, attempt, parent_call_id
        )
        if tool in self._local:
            decision = await self._tools.aauthorize_local(
                self.run_id,
                tool,
                args,
                logical_call_id=call_id,
                repetition=self.repetition,
                attempt=attempt,
                parent_call_id=parent_call_id,
            )
            kind = decision.get("decision")
            if kind == "local":
                record_id = _record_id_of(tool, decision)
                started = time.monotonic()
                try:
                    produced: Union[Any, Awaitable[Any]] = self._local[tool](**args)
                    value = await produced if inspect.isawaitable(produced) else produced
                    is_error = False
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
                    value = {"code": type(exc).__name__, "message": str(exc)}
                    is_error = True
                duration = int((time.monotonic() - started) * 1000)
                envelope = _local_envelope(value, is_error, record_id, duration)
                try:
                    await self._tools.areport_local(
                        self.run_id,
                        tool,
                        args,
                        value,
                        is_error=is_error,
                        duration_ms=duration,
                        logical_call_id=call_id,
                        repetition=self.repetition,
                        attempt=attempt,
                        parent_call_id=parent_call_id,
                        permit=decision.get("permit"),
                    )
                except Exception:
                    envelope.reported = False
                    envelope.external_state = "indeterminate"
                    self.calls.append(envelope)
                    raise
                return self._finish(tool, envelope)
            if kind == "pending":
                pending = _decision_envelope(decision, "pending")
                raise self._hold(tool, args, call_id, attempt, parent_call_id, pending)
            if kind != "gateway":
                raise self._refuse(tool, _decision_envelope(decision, "denied"))
        try:
            envelope = await self._tools.ainvoke(
                self.run_id,
                tool,
                args,
                logical_call_id=call_id,
                repetition=self.repetition,
                attempt=attempt,
                parent_call_id=parent_call_id,
            )
        except ToolCallDenied as denied:
            raise self._refuse(tool, denied.envelope) from None
        if envelope.status == "pending":
            raise self._hold(tool, args, call_id, attempt, parent_call_id, envelope)
        if envelope.status == "denied":
            raise self._refuse(tool, envelope)
        return self._finish(tool, envelope)

    async def resume(
        self,
        approval: Union[ToolApprovalPending, str],
        *,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> Any:
        """Async counterpart of :meth:`ToolRouter.resume` (waits with ``asyncio.sleep``).

        Example:
            >>> import asyncio
            >>> from agenomic import Client
            >>> asyncio.run(Client().tools.arouter("run").resume("apr_unknown"))
            Traceback (most recent call last):
            ...
            agenomic.tools.models.ToolExecutionError: approval_unknown: this router holds no pending call for approval 'apr_unknown'
        """
        approval_id, pending = self._pending_call(approval)
        deadline = time.monotonic() + timeout
        status = ""
        while True:
            record = await self._tools._arequest("GET", f"/v1/protect/approvals/{approval_id}")
            status = _approval_status(record)
            if self._granted(pending, status, deadline):
                break
            await asyncio.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))
        del self._pending[approval_id]
        try:
            return await self.call(
                pending.tool,
                pending.arguments,
                parent_call_id=pending.parent_call_id,
                attempt=pending.attempt,
                logical_call_id=pending.logical_call_id,
            )
        except ToolExecutionError as error:
            raise self._replay_error(approval_id, pending, status, error) from None

    def wrap(self, tool: str) -> Callable[..., Awaitable[Any]]:
        """Return an ``async fn(**arguments)`` routed through this run.

        Example:
            >>> from agenomic import Client
            >>> Client().tools.arouter("run").wrap("email.send").__name__
            'email_send'
        """

        async def routed(**arguments: Any) -> Any:
            return await self.call(tool, arguments)

        routed.__name__ = tool.replace(".", "_")
        return routed


def dumps_config(config: Mapping[str, Any]) -> str:
    """Serialize a configuration block to JSON text (accepted as ``config_text``).

    Example:
        >>> dumps_config({"mode": "mock", "schema_version": "agenomic.tool_execution/v1"})
        '{"mode": "mock", "schema_version": "agenomic.tool_execution/v1"}'
    """
    return json.dumps(dict(config), sort_keys=True)
