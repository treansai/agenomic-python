"""``client.tools``: profiles, fixtures, scenarios, runs and the invoke path."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import httpx

from agenomic.exceptions import CloudError

TOOL_EXECUTION_SCHEMA_VERSION = "agenomic.tool_execution/v1"

_IDEMPOTENCY_NAMESPACE = uuid.UUID("6b1f7a2e-9c44-4d0e-8f1a-2e7c0d9b5a31")


class ToolExecutionError(CloudError):
    """A tool-execution API call was refused.

    ``code`` carries the server error code (for example ``mock_unmatched``,
    ``tool_unknown``, ``live_call_denied``, ``plan_approval_required``,
    ``env_reference_missing``); ``status`` the HTTP status.
    """

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.status = status


@dataclass
class ToolCallResult:
    """Native tool result plus the Agenomic technical envelope."""

    result: Any
    record_id: str
    status: str
    provenance: dict[str, Any]
    external_state: str
    effects: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    virtual_time: Optional[str] = None
    expected_error: bool = False

    @property
    def source(self) -> str:
        return str(self.provenance.get("source", ""))

    @property
    def is_real(self) -> bool:
        return self.source in {"live", "runtime_local"}

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @classmethod
    def from_response(cls, body: Mapping[str, Any]) -> ToolCallResult:
        env = body.get("agenomic")
        if not isinstance(env, dict) or "record_id" not in env:
            raise CloudError("invoke response did not include an agenomic envelope")
        return cls(
            result=body.get("result"),
            record_id=str(env["record_id"]),
            status=str(env.get("status", "")),
            provenance=dict(env.get("provenance") or {}),
            external_state=str(env.get("external_state", "")),
            effects=list(env.get("effects") or []),
            duration_ms=int(env.get("duration_ms") or 0),
            virtual_time=env.get("virtual_time"),
            expected_error=bool(env.get("expected_error", False)),
        )


class ToolCallError(CloudError):
    """The tool answered with an error outcome (business, protocol, timeout)."""

    def __init__(self, tool: str, envelope: ToolCallResult) -> None:
        detail = envelope.result.get("error") if isinstance(envelope.result, dict) else None
        code = detail.get("code") if isinstance(detail, dict) else envelope.status
        super().__init__(f"tool {tool} returned {envelope.status} ({code})")
        self.tool = tool
        self.envelope = envelope
        self.code = str(code)


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


class ToolsResource:
    """The ``client.tools`` namespace. Cloud mode only.

    Example:
        >>> from agenomic import Client
        >>> tools = Client(api_key="k", base_url="https://cloud.example").tools
        >>> tools.schema_version
        'agenomic.tool_execution/v1'
    """

    schema_version = TOOL_EXECUTION_SCHEMA_VERSION

    def __init__(self, client: Any) -> None:
        self._client = client

    # ── transport ─────────────────────────────────────────────────────

    def _require_cloud(self) -> None:
        if not getattr(self._client, "is_cloud", False):
            raise ToolExecutionError(
                "cloud_required",
                "tool execution requires a cloud client (base_url); there is no local fallback",
                0,
            )

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        self._require_cloud()
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            with self._client._http() as http:
                response = http.request(method, path, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                "transport_error", f"{method} {path} failed: {exc}", 0
            ) from exc
        if response.status_code >= 400:
            code, message = "http_error", f"{method} {path} returned {response.status_code}"
            try:
                error = response.json().get("error")
                if isinstance(error, dict):
                    code = str(error.get("code", code))
                    message = str(error.get("message", message))
            except (ValueError, AttributeError):
                pass
            raise ToolExecutionError(code, message, response.status_code)
        if not response.content:
            return {}
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    # ── profiles and variables ────────────────────────────────────────

    def create_profile(
        self, *, name: str, environment: str = "dev", allowed_env: Optional[list[str]] = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/tool-execution/profiles",
            {"name": name, "environment": environment, "allowed_env": list(allowed_env or [])},
        )

    def list_profiles(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/tool-execution/profiles").get("profiles", []))

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/tool-execution/profiles/{profile_id}")

    def set_variable(self, profile_id: str, name: str, value: str) -> dict[str, Any]:
        """Write-only: the value is encrypted server-side and never returned."""
        return self._request(
            "PUT", f"/v1/tool-execution/profiles/{profile_id}/variables/{name}", {"value": value}
        )

    def variable_status(self, profile_id: str) -> list[dict[str, Any]]:
        """Availability per allowed variable. Never carries values."""
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
        body: dict[str, Any] = {"name": name, "version": version, "effect": effect}
        if input_schema is not None:
            body["input_schema"] = dict(input_schema)
        if output_schema is not None:
            body["output_schema"] = dict(output_schema)
        if description:
            body["description"] = description
        return self._request("POST", "/v1/tool-execution/contracts", body)

    def create_fixture_set(
        self, *, name: str, version: int, fixtures: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/tool-execution/fixture-sets",
            {"name": name, "version": version, "fixtures": [dict(f) for f in fixtures]},
        )

    def approve_fixture_set(self, fixture_set_id: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/tool-execution/fixture-sets/{fixture_set_id}/approve", {}
        )

    def create_scenario(
        self,
        *,
        name: str,
        version: int,
        entities: Mapping[str, Any],
        tools: Mapping[str, Any],
        initial_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/tool-execution/scenarios",
            {
                "name": name,
                "version": version,
                "entities": dict(entities),
                "tools": dict(tools),
                "initial_state": dict(initial_state or {}),
            },
        )

    def adapters(self) -> dict[str, Any]:
        """Capability matrix of the adapters the gateway implements."""
        return self._request("GET", "/v1/tool-execution/adapters")

    # ── configuration ─────────────────────────────────────────────────

    def validate(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_text: Optional[str] = None,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        """Structural validation only: no store lookups, no network."""
        return self._request(
            "POST", "/v1/tool-execution/validate", _config_body(config, config_text, repetitions)
        )

    def preflight(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_text: Optional[str] = None,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        """Immutable execution plan: live vs mock tools, destinations, missing
        variables, possible effects and the ``plan_hash`` an approval binds to."""
        return self._request(
            "POST", "/v1/tool-execution/preflight", _config_body(config, config_text, repetitions)
        )

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
        live adapter is touched."""
        body: dict[str, Any] = {
            "tool": tool,
            "binding": dict(binding),
            "arguments": dict(arguments or {}),
            "occurrence": occurrence,
            "seed": seed,
        }
        if prior_state is not None:
            body["prior_state"] = dict(prior_state)
        return self._request("POST", "/v1/tool-execution/mock/test", body)

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
        body = _config_body(config, config_text, repetitions)
        body["name"] = name
        if replay_job_id:
            body["replay_job_id"] = replay_job_id
        if rmp_session_id:
            body["rmp_session_id"] = rmp_session_id
        return _unwrap_run(self._request("POST", "/v1/tool-execution/runs", body))

    def get_run(self, run_id: str) -> dict[str, Any]:
        return _unwrap_run(self._request("GET", f"/v1/tool-execution/runs/{run_id}"))

    def approve_run(self, run_id: str, *, plan_hash: str) -> dict[str, Any]:
        """Approve the exact plan version; a changed plan invalidates it."""
        return _unwrap_run(
            self._request(
                "POST", f"/v1/tool-execution/runs/{run_id}/approve", {"plan_hash": plan_hash}
            )
        )

    def start_run(self, run_id: str) -> dict[str, Any]:
        return _unwrap_run(self._request("POST", f"/v1/tool-execution/runs/{run_id}/start", {}))

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return _unwrap_run(self._request("POST", f"/v1/tool-execution/runs/{run_id}/cancel", {}))

    def complete_run(
        self, run_id: str, *, failed: bool = False, error_message: Optional[str] = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"failed": failed}
        if error_message:
            body["error_message"] = error_message
        return _unwrap_run(
            self._request("POST", f"/v1/tool-execution/runs/{run_id}/complete", body)
        )

    def report(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/tool-execution/runs/{run_id}/report")

    def export(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/tool-execution/runs/{run_id}/export")

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
        """Route one tool call through the Tool Gateway."""
        call_id = logical_call_id or f"call_{uuid.uuid4().hex}"
        body: dict[str, Any] = {
            "repetition": repetition,
            "logical_call_id": call_id,
            "attempt": attempt,
            "tool": tool,
            "arguments": dict(arguments or {}),
        }
        if parent_call_id:
            body["parent_call_id"] = parent_call_id
        if deadline_ms is not None:
            body["deadline_ms"] = deadline_ms
        key = str(uuid.uuid5(_IDEMPOTENCY_NAMESPACE, f"{run_id}:{repetition}:{call_id}"))
        response = self._request(
            "POST", f"/v1/tool-execution/runs/{run_id}/invoke", body, idempotency_key=key
        )
        return ToolCallResult.from_response(response)

    def report_local(
        self,
        run_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        is_error: bool = False,
        duration_ms: int = 0,
        logical_call_id: Optional[str] = None,
        repetition: int = 1,
        attempt: int = 1,
        parent_call_id: Optional[str] = None,
    ) -> str:
        """Record a call the runtime executed itself (local adapter)."""
        call_id = logical_call_id or f"call_{uuid.uuid4().hex}"
        body: dict[str, Any] = {
            "repetition": repetition,
            "logical_call_id": call_id,
            "attempt": attempt,
            "tool": tool,
            "arguments": dict(arguments),
            "result": result,
            "is_error": is_error,
            "duration_ms": duration_ms,
        }
        if parent_call_id:
            body["parent_call_id"] = parent_call_id
        response = self._request("POST", f"/v1/tool-execution/runs/{run_id}/report-local", body)
        return str(response.get("record_id", ""))

    def router(
        self,
        run_id: str,
        *,
        repetition: int = 1,
        local_functions: Optional[Mapping[str, Callable[..., Any]]] = None,
        raise_on_error: bool = True,
    ) -> ToolRouter:
        return ToolRouter(
            self,
            run_id,
            repetition=repetition,
            local_functions=local_functions,
            raise_on_error=raise_on_error,
        )


class ToolRouter:
    """Lightweight runtime-side wrapper around one run.

    ``call`` returns the tool's native result so existing agent code keeps its
    shape. Tools listed in ``local_functions`` run in this process and are
    reported to the gateway; every other tool is routed by the cloud.

    Example:
        >>> ToolRouter.__name__
        'ToolRouter'
    """

    def __init__(
        self,
        tools: ToolsResource,
        run_id: str,
        *,
        repetition: int = 1,
        local_functions: Optional[Mapping[str, Callable[..., Any]]] = None,
        raise_on_error: bool = True,
    ) -> None:
        self._tools = tools
        self.run_id = run_id
        self.repetition = repetition
        self._local = dict(local_functions or {})
        self._raise = raise_on_error
        self._sequence = 0
        self.calls: list[ToolCallResult] = []

    def _next_call_id(self, tool: str) -> str:
        self._sequence += 1
        return f"{tool}#{self._sequence}"

    def call(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        parent_call_id: Optional[str] = None,
        attempt: int = 1,
        logical_call_id: Optional[str] = None,
    ) -> Any:
        args = dict(arguments or {})
        call_id = logical_call_id or self._next_call_id(tool)
        if tool in self._local:
            started = time.monotonic()
            try:
                value = self._local[tool](**args)
                is_error = False
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
                value = {"code": type(exc).__name__, "message": str(exc)}
                is_error = True
            duration = int((time.monotonic() - started) * 1000)
            record_id = self._tools.report_local(
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
            )
            envelope = ToolCallResult(
                result=value if not is_error else {"error": value},
                record_id=record_id,
                status="error" if is_error else "success",
                provenance={"source": "runtime_local", "binding_mode": "live"},
                external_state="confirmed",
                duration_ms=duration,
            )
            self.calls.append(envelope)
            if is_error and self._raise:
                raise ToolCallError(tool, envelope)
            return envelope.result
        envelope = self._tools.invoke(
            self.run_id,
            tool,
            args,
            logical_call_id=call_id,
            repetition=self.repetition,
            attempt=attempt,
            parent_call_id=parent_call_id,
        )
        self.calls.append(envelope)
        if self._raise and not envelope.ok:
            raise ToolCallError(tool, envelope)
        return envelope.result

    def wrap(self, tool: str) -> Callable[..., Any]:
        """Return a callable ``fn(**arguments)`` routed through this run."""

        def routed(**arguments: Any) -> Any:
            return self.call(tool, arguments)

        routed.__name__ = tool.replace(".", "_")
        return routed

    @property
    def has_real_calls(self) -> bool:
        return any(c.is_real for c in self.calls)

    def summary(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        for c in self.calls:
            by_source[c.source] = by_source.get(c.source, 0) + 1
        return {
            "calls": len(self.calls),
            "by_source": by_source,
            "has_real_calls": self.has_real_calls,
        }

    def __repr__(self) -> str:
        return f"ToolRouter(run_id={self.run_id!r}, calls={len(self.calls)})"


def dumps_config(config: Mapping[str, Any]) -> str:
    """Serialize a configuration block to JSON text (accepted as ``config_text``)."""
    return json.dumps(dict(config), sort_keys=True)
