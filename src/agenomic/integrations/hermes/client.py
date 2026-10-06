"""Synchronous clients for the Hermes runtime and supervisor APIs.

The Hermes hooks are synchronous and run on Hermes threads, so the clients use
``httpx.Client`` with strict per call timeouts. TLS verification is always on.
Logs carry the route and the status, never the bearer token, the arguments or
the response body.

Example:
    >>> import httpx
    >>> transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"effective_state": "observe"}))
    >>> client = RuntimeClient("https://agenomic.example", "agmhr_x", transport=transport)
    >>> client.heartbeat({"active_sessions": []})["effective_state"]
    'observe'
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Optional, cast
from urllib.parse import quote

import httpx
from pydantic import JsonValue

from agenomic._version import __version__
from agenomic.exceptions import CloudError
from agenomic.integrations.hermes.config import check_endpoint

logger = logging.getLogger("agenomic.integrations.hermes.client")

RUNTIME_BASE = "/v1/hermes/runtime"
SUPERVISOR_BASE = "/v1/hermes/supervisor"
_MAX_MESSAGE = 300


class HermesApiError(CloudError):
    """A runtime or supervisor API call failed.

    ``status`` is the HTTP status (0 for transport failures and timeouts),
    ``code`` the gateway error code (``transport_error``, ``timeout``,
    ``invalid_response`` when the body is not a JSON object).
    """

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(f"{code} ({status}): {message}")
        self.code = code
        self.message = message
        self.status = status

    @property
    def retryable(self) -> bool:
        """``True`` when the same request may succeed later: a transport failure or
        timeout, a 5xx, or a transient 4xx (408 timeout, 425 too early, 429 rate limited).

        Example:
            >>> HermesApiError("rate_limited", "slow down", 429).retryable
            True
            >>> HermesApiError("not_found", "no such command", 404).retryable
            False
        """
        return self.status == 0 or self.status >= 500 or self.status in (408, 425, 429)


def _seg(value: str) -> str:
    return quote(value, safe="")


def _echo_transport() -> httpx.MockTransport:
    """Offline transport for the examples: answers 200 with the method, path and body."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        return httpx.Response(
            200, json={"method": request.method, "path": request.url.path, "body": body}
        )

    return httpx.MockTransport(handle)


class _ApiClient:
    def __init__(
        self,
        endpoint: str,
        token: str,
        base_path: str,
        *,
        connect_s: float = 3.0,
        decision_s: float = 5.0,
        report_s: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not token:
            raise ValueError("token is required")
        # The rule the adapter and the supervisor apply: credentials in the URL would make
        # HTTPX send Basic authentication in place of the bearer token.
        self._base = check_endpoint(endpoint) + base_path
        self._connect_s = connect_s
        self.decision_s = decision_s
        self.report_s = report_s
        self._http = httpx.Client(
            timeout=httpx.Timeout(report_s, connect=connect_s),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": f"agenomic-hermes-adapter/{__version__}",
                "Accept": "application/json",
            },
            verify=True,
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        """Release the connection pool. Later requests fail with ``HermesApiError("closed")``.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.close()
            >>> c.closed
            True
        """
        self._http.close()

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` released the connection pool.

        Example:
            >>> RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport()).closed
            False
        """
        return self._http.is_closed

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Mapping[str, JsonValue]] = None,
        timeout_s: Optional[float] = None,
        ok_statuses: tuple[int, ...] = (200, 201),
        decision_statuses: tuple[int, ...] = (),
    ) -> tuple[int, dict[str, JsonValue]]:
        """Send one request; return ``(status, body)`` for accepted statuses.

        ``decision_statuses`` are non 2xx statuses whose body is a decision
        (``403`` deny, ``202`` approval) and must carry a ``decision`` field.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.request("POST", "/hello", json_body={"a": 1})
            (200, {'method': 'POST', 'path': '/v1/hermes/runtime/hello', 'body': {'a': 1}})
        """
        budget = self.report_s if timeout_s is None else timeout_s
        timeout = httpx.Timeout(budget, connect=min(self._connect_s, budget))
        if self._http.is_closed:
            raise HermesApiError("closed", f"{method} {path}: client closed", 0)
        try:
            response = self._http.request(
                method, self._base + path, json=json_body, timeout=timeout
            )
        except RuntimeError:
            # httpx refuses a request on a client closed meanwhile (adapter shutdown).
            raise HermesApiError("closed", f"{method} {path}: client closed", 0) from None
        except httpx.TimeoutException as e:
            logger.warning("%s %s timed out (%s)", method, path, type(e).__name__)
            raise HermesApiError("timeout", f"{method} {path} timed out", 0) from None
        except httpx.HTTPError as e:
            logger.warning("%s %s failed: %s", method, path, type(e).__name__)
            raise HermesApiError(
                "transport_error", f"{method} {path}: {type(e).__name__}", 0
            ) from None
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        if status in ok_statuses or status in decision_statuses:
            if not isinstance(body, dict):
                raise HermesApiError(
                    "invalid_response", f"{method} {path}: body is not an object", status
                )
            if status in decision_statuses and status not in ok_statuses and "decision" not in body:
                raise HermesApiError(*self._error_fields(body, status), status)
            return status, body
        code, message = self._error_fields(body, status)
        logger.warning("%s %s answered %d (%s)", method, path, status, code)
        raise HermesApiError(code, message, status)

    @staticmethod
    def _error_fields(body: object, status: int) -> tuple[str, str]:
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            code = str(err.get("code") or f"http_{status}")
            message = str(err.get("message") or "")[:_MAX_MESSAGE]
            return code, message
        return f"http_{status}", f"HTTP {status}"


class RuntimeClient(_ApiClient):
    """``/v1/hermes/runtime`` with an ``agmhr_`` bearer token.

    Example:
        >>> import httpx
        >>> t = httpx.MockTransport(lambda r: httpx.Response(403, json={"decision": "deny"}))
        >>> RuntimeClient("https://a.example", "agmhr_x", transport=t).authorize("s1", {})[1]["decision"]
        'deny'
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        connect_s: float = 3.0,
        decision_s: float = 5.0,
        report_s: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        super().__init__(
            endpoint,
            token,
            RUNTIME_BASE,
            connect_s=connect_s,
            decision_s=decision_s,
            report_s=report_s,
            transport=transport,
        )

    def hello(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /hello``: capability and compatibility report.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.hello({"platform": "cli"})["path"]
            '/v1/hermes/runtime/hello'
        """
        return self.request("POST", "/hello", json_body=body, timeout_s=self.decision_s)[1]

    def heartbeat(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /heartbeat``: liveness and exporter stats; returns pending commands.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.heartbeat({"active_sessions": ["s1"]})["body"]
            {'active_sessions': ['s1']}
        """
        return self.request("POST", "/heartbeat", json_body=body, timeout_s=self.decision_s)[1]

    def tools_discovered(self, tools: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
        """``POST /tools/discovered`` (at most 500 tools).

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.tools_discovered([{"tool_name": "terminal"}])["body"]
            {'tools': [{'tool_name': 'terminal'}]}
        """
        return self.request(
            "POST", "/tools/discovered", json_body={"tools": cast(JsonValue, tools)}
        )[1]

    def create_session(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /sessions``: idempotent on ``hermes_session_id``.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.create_session({"hermes_session_id": "s1", "platform": "cli"})["path"]
            '/v1/hermes/runtime/sessions'
        """
        return self.request("POST", "/sessions", json_body=body, timeout_s=self.decision_s)[1]

    def end_session(
        self, hermes_session_id: str, body: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """``POST /sessions/:sid/end``.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.end_session("s1", {"final": True, "status": "completed"})["path"]
            '/v1/hermes/runtime/sessions/s1/end'
        """
        path = f"/sessions/{_seg(hermes_session_id)}/end"
        return self.request("POST", path, json_body=body)[1]

    def reserve_delegation(
        self, hermes_session_id: str, body: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """``POST /sessions/:sid/delegations``; a 403 deny is returned, not raised.

        Example:
            >>> import httpx
            >>> t = httpx.MockTransport(lambda r: httpx.Response(403, json={"decision": "deny"}))
            >>> RuntimeClient("https://a.example", "agmhr_x", transport=t).reserve_delegation("s1", {"count": 2})
            {'decision': 'deny'}
        """
        path = f"/sessions/{_seg(hermes_session_id)}/delegations"
        return self.request(
            "POST", path, json_body=body, timeout_s=self.decision_s, decision_statuses=(403,)
        )[1]

    def authorize(
        self, hermes_session_id: str, body: Mapping[str, JsonValue]
    ) -> tuple[int, dict[str, JsonValue]]:
        """``POST /sessions/:sid/actions/authorize``: 200, 202 and a 403 carrying a decision.

        Example:
            >>> import httpx
            >>> t = httpx.MockTransport(lambda r: httpx.Response(202, json={"decision": "require_approval"}))
            >>> RuntimeClient("https://a.example", "agmhr_x", transport=t).authorize("s1", {"tool": "terminal"})
            (202, {'decision': 'require_approval'})
        """
        path = f"/sessions/{_seg(hermes_session_id)}/actions/authorize"
        return self.request(
            "POST",
            path,
            json_body=body,
            timeout_s=self.decision_s,
            ok_statuses=(200,),
            decision_statuses=(202, 403),
        )

    def report(self, hermes_session_id: str, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /sessions/:sid/actions/report`` with the permit.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.report("s1", {"logical_call_id": "c1", "is_error": False})["path"]
            '/v1/hermes/runtime/sessions/s1/actions/report'
        """
        path = f"/sessions/{_seg(hermes_session_id)}/actions/report"
        return self.request("POST", path, json_body=body)[1]

    def approval(self, approval_id: str) -> dict[str, JsonValue]:
        """``GET /approvals/:id``.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.approval("apr_1")["method"], c.approval("apr_1")["path"]
            ('GET', '/v1/hermes/runtime/approvals/apr_1')
        """
        return self.request("GET", f"/approvals/{_seg(approval_id)}", timeout_s=self.decision_s)[1]

    def post_events(self, events: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
        """``POST /events`` (at most 500 events, 1 MiB).

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.post_events([{"event_id": "e1"}])["body"]
            {'events': [{'event_id': 'e1'}]}
        """
        return self.request("POST", "/events", json_body={"events": cast(JsonValue, events)})[1]

    def ack_command(
        self, command_id: str, status: str, detail: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """``POST /commands/:id/ack`` with ``received | applied | refused``.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.ack_command("cmd_1", "received", {"executor": "plugin"})["body"]
            {'status': 'received', 'detail': {'executor': 'plugin'}}
        """
        path = f"/commands/{_seg(command_id)}/ack"
        return self.request("POST", path, json_body={"status": status, "detail": dict(detail)})[1]

    def propose(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /proposals``: a change proposal; the runtime never approves.

        Example:
            >>> c = RuntimeClient("https://a.example", "agmhr_x", transport=_echo_transport())
            >>> c.propose({"kind": "skill", "target": "skills/demo/SKILL.md"})["path"]
            '/v1/hermes/runtime/proposals'
        """
        return self.request("POST", "/proposals", json_body=body)[1]


class SupervisorClient(_ApiClient):
    """``/v1/hermes/supervisor`` with an ``agmhs_`` bearer token.

    Example:
        >>> import httpx
        >>> t = httpx.MockTransport(lambda r: httpx.Response(200, json={"skills": []}))
        >>> SupervisorClient("https://a.example", "agmhs_x", transport=t).approved_skills()
        {'skills': []}
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        connect_s: float = 3.0,
        decision_s: float = 5.0,
        report_s: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        super().__init__(
            endpoint,
            token,
            SUPERVISOR_BASE,
            connect_s=connect_s,
            decision_s=decision_s,
            report_s=report_s,
            transport=transport,
        )

    def heartbeat(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """``POST /heartbeat`` with process state and isolation attestation.

        Example:
            >>> c = SupervisorClient("https://a.example", "agmhs_x", transport=_echo_transport())
            >>> c.heartbeat({"process": {"state": "running"}})["path"]
            '/v1/hermes/supervisor/heartbeat'
        """
        return self.request("POST", "/heartbeat", json_body=body, timeout_s=self.decision_s)[1]

    def ack_command(
        self, command_id: str, status: str, detail: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """``POST /commands/:id/ack``.

        Example:
            >>> c = SupervisorClient("https://a.example", "agmhs_x", transport=_echo_transport())
            >>> c.ack_command("cmd_1", "applied", {"restarted": True})["path"]
            '/v1/hermes/supervisor/commands/cmd_1/ack'
        """
        path = f"/commands/{_seg(command_id)}/ack"
        return self.request("POST", path, json_body={"status": status, "detail": dict(detail)})[1]

    def approved_skills(self) -> dict[str, JsonValue]:
        """``GET /skills/approved``.

        Example:
            >>> c = SupervisorClient("https://a.example", "agmhs_x", transport=_echo_transport())
            >>> c.approved_skills()["path"]
            '/v1/hermes/supervisor/skills/approved'
        """
        return self.request("GET", "/skills/approved")[1]
