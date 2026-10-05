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

import logging
from typing import Any, Optional
from urllib.parse import quote

import httpx

from agenomic._version import __version__
from agenomic.exceptions import CloudError

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


def _seg(value: str) -> str:
    return quote(value, safe="")


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
        self._base = endpoint.rstrip("/") + base_path
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
        """Release the connection pool."""
        self._http.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
        ok_statuses: tuple[int, ...] = (200, 201),
        decision_statuses: tuple[int, ...] = (),
    ) -> tuple[int, dict[str, Any]]:
        """Send one request; return ``(status, body)`` for accepted statuses.

        ``decision_statuses`` are non 2xx statuses whose body is a decision
        (``403`` deny, ``202`` approval) and must carry a ``decision`` field.
        """
        budget = self.report_s if timeout_s is None else timeout_s
        timeout = httpx.Timeout(budget, connect=min(self._connect_s, budget))
        try:
            response = self._http.request(
                method, self._base + path, json=json_body, timeout=timeout
            )
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

    def __init__(self, endpoint: str, token: str, **kwargs: Any) -> None:
        super().__init__(endpoint, token, RUNTIME_BASE, **kwargs)

    def hello(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /hello``: capability and compatibility report."""
        return self.request("POST", "/hello", json_body=body, timeout_s=self.decision_s)[1]

    def heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /heartbeat``: liveness and exporter stats; returns pending commands."""
        return self.request("POST", "/heartbeat", json_body=body, timeout_s=self.decision_s)[1]

    def tools_discovered(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """``POST /tools/discovered`` (at most 500 tools)."""
        return self.request("POST", "/tools/discovered", json_body={"tools": tools})[1]

    def create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /sessions``: idempotent on ``hermes_session_id``."""
        return self.request("POST", "/sessions", json_body=body, timeout_s=self.decision_s)[1]

    def end_session(self, hermes_session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /sessions/:sid/end``."""
        path = f"/sessions/{_seg(hermes_session_id)}/end"
        return self.request("POST", path, json_body=body)[1]

    def reserve_delegation(self, hermes_session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /sessions/:sid/delegations``; a 403 deny is returned, not raised."""
        path = f"/sessions/{_seg(hermes_session_id)}/delegations"
        return self.request(
            "POST", path, json_body=body, timeout_s=self.decision_s, decision_statuses=(403,)
        )[1]

    def authorize(self, hermes_session_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """``POST /sessions/:sid/actions/authorize``: 200, 202 and a 403 carrying a decision."""
        path = f"/sessions/{_seg(hermes_session_id)}/actions/authorize"
        return self.request(
            "POST",
            path,
            json_body=body,
            timeout_s=self.decision_s,
            ok_statuses=(200,),
            decision_statuses=(202, 403),
        )

    def report(self, hermes_session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /sessions/:sid/actions/report`` with the permit."""
        path = f"/sessions/{_seg(hermes_session_id)}/actions/report"
        return self.request("POST", path, json_body=body)[1]

    def approval(self, approval_id: str) -> dict[str, Any]:
        """``GET /approvals/:id``."""
        return self.request("GET", f"/approvals/{_seg(approval_id)}", timeout_s=self.decision_s)[1]

    def post_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """``POST /events`` (at most 500 events, 1 MiB)."""
        return self.request("POST", "/events", json_body={"events": events})[1]

    def ack_command(self, command_id: str, status: str, detail: dict[str, Any]) -> dict[str, Any]:
        """``POST /commands/:id/ack`` with ``received | applied | refused``."""
        path = f"/commands/{_seg(command_id)}/ack"
        return self.request("POST", path, json_body={"status": status, "detail": detail})[1]

    def propose(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /proposals``: a change proposal; the runtime never approves."""
        return self.request("POST", "/proposals", json_body=body)[1]


class SupervisorClient(_ApiClient):
    """``/v1/hermes/supervisor`` with an ``agmhs_`` bearer token.

    Example:
        >>> import httpx
        >>> t = httpx.MockTransport(lambda r: httpx.Response(200, json={"skills": []}))
        >>> SupervisorClient("https://a.example", "agmhs_x", transport=t).approved_skills()
        {'skills': []}
    """

    def __init__(self, endpoint: str, token: str, **kwargs: Any) -> None:
        super().__init__(endpoint, token, SUPERVISOR_BASE, **kwargs)

    def heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /heartbeat`` with process state and isolation attestation."""
        return self.request("POST", "/heartbeat", json_body=body, timeout_s=self.decision_s)[1]

    def ack_command(self, command_id: str, status: str, detail: dict[str, Any]) -> dict[str, Any]:
        """``POST /commands/:id/ack``."""
        path = f"/commands/{_seg(command_id)}/ack"
        return self.request("POST", path, json_body={"status": status, "detail": detail})[1]

    def approved_skills(self) -> dict[str, Any]:
        """``GET /skills/approved``."""
        return self.request("GET", "/skills/approved")[1]
