"""The top-level Agenomic SDK client facade.

Local-first: with no ``base_url`` the client records tracking sessions in
memory / on disk. Pass ``base_url`` to stream to Agenomic Cloud — there is no
silent fallback from cloud to local.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import httpx

from agenomic._version import __version__
from agenomic.agent import AgentResource
from agenomic.benchmarks.resources import BenchmarksResource
from agenomic.client.auth import bearer_header
from agenomic.client.retry import RetryPolicy
from agenomic.exceptions import CloudError
from agenomic.protect import ProtectResource
from agenomic.rmp import MonitorResource, ReviewResource, RmpResource
from agenomic.tools import ToolsResource
from agenomic.tracking import TrackingResource
from agenomic.vault.admin import VaultResource
from agenomic.vault.replay import VaultReplay
from agenomic.vault.sensitive import Sensitive


def _held(token: Optional[Union[str, Sensitive]]) -> Optional[Sensitive]:
    if isinstance(token, str):
        return Sensitive(token) if token else None
    return token


class Client:
    """Entry point for the Agenomic SDK.

    ``runtime_token`` (a ``vrt_`` string or a :class:`~agenomic.vault.Sensitive`)
    is the credential of an agent runtime for Agents Vault (``client.tools.execute``);
    it is held masked and never sent on an administrative route, and ``api_key``
    is never sent on a runtime route. ``vault_replay`` answers vault executions
    from fixtures and never falls back to a live call; ``vault_retry`` bounds the
    technical retries of vault calls (default: ``RetryPolicy()``).

    Example:
        >>> client = Client()                      # local mode
        >>> session = client.tracking.start(agent="agent://acme/demo")
        >>> _ = session.intent("answer_question")
        >>> session.stop()
        >>> [e["type"] for e in session.events]
        ['intent.detected']
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        runtime_token: Optional[Union[str, Sensitive]] = None,
        vault_replay: Optional[VaultReplay] = None,
        vault_retry: Optional[RetryPolicy] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._transport = transport
        self._runtime_token = _held(runtime_token)
        self._vault_replay = vault_replay
        self._vault_retry = vault_retry or RetryPolicy()
        #: Online tracking namespace.
        self.tracking = TrackingResource(self)
        #: Local agent genome namespace (load + configure_model).
        self.agent = AgentResource(self)
        #: Review · Monitor · Protect loop namespaces.
        self.rmp = RmpResource(self)
        self.review = ReviewResource(self)
        self.monitor = MonitorResource(self)
        self.protect = ProtectResource(self)
        #: RMP benchmarks (cloud only): catalogue, plans, launches, runs, policies.
        self.benchmarks = BenchmarksResource(self)
        #: Replay tool execution: Tool Gateway (real calls) and Tool Mock Engine.
        self.tools = ToolsResource(self)
        #: Agents Vault (optional commercial module): admin plane and ``vault.runtime``.
        self.vault = VaultResource(self)

    @property
    def is_cloud(self) -> bool:
        """True when a ``base_url`` was configured (cloud mode)."""
        return self.base_url is not None

    @property
    def vault_replay(self) -> Optional[VaultReplay]:
        """The fixture set that answers vault executions offline, if one is attached."""
        return self._vault_replay

    def _http_kwargs(self, bearer: Optional[str] = None) -> dict[str, Any]:
        headers: dict[str, str] = {"User-Agent": f"agenomic-python/{__version__}"}
        credential = self.api_key if bearer is None else bearer
        if credential:
            headers.update(bearer_header(credential))
        return {
            "base_url": self.base_url or "",
            "headers": headers,
            "timeout": self._timeout,
        }

    def _http(self, bearer: Optional[str] = None) -> httpx.Client:
        """Sync transport; ``bearer`` replaces the API key for one plane (the vault runtime)."""
        kwargs = self._http_kwargs(bearer)
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _ahttp(self, bearer: Optional[str] = None) -> httpx.AsyncClient:
        """Async transport for I/O-bound namespaces (``client.tools``)."""
        kwargs = self._http_kwargs(bearer)
        if isinstance(self._transport, httpx.AsyncBaseTransport):
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def _post(self, path: str, body: Any) -> dict[str, Any]:
        try:
            with self._http() as http:
                response = http.post(path, json=body)
                response.raise_for_status()
                return response.json() if response.content else {}
        except httpx.HTTPError as exc:
            raise CloudError(f"POST {path} failed: {exc}") from exc

    def _put(self, path: str, body: Any) -> dict[str, Any]:
        try:
            with self._http() as http:
                response = http.put(path, json=body)
                response.raise_for_status()
                return response.json() if response.content else {}
        except httpx.HTTPError as exc:
            raise CloudError(f"PUT {path} failed: {exc}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with self._http() as http:
                response = http.get(path)
                response.raise_for_status()
                data: dict[str, Any] = response.json() if response.content else {}
                return data
        except httpx.HTTPError as exc:
            raise CloudError(f"GET {path} failed: {exc}") from exc
