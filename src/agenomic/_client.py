"""The top-level Agenomic SDK client facade.

Local-first: with no ``base_url`` the client records tracking sessions in
memory / on disk. Pass ``base_url`` to stream to Agenomic Cloud — there is no
silent fallback from cloud to local.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from agenomic._version import __version__
from agenomic.client.auth import bearer_header
from agenomic.exceptions import CloudError
from agenomic.tracking import TrackingResource


class Client:
    """Entry point for the Agenomic SDK.

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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._transport = transport
        #: Online tracking namespace.
        self.tracking = TrackingResource(self)

    @property
    def is_cloud(self) -> bool:
        """True when a ``base_url`` was configured (cloud mode)."""
        return self.base_url is not None

    def _http(self) -> httpx.Client:
        headers: dict[str, str] = {"User-Agent": f"agenomic-python/{__version__}"}
        if self.api_key:
            headers.update(bearer_header(self.api_key))
        kwargs: dict[str, Any] = {
            "base_url": self.base_url or "",
            "headers": headers,
            "timeout": self._timeout,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _post(self, path: str, body: Any) -> dict[str, Any]:
        try:
            with self._http() as http:
                response = http.post(path, json=body)
                response.raise_for_status()
                return response.json() if response.content else {}
        except httpx.HTTPError as exc:
            raise CloudError(f"POST {path} failed: {exc}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with self._http() as http:
                response = http.get(path)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise CloudError(f"GET {path} failed: {exc}") from exc
