"""The top-level Agenomic SDK client facade.

Local-first: with no ``base_url`` the client records tracking sessions in
memory / on disk. Pass ``base_url`` to stream to Agenomic Cloud — there is no
silent fallback from cloud to local.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import httpx

from agenomic._transport import aclose_pool, close_pool
from agenomic._version import __version__
from agenomic.agent import AgentResource
from agenomic.benchmarks.resources import BenchmarksResource
from agenomic.client.auth import bearer_header
from agenomic.exceptions import ApiError, CloudError
from agenomic.experiments.resources import ExperimentsResource
from agenomic.knowledge.resources import KnowledgeResource
from agenomic.prompts.cache import PromptCache
from agenomic.prompts.errors import PromptRefError
from agenomic.prompts.local import LocalPromptEngine
from agenomic.prompts.refs import is_uuid
from agenomic.prompts.resources import (
    BindingsResource,
    ChannelsResource,
    PromptsResource,
    arun_flow,
    run_flow,
    whoami_flow,
)
from agenomic.protect import ProtectResource
from agenomic.rmp import MonitorResource, ReviewResource, RmpResource
from agenomic.tools import ToolsResource
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
        workspace_id: Optional[str] = None,
        prompt_cache: Optional[PromptCache] = None,
    ) -> None:
        if workspace_id is not None and not is_uuid(workspace_id):
            raise ValueError("workspace_id must be a lowercase uuid")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._transport = transport
        self._workspace_id = workspace_id
        self._identity_lock = threading.Lock()
        self._identity: Optional[dict[str, Any]] = None
        self._prompt_engine: Optional[LocalPromptEngine] = (
            None if self.base_url else LocalPromptEngine(workspace_id)
        )
        self.prompt_cache = prompt_cache if prompt_cache is not None else PromptCache()
        self.prompts = PromptsResource(self)
        self.bindings = BindingsResource(self)
        self.channels = ChannelsResource(self)
        self.experiments = ExperimentsResource(self)
        self.knowledge = KnowledgeResource(self)
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

    @classmethod
    def from_env(cls, **overrides: Any) -> Client:
        settings: dict[str, Any] = {}
        for variable, name in (
            ("AGENOMIC_ENDPOINT", "base_url"),
            ("AGENOMIC_API_KEY", "api_key"),
            ("AGENOMIC_WORKSPACE_ID", "workspace_id"),
        ):
            value = os.environ.get(variable)
            if value:
                settings[name] = value
        timeout = os.environ.get("AGENOMIC_TIMEOUT")
        if timeout:
            try:
                settings["timeout"] = float(timeout)
            except ValueError as error:
                raise ValueError("AGENOMIC_TIMEOUT must be a number of seconds") from error
        cache_dir = os.environ.get("AGENOMIC_PROMPT_CACHE_DIR")
        if cache_dir and "prompt_cache" not in overrides:
            settings["prompt_cache"] = PromptCache(Path(cache_dir))
        settings.update(overrides)
        return cls(**settings)

    @property
    def is_cloud(self) -> bool:
        """True when a ``base_url`` was configured (cloud mode)."""
        return self.base_url is not None

    @property
    def workspace_id(self) -> Optional[str]:
        return self._known_workspace()

    def _cached_whoami(self) -> Optional[dict[str, Any]]:
        with self._identity_lock:
            identity = None if self._identity is None else dict(self._identity)
        if identity is not None:
            self._known_workspace()
        return identity

    def _remember_whoami(self, body: Mapping[str, Any]) -> dict[str, Any]:
        org_id = body.get("org_id")
        if not isinstance(org_id, str) or not is_uuid(org_id):
            raise ApiError("invalid_response", 200, "GET /v1/whoami returned no org_id")
        with self._identity_lock:
            self._identity = dict(body)
        self._known_workspace()
        return dict(body)

    def _known_workspace(self) -> Optional[str]:
        if self._prompt_engine is not None:
            return self._prompt_engine.workspace_id
        with self._identity_lock:
            org_id = None if self._identity is None else self._identity.get("org_id")
        if self._workspace_id is None:
            return org_id
        if org_id is not None and org_id != self._workspace_id:
            raise PromptRefError(
                "workspace_mismatch",
                0,
                "the configured workspace_id differs from the workspace of the API key",
            )
        return self._workspace_id

    def whoami(self) -> dict[str, Any]:
        return run_flow(self, whoami_flow(self))

    async def awhoami(self) -> dict[str, Any]:
        return await arun_flow(self, whoami_flow(self))

    def close(self) -> None:
        close_pool(self)

    async def aclose(self) -> None:
        await aclose_pool(self)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _http_kwargs(self) -> dict[str, Any]:
        headers: dict[str, str] = {"User-Agent": f"agenomic-python/{__version__}"}
        if self.api_key:
            headers.update(bearer_header(self.api_key))
        return {
            "base_url": self.base_url or "",
            "headers": headers,
            "timeout": self._timeout,
        }

    def _http(self) -> httpx.Client:
        kwargs = self._http_kwargs()
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _ahttp(self) -> httpx.AsyncClient:
        """Async transport for I/O-bound namespaces (``client.tools``)."""
        kwargs = self._http_kwargs()
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
