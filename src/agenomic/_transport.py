from __future__ import annotations

import asyncio
import re
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import httpx

from agenomic.exceptions import ApiError
from agenomic.prompts.errors import RegistryUnavailableError, api_error

RETRY_STATUSES = frozenset({429, 502, 503, 504})
BACKOFF = (0.2, 0.8, 3.2)
_ETAG = re.compile(r'(?:W/)?"([0-9]+)"', re.ASCII)
_SECONDS = re.compile(r"[0-9]+", re.ASCII)
_ERROR_EXTRAS = ("request_id", "capability", "reason", "required_plan")

_sleep = time.sleep
_asleep = asyncio.sleep


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: dict[str, Any]
    etag: Optional[int]
    headers: Mapping[str, str]


def segment(value: str) -> str:
    return quote(value, safe="")


class HttpPool:
    def __init__(self, kwargs: Mapping[str, Any], transport: Any) -> None:
        self._kwargs = dict(kwargs)
        self._transport = transport
        self._lock = threading.Lock()
        self._sync: Optional[httpx.Client] = None
        self._async: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )

    def sync(self) -> httpx.Client:
        with self._lock:
            if self._sync is None:
                kwargs = dict(self._kwargs)
                if isinstance(self._transport, httpx.BaseTransport):
                    kwargs["transport"] = self._transport
                self._sync = httpx.Client(**kwargs)
            return self._sync

    def current(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._async.get(loop)
            if client is None:
                kwargs = dict(self._kwargs)
                if isinstance(self._transport, httpx.AsyncBaseTransport):
                    kwargs["transport"] = self._transport
                client = httpx.AsyncClient(**kwargs)
                self._async[loop] = client
            return client

    def close(self) -> None:
        with self._lock:
            client, self._sync = self._sync, None
            self._async = weakref.WeakKeyDictionary()
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            current = self._async.pop(loop, None)
        self.close()
        if current is not None:
            await current.aclose()


_POOLS: weakref.WeakKeyDictionary[Any, HttpPool] = weakref.WeakKeyDictionary()
_POOLS_LOCK = threading.Lock()


def pool_for(client: Any) -> HttpPool:
    with _POOLS_LOCK:
        pool = _POOLS.get(client)
        if pool is None:
            pool = HttpPool(client._http_kwargs(), getattr(client, "_transport", None))
            _POOLS[client] = pool
        return pool


def close_pool(client: Any) -> None:
    with _POOLS_LOCK:
        pool = _POOLS.pop(client, None)
    if pool is not None:
        pool.close()


async def aclose_pool(client: Any) -> None:
    with _POOLS_LOCK:
        pool = _POOLS.pop(client, None)
    if pool is not None:
        await pool.aclose()


def _headers(idempotency_key: Optional[str], if_match: Optional[int]) -> dict[str, str]:
    headers: dict[str, str] = {}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if if_match is not None:
        headers["If-Match"] = f'"{if_match}"'
    return headers


def _etag(response: httpx.Response) -> Optional[int]:
    match = _ETAG.fullmatch(response.headers.get("etag", ""))
    return int(match.group(1)) if match else None


def _retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("retry-after", "").strip()
    return float(value) if _SECONDS.fullmatch(value) else None


def _error_fields(response: httpx.Response) -> Optional[dict[str, Any]]:
    try:
        payload = response.json()
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else None


def _details(error: Mapping[str, Any]) -> dict[str, Any]:
    raw = error.get("details")
    details: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    for key in _ERROR_EXTRAS:
        if key in error and key not in details:
            details[key] = error[key]
    return details


def _error(method: str, path: str, response: httpx.Response) -> ApiError:
    default = f"{method} {path} returned {response.status_code}"
    error = _error_fields(response)
    if error is None:
        return ApiError("http_error", response.status_code, default)
    code = str(error.get("code") or "http_error")
    message = str(error.get("message") or default)
    return api_error(code, response.status_code, message, _details(error))


def _unavailable(method: str, path: str, response: httpx.Response) -> RegistryUnavailableError:
    error = _error_fields(response)
    details = _details(error) if error is not None else {}
    details["cause"] = str(error.get("code") or "http_error") if error is not None else "http_error"
    retry_after = _retry_after(response)
    if retry_after is not None:
        details["retry_after"] = retry_after
    return RegistryUnavailableError(
        "registry_unavailable",
        response.status_code,
        f"{method} {path} returned {response.status_code}",
        details,
    )


def _transport_failure(method: str, path: str, error: httpx.HTTPError) -> RegistryUnavailableError:
    return RegistryUnavailableError(
        "registry_unavailable",
        0,
        f"{method} {path} failed: {type(error).__name__}",
        {"cause": "transport_error"},
    )


def _parse(method: str, path: str, response: httpx.Response) -> ApiResponse:
    if response.status_code >= 400:
        raise _error(method, path, response)
    body: Any = {}
    if response.content:
        try:
            body = response.json()
        except ValueError as error:
            raise ApiError(
                "invalid_response",
                response.status_code,
                f"{method} {path} returned a non-JSON body",
            ) from error
    if not isinstance(body, dict):
        raise ApiError(
            "invalid_response", response.status_code, f"{method} {path} returned a non-object body"
        )
    return ApiResponse(response.status_code, body, _etag(response), dict(response.headers))


def _delay(response: Optional[httpx.Response], attempt: int) -> float:
    retry_after = _retry_after(response) if response is not None else None
    return retry_after if retry_after is not None else BACKOFF[attempt]


def api_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
    if_match: Optional[int] = None,
    retry: bool = False,
) -> ApiResponse:
    http = pool_for(client).sync()
    attempts = len(BACKOFF) + 1 if retry else 1
    headers = _headers(idempotency_key, if_match)
    attempt = 0
    while True:
        last = attempt + 1 >= attempts
        try:
            response = http.request(
                method, path, json=None if body is None else dict(body), headers=headers
            )
        except httpx.HTTPError as error:
            if last:
                raise _transport_failure(method, path, error) from error
            _sleep(_delay(None, attempt))
            attempt += 1
            continue
        if response.status_code in RETRY_STATUSES:
            if last:
                raise _unavailable(method, path, response)
            _sleep(_delay(response, attempt))
            attempt += 1
            continue
        return _parse(method, path, response)


async def aapi_request(
    client: Any,
    method: str,
    path: str,
    body: Optional[Mapping[str, Any]] = None,
    *,
    idempotency_key: Optional[str] = None,
    if_match: Optional[int] = None,
    retry: bool = False,
) -> ApiResponse:
    http = pool_for(client).current()
    attempts = len(BACKOFF) + 1 if retry else 1
    headers = _headers(idempotency_key, if_match)
    attempt = 0
    while True:
        last = attempt + 1 >= attempts
        try:
            response = await http.request(
                method, path, json=None if body is None else dict(body), headers=headers
            )
        except httpx.HTTPError as error:
            if last:
                raise _transport_failure(method, path, error) from error
            await _asleep(_delay(None, attempt))
            attempt += 1
            continue
        if response.status_code in RETRY_STATUSES:
            if last:
                raise _unavailable(method, path, response)
            await _asleep(_delay(response, attempt))
            attempt += 1
            continue
        return _parse(method, path, response)
