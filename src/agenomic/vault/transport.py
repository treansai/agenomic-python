"""HTTP transport of the Agents Vault surface.

One credential per plane: administrative calls carry the API key, runtime
calls carry the runtime-identity token, and neither ever travels on the other
plane. Technical retries (network failure, 429, 502, 503, 504) are bounded by
the client retry policy and only happen where resending is safe: reads, throttled
writes and executions, which reuse the same ``action_id`` so the server
de-duplicates them. Failures are raised outside the ``except`` block that caught
them, so no chained exception keeps a request, and with it a body, alive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, Optional, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from agenomic.client.retry import RetryPolicy
from agenomic.vault.errors import (
    VaultError,
    VaultNotConfigured,
    VaultTransportError,
    VaultValidationError,
    build_error,
)
from agenomic.vault.sensitive import Sensitive, _unseal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agenomic._client import Client

logger = logging.getLogger("agenomic.vault")

Plane = Literal["admin", "runtime"]
Retry = Literal["none", "throttled", "reads", "action"]

_GATEWAY_STATUS = frozenset({429, 502, 503, 504})
_MAX_RETRY_AFTER = 60.0
_sleep = time.sleep
_asleep = asyncio.sleep

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Call:
    """One vault request. ``body`` may carry a :class:`Sensitive` only when ``secret_bearing``."""

    method: str
    path: str
    plane: Plane = "admin"
    body: Optional[Mapping[str, object]] = None
    query: Mapping[str, str] = field(default_factory=dict)
    retry: Retry = "none"
    action_id: Optional[str] = None
    secret_bearing: bool = False


@dataclass(frozen=True)
class Reply:
    """A successful (below 300) or failed response, reduced to what the SDK reads."""

    status: int
    payload: object = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Op(Generic[T]):
    """A call and the parser of its reply: the unit every vault operation is built from."""

    call: Call
    parse: Callable[[Reply], T]


@dataclass(frozen=True)
class _Prepared:
    call: Call
    bearer: Sensitive = field(repr=False)
    content: Optional[bytes] = field(repr=False)
    needles: tuple[str, ...] = field(repr=False)
    policy: RetryPolicy


@dataclass(frozen=True)
class _Outcome:
    reply: Optional[Reply] = None
    failure: Optional[str] = None


def segment(value: object) -> str:
    """Percent-encode one path segment so an id can never reach another route."""
    return quote(str(value), safe="")


def query_of(**params: object) -> dict[str, str]:
    return {key: str(value) for key, value in params.items() if value is not None}


def present(**fields: object) -> dict[str, object]:
    """The fields that are set: an unset optional never reaches the wire."""
    return {key: value for key, value in fields.items() if value is not None}


def get(path: str, *, plane: Plane = "admin", query: Optional[Mapping[str, str]] = None) -> Call:
    return Call("GET", path, plane, None, query or {}, "reads")


def post(
    path: str,
    body: Optional[Mapping[str, object]] = None,
    *,
    plane: Plane = "admin",
    secret: bool = False,
) -> Call:
    return Call("POST", path, plane, body, {}, "throttled", None, secret)


def _unseal_hook(value: object) -> str:
    if isinstance(value, Sensitive):
        return _unseal(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _reject_hook(value: object) -> str:
    if isinstance(value, Sensitive):
        raise VaultValidationError(
            "sensitive_not_allowed", "a Sensitive value can only be sent to store a secret", 0
        )
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dump(call: Call) -> Optional[str]:
    if call.body is None:
        return None
    hook = _unseal_hook if call.secret_bearing else _reject_hook
    try:
        return json.dumps(
            call.body, default=hook, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
    except (TypeError, ValueError):
        pass
    raise VaultValidationError(
        "invalid_arguments",
        "the request body is not JSON serializable",
        0,
        action_id=call.action_id,
    )


def _needles(call: Call) -> tuple[str, ...]:
    values = [v for v in (call.body or {}).values() if isinstance(v, Sensitive)]
    raw = [_unseal(v) for v in values] if call.secret_bearing else []
    return tuple(raw + [json.dumps(r, ensure_ascii=False)[1:-1] for r in raw])


def _credential(client: Client, plane: Plane) -> Sensitive:
    if plane == "runtime":
        if client._runtime_token is None:
            raise VaultNotConfigured(
                "runtime_token_required",
                "this call needs a runtime token: Client(runtime_token=...)",
            )
        return client._runtime_token
    if not client.api_key:
        raise VaultNotConfigured(
            "api_key_required", "this call needs an API key: Client(api_key=...)"
        )
    return Sensitive(client.api_key)


def _prepare(client: Client, call: Call) -> _Prepared:
    if not client.is_cloud:
        raise VaultNotConfigured(
            "cloud_required",
            "Agents Vault needs Agenomic Cloud (base_url) or a replay fixture set",
        )
    text = _dump(call)
    return _Prepared(
        call,
        _credential(client, call.plane),
        None if text is None else text.encode("utf-8"),
        _needles(call),
        client._vault_retry,
    )


def _headers(prepared: _Prepared) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if prepared.content is not None:
        headers["Content-Type"] = "application/json"
    return headers


def _reply_of(response: httpx.Response) -> Reply:
    try:
        payload: object = response.json() if response.content else None
    except ValueError:
        payload = None
    hint = response.headers.get("retry-after")
    return Reply(response.status_code, payload, {"retry-after": hint} if hint else {})


def _once(client: Client, prepared: _Prepared) -> _Outcome:
    call = prepared.call
    try:
        with client._http(bearer=_unseal(prepared.bearer)) as http:
            response = http.request(
                call.method,
                call.path,
                content=prepared.content,
                headers=_headers(prepared),
                params=dict(call.query) or None,
            )
    except httpx.HTTPError as error:
        return _Outcome(failure=type(error).__name__)
    return _Outcome(reply=_reply_of(response))


async def _aonce(client: Client, prepared: _Prepared) -> _Outcome:
    call = prepared.call
    try:
        async with client._ahttp(bearer=_unseal(prepared.bearer)) as http:
            response = await http.request(
                call.method,
                call.path,
                content=prepared.content,
                headers=_headers(prepared),
                params=dict(call.query) or None,
            )
    except httpx.HTTPError as error:
        return _Outcome(failure=type(error).__name__)
    return _Outcome(reply=_reply_of(response))


def _retryable(mode: Retry, outcome: _Outcome) -> bool:
    status = outcome.reply.status if outcome.reply is not None else None
    if mode == "none":
        return False
    if mode == "throttled":
        return status == 429
    return outcome.failure is not None or status in _GATEWAY_STATUS


def _hint(outcome: _Outcome) -> Optional[float]:
    raw = outcome.reply.headers.get("retry-after") if outcome.reply is not None else None
    try:
        return min(float(raw), _MAX_RETRY_AFTER) if raw is not None else None
    except ValueError:
        return None


def _delay(prepared: _Prepared, outcome: _Outcome, attempt: int) -> Optional[float]:
    if attempt >= prepared.policy.max_retries or not _retryable(prepared.call.retry, outcome):
        return None
    hint = _hint(outcome)
    return hint if hint is not None else prepared.policy.delay_for(attempt)


def _log_retry(call: Call, outcome: _Outcome, attempt: int, delay: float) -> None:
    cause = outcome.failure or (outcome.reply.status if outcome.reply else "unknown")
    logger.info(
        "vault %s %s retry %d in %.2fs after %s (action_id=%s)",
        call.method,
        call.path,
        attempt + 1,
        delay,
        cause,
        call.action_id,
    )


def _finish(prepared: _Prepared, outcome: _Outcome) -> Reply:
    call = prepared.call
    if outcome.reply is None:
        resend = " Resend with the same action_id." if call.action_id else ""
        raise VaultTransportError(
            f"{call.method} {call.path} failed ({outcome.failure}).{resend}",
            action_id=call.action_id,
        )
    logger.debug("vault %s %s -> %s", call.method, call.path, outcome.reply.status)
    if outcome.reply.status >= 300:
        raise build_error(
            outcome.reply.status,
            outcome.reply.payload,
            outcome.reply.headers,
            action_id=call.action_id,
            scrub=prepared.needles,
        )
    return outcome.reply


def send(client: Client, call: Call) -> Reply:
    """Send one call, retrying only what is safe to resend; failures raise typed errors."""
    prepared = _prepare(client, call)
    attempt = 0
    while True:
        outcome = _once(client, prepared)
        delay = _delay(prepared, outcome, attempt)
        if delay is None:
            return _finish(prepared, outcome)
        _log_retry(call, outcome, attempt, delay)
        _sleep(delay)
        attempt += 1


async def asend(client: Client, call: Call) -> Reply:
    """Async counterpart of :func:`send`."""
    prepared = _prepare(client, call)
    attempt = 0
    while True:
        outcome = await _aonce(client, prepared)
        delay = _delay(prepared, outcome, attempt)
        if delay is None:
            return _finish(prepared, outcome)
        _log_retry(call, outcome, attempt, delay)
        await _asleep(delay)
        attempt += 1


def validate(model: type[M], payload: object, status: int) -> M:
    """Validate a payload into a response model without echoing it on failure."""
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        count = error.error_count()
    raise VaultError(
        "invalid_response",
        f"the response does not match {model.__name__} ({count} field error(s))",
        status,
    )


def model_of(model: type[M]) -> Callable[[Reply], M]:
    return lambda reply: validate(model, reply.payload, reply.status)


def list_of(model: type[M]) -> Callable[[Reply], list[M]]:
    def parse(reply: Reply) -> list[M]:
        if not isinstance(reply.payload, list):
            raise VaultError("invalid_response", "the response is not a list", reply.status)
        return [validate(model, item, reply.status) for item in reply.payload]

    return parse


class Namespace:
    """Base of every ``client.vault`` namespace: run an :class:`Op` sync or async."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def _run(self, op: Op[T]) -> T:
        return op.parse(send(self._client, op.call))

    async def _arun(self, op: Op[T]) -> T:
        return op.parse(await asend(self._client, op.call))
