from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import pytest
from prompt_fakes import AGENT, FakePromptServer, chat_content, release_with_child, seeded_engine

from agenomic import _transport
from agenomic._client import Client
from agenomic._transport import (
    aapi_request,
    aclose_pool,
    api_request,
    close_pool,
    pool_for,
    segment,
)
from agenomic.exceptions import ApiError, CloudError
from agenomic.prompts.errors import (
    PromptBindingError,
    PromptConflictError,
    PromptImportError,
    PromptIntegrityError,
    PromptRefError,
    PromptRenderError,
    PromptTemplateError,
    RegistryUnavailableError,
    api_error,
)


@pytest.fixture
def server() -> FakePromptServer:
    return FakePromptServer(seeded_engine(), api_key_scopes=["write"])


@pytest.fixture
def client(server: FakePromptServer) -> Client:
    return Client(api_key="key", base_url="https://api.test", transport=server.transport())


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(_transport, "_sleep", recorded.append)

    async def asleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(_transport, "_asleep", asleep)
    return recorded


def test_get_returns_body_and_etag(client: Client) -> None:
    response = api_request(client, "GET", f"/v1/agents/{AGENT}/channels/production")
    assert response.status == 200
    assert response.etag == 0
    assert response.body["channel"]["name"] == "production"
    assert response.headers["etag"] == '"0"'


def test_if_match_and_idempotency_headers(client: Client, server: FakePromptServer) -> None:
    response = api_request(
        client, "POST", "/v1/echo", {"a": 1}, if_match=3, idempotency_key="key-1"
    )
    assert response.body == {"body": {"a": 1}, "if_match": '"3"', "idempotency_key": "key-1"}
    assert server.requests[-1].headers["authorization"] == "Bearer key"
    assert response.etag is None


def test_draft_conflict_carries_current(client: Client, server: FakePromptServer) -> None:
    content = chat_content([{"role": "user", "content": "hi"}])
    server.engine.save_draft("prm_planner", content, base_version=1, expected_revision=0)
    path = "/v1/prompts/prm_planner/draft"
    saved = api_request(client, "PUT", path, {"content": content}, if_match=1)
    assert saved.etag == 2
    with pytest.raises(PromptConflictError) as raised:
        api_request(client, "PUT", path, {"content": content}, if_match=1)
    assert raised.value.code == "prompt_draft_conflict"
    assert raised.value.status == 409
    assert raised.value.details["current"] == 2
    assert raised.value.request_id == "6c1e9f4a-2a7b-4d3e-9b8f-0f3c2d1e4a5b"
    assert str(raised.value).startswith("prompt_draft_conflict: ")
    assert isinstance(raised.value, CloudError)


def test_missing_if_match_is_a_plain_api_error(client: Client) -> None:
    with pytest.raises(ApiError) as raised:
        api_request(client, "PUT", "/v1/prompts/prm_planner/draft", {"content": {}})
    assert type(raised.value) is ApiError
    assert raised.value.code == "if_match_required"
    assert raised.value.reason is None
    assert raised.value.errors == []


def test_binding_conflict_maps_to_binding_error(client: Client, server: FakePromptServer) -> None:
    root, _ = release_with_child(server.engine)
    path = f"/v1/agents/{AGENT}/bindings"
    body = {
        "thread_key": "thread:sha256:" + "a" * 64,
        "scope": "thread",
        "selector": {"release_id": root},
    }
    first = api_request(client, "POST", path, body, retry=True)
    assert first.status == 201
    again = api_request(client, "POST", path, body, retry=True)
    assert again.status == 200
    assert again.body["binding"]["binding_id"] == first.body["binding"]["binding_id"]
    with pytest.raises(PromptBindingError) as raised:
        api_request(client, "POST", path, {**body, "scope": "execution"})
    assert raised.value.code == "execution_binding_conflict"


def test_retry_honours_retry_after(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    server.fail_next = [(503, {"Retry-After": "2"}), (429, {})]
    response = api_request(client, "GET", "/v1/whoami", retry=True)
    assert response.body["org_id"] == server.engine.workspace_id
    assert sleeps == [2.0, 0.8]


def test_retry_exhausted_is_registry_unavailable(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    server.fail_next = [(503, {}), (502, {}), (504, {}), (503, {"Retry-After": "1"})]
    with pytest.raises(RegistryUnavailableError) as raised:
        api_request(client, "GET", "/v1/whoami", retry=True)
    assert sleeps == [0.2, 0.8, 3.2]
    assert raised.value.status == 503
    assert raised.value.details["cause"] == "busy"
    assert raised.value.details["retry_after"] == 1.0


def test_no_retry_without_flag(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    server.outage = "http_503"
    with pytest.raises(RegistryUnavailableError) as raised:
        api_request(client, "GET", "/v1/whoami")
    assert sleeps == []
    assert raised.value.details["cause"] == "service_unavailable"
    assert len(server.requests) == 1


def test_transport_error_is_status_zero(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    server.outage = "transport_error"
    with pytest.raises(RegistryUnavailableError) as raised:
        api_request(client, "GET", "/v1/whoami", retry=True)
    assert raised.value.status == 0
    assert raised.value.details == {"cause": "transport_error"}
    assert sleeps == [0.2, 0.8, 3.2]


def test_forbidden_is_never_retried(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    server.outage = "http_403"
    with pytest.raises(ApiError) as raised:
        api_request(client, "GET", "/v1/whoami", retry=True)
    assert raised.value.status == 403
    assert sleeps == []


def _raw_client(handler: Any) -> Client:
    return Client(base_url="https://api.test", transport=httpx.MockTransport(handler))


def test_non_json_error_is_http_error() -> None:
    client = _raw_client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ApiError) as raised:
        api_request(client, "GET", "/v1/x")
    assert raised.value.code == "http_error"
    assert raised.value.message == "GET /v1/x returned 500"


def test_unavailable_without_json_body(sleeps: list[float]) -> None:
    client = _raw_client(lambda request: httpx.Response(502, text="bad gateway"))
    with pytest.raises(RegistryUnavailableError) as raised:
        api_request(client, "GET", "/v1/x")
    assert raised.value.details == {"cause": "http_error"}


def test_error_extras_and_details_are_merged() -> None:
    payload = {
        "error": {
            "code": "capability_denied",
            "message": "denied",
            "request_id": "r1",
            "capability": "prompts.experiments",
            "details": {"reason": "plan"},
        }
    }
    client = _raw_client(lambda request: httpx.Response(403, json=payload))
    with pytest.raises(ApiError) as raised:
        api_request(client, "GET", "/v1/x")
    assert raised.value.details == {
        "reason": "plan",
        "request_id": "r1",
        "capability": "prompts.experiments",
    }
    assert raised.value.reason == "plan"


def test_error_body_without_error_object() -> None:
    client = _raw_client(lambda request: httpx.Response(404, json={"message": "missing"}))
    with pytest.raises(ApiError) as raised:
        api_request(client, "GET", "/v1/x")
    assert raised.value.code == "http_error"


def test_success_bodies_must_be_json_objects(client: Client) -> None:
    with pytest.raises(ApiError) as raised:
        api_request(client, "GET", "/v1/text")
    assert raised.value.code == "invalid_response"
    with pytest.raises(ApiError) as listed:
        api_request(client, "GET", "/v1/list")
    assert listed.value.code == "invalid_response"
    empty = _raw_client(lambda request: httpx.Response(204))
    assert api_request(empty, "DELETE", "/v1/x").body == {}


def test_error_codes_map_to_classes() -> None:
    expected = {
        "prompt_ref_invalid": PromptRefError,
        "prompt_secret_detected": PromptTemplateError,
        "prompt_content_too_large": PromptTemplateError,
        "prompt_fragment_cycle": PromptTemplateError,
        "prompt_fragment_depth_exceeded": PromptTemplateError,
        "prompt_render_error": PromptRenderError,
        "artifact_integrity_error": PromptIntegrityError,
        "prompt_import_plan_stale": PromptImportError,
        "release_not_bindable": PromptBindingError,
        "session_required": PromptBindingError,
        "channel_conflict": PromptConflictError,
        "registry_unavailable": RegistryUnavailableError,
        "api_key_scope_insufficient": ApiError,
    }
    for code, cls in expected.items():
        error = api_error(code, 409, "message", {"current": 4})
        assert type(error) is cls
        assert error.details == {"current": 4}


def test_segment_quotes_every_separator() -> None:
    assert segment("prm_x/../y") == "prm_x%2F..%2Fy"
    assert segment("a b?c") == "a%20b%3Fc"


def test_sync_pool_is_reused_and_closed(client: Client) -> None:
    pool = pool_for(client)
    assert pool_for(client) is pool
    http = pool.sync()
    assert pool.sync() is http
    close_pool(client)
    assert http.is_closed
    assert pool_for(client) is not pool
    close_pool(client)
    close_pool(client)


def test_async_requests_use_one_client_per_loop(client: Client, server: FakePromptServer) -> None:
    async def call() -> tuple[int, Any]:
        response = await aapi_request(client, "GET", "/v1/whoami")
        return response.status, pool_for(client).current()

    first_status, first_http = asyncio.run(call())
    second_status, second_http = asyncio.run(call())
    assert first_status == second_status == 200
    assert first_http is not second_http

    async def closing() -> None:
        current = pool_for(client).current()
        await aclose_pool(client)
        assert current.is_closed
        await aclose_pool(client)

    asyncio.run(closing())


def test_closing_a_pool_releases_clients_of_other_loops(client: Client) -> None:
    loops: dict[str, Any] = {}
    ready = threading.Event()
    stop = threading.Event()

    def other_loop() -> None:
        async def main() -> None:
            loops["client"] = pool_for(client).current()
            loops["loop"] = asyncio.get_running_loop()
            ready.set()
            await asyncio.to_thread(stop.wait)

        asyncio.run(main())

    thread = threading.Thread(target=other_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    async def closing() -> None:
        await aclose_pool(client)

    asyncio.run(closing())
    assert loops["client"].is_closed

    stop.set()
    thread.join(timeout=5)

    ready.clear()
    stop.clear()
    thread = threading.Thread(target=other_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    close_pool(client)
    deadline = time.monotonic() + 5
    while not loops["client"].is_closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loops["client"].is_closed
    stop.set()
    thread.join(timeout=5)


def test_async_retry_and_errors(
    client: Client, server: FakePromptServer, sleeps: list[float]
) -> None:
    async def scenario() -> None:
        server.fail_next = [(503, {})]
        response = await aapi_request(client, "POST", "/v1/echo", {"x": 1}, retry=True, if_match=0)
        assert response.body["if_match"] == '"0"'
        server.fail_next = [(503, {})] * 4
        with pytest.raises(RegistryUnavailableError):
            await aapi_request(client, "GET", "/v1/whoami", retry=True)
        server.outage = "transport_error"
        with pytest.raises(RegistryUnavailableError) as raised:
            await aapi_request(client, "GET", "/v1/whoami", retry=True)
        assert raised.value.status == 0

    asyncio.run(scenario())
    assert sleeps == [0.2, 0.2, 0.8, 3.2, 0.2, 0.8, 3.2]
