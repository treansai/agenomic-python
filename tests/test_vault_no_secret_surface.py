"""No public name reads a secret back, and a canary secret appears nowhere it should not."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import logging
import pickle
import pkgutil
import re
import traceback
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

import agenomic
from agenomic import Client
from agenomic.client.retry import RetryPolicy
from agenomic.vault import (
    ExecuteResult,
    IssuedIdentity,
    RuntimeIdentity,
    Sensitive,
    VaultError,
    VaultOutcomeUnknown,
)

FORBIDDEN = re.compile(r"read_value|get_value|reveal|export_secret|decrypt", re.IGNORECASE)
CORE = "do-not-leak-6d1f"
CANARY = f'canary"secret\\value-{CORE}'
CANARY_ESCAPED = json.dumps(CANARY)[1:-1]
BASE = "https://api.test"
ACTION = "0a1b2c3d-0000-4000-8000-000000000001"
RUNTIME_TOKEN = "vrt_runtime_token_for_surface_tests"
API_KEY = "agm_key_for_surface_tests"


def _agenomic_modules() -> Iterator[Any]:
    yield agenomic
    for info in pkgutil.walk_packages(agenomic.__path__, "agenomic."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:
            continue


def _members(owner: object) -> Iterator[str]:
    for name, value in inspect.getmembers(owner):
        if not name.startswith("_"):
            yield name
        if isinstance(value, type) and getattr(value, "__module__", "").startswith("agenomic"):
            yield from (f"{name}.{member}" for member in _class_surface(value))


def _class_surface(cls: type) -> Iterator[str]:
    for name in dir(cls):
        if not name.startswith("_"):
            yield name
    if issubclass(cls, BaseModel):
        yield from cls.model_fields


def offending_names(modules: Iterator[Any]) -> list[str]:
    found: list[str] = []
    for module in modules:
        for name in _members(module):
            if FORBIDDEN.search(name):
                found.append(f"{module.__name__}:{name}")
    return found


def test_no_public_name_of_the_package_reads_a_value_back() -> None:
    assert offending_names(_agenomic_modules()) == []


def test_the_surface_scan_would_catch_a_planted_accessor() -> None:
    class Leaky:
        def read_value(self) -> str:
            return "x"

        class Inner(BaseModel):
            decrypted_payload_get_value: str = ""

    Leaky.__module__ = Leaky.Inner.__module__ = "agenomic.planted"
    module = type("Planted", (), {"__name__": "planted", "Leaky": Leaky, "Inner": Leaky.Inner})
    found = offending_names(iter([module]))
    assert any("read_value" in name for name in found)
    assert any("get_value" in name for name in found)
    for word in ("reveal_secret", "export_secret_key", "Decrypt"):
        assert FORBIDDEN.search(word)


def test_the_live_objects_expose_no_accessor_either() -> None:
    client = Client(api_key=API_KEY, base_url=BASE, runtime_token=RUNTIME_TOKEN)
    objects: list[object] = [
        client,
        client.vault,
        client.tools,
        Sensitive("a value that must never be printed"),
        *(getattr(client.vault, name) for name in vars(client.vault) if not name.startswith("_")),
    ]
    for obj in objects:
        offending = [name for name in dir(obj) if FORBIDDEN.search(name)]
        assert offending == [], (type(obj).__name__, offending)


def test_no_response_model_has_a_field_that_can_hold_a_value() -> None:
    from agenomic.vault import models

    suspicious = {
        "value",
        "secret_value",
        "secret",
        "password",
        "plaintext",
        "private_key",
        "credential",
    }
    for _, cls in inspect.getmembers(models, inspect.isclass):
        if not issubclass(cls, BaseModel) or cls.__module__ != models.__name__:
            continue
        if cls.model_config.get("extra") == "forbid":
            continue
        bad = suspicious & set(cls.model_fields)
        assert bad <= {"secret"} or cls.__name__ == "SecretDetail", (cls.__name__, bad)
    assert "value" not in models.Secret.model_fields
    assert models.IssuedIdentity.model_fields["token"].annotation.__name__ == "IssuedToken"  # type: ignore[union-attr]


def _handler(sent: list[httpx.Request]) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        path = request.url.path
        if path == "/v1/vault/secrets" and b"echo" in request.content:
            message = f"invalid value {CANARY_ESCAPED} for field value"
            return httpx.Response(
                400, json={"error": {"code": "validation_error", "message": message}}
            )
        if path.endswith("/versions"):
            message = f"conflict on {CANARY_ESCAPED}"
            return httpx.Response(409, json={"error": {"code": "conflict", "message": message}})
        if path == "/v1/vault/secrets":
            return httpx.Response(
                201,
                json={"secret": {"id": "s-1", "name": "crm", "state": "active"}, "versions": []},
            )
        if path.endswith("/rotations"):
            return httpx.Response(201, json={"id": "r-1", "secret_id": "s-1", "state": "prepared"})
        if path == "/v1/vault/runtime/executions":
            body = {
                "status": "finished",
                "action_id": ACTION,
                "state": "succeeded",
                "receipt_id": "rcpt-1",
                "result": {"ok": True},
            }
            return httpx.Response(200, json=body)
        return httpx.Response(
            200, json={"identity": {"id": "i-1"}, "token": "vrt_one_time_surface_token"}
        )

    return handle


def _frame_reprs(error: BaseException) -> list[str]:
    reprs: list[str] = []
    for frame, _ in traceback.walk_tb(error.__traceback__):
        reprs.extend(repr(value) for value in frame.f_locals.values())
    return reprs


def _scan_targets(obj: object) -> list[str]:
    texts = [repr(obj), str(obj)]
    if isinstance(obj, BaseModel):
        texts.append(obj.model_dump_json())
        texts.append(repr(obj.model_dump()))
    if isinstance(obj, BaseException):
        texts += [repr(obj.args), repr(vars(obj)), *_frame_reprs(obj)]
    with contextlib.suppress(pickle.PicklingError, TypeError, AttributeError):
        texts.append(pickle.dumps(obj).decode("latin-1"))
    return texts


def _leaks(texts: list[str]) -> bool:
    """Any trace of the unique core of the canary, whatever escaping a layer applied to it."""
    return any(CORE in text for text in texts)


def test_a_canary_secret_appears_in_no_repr_exception_log_pickle_or_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    sent: list[httpx.Request] = []
    client = Client(
        api_key=API_KEY,
        base_url=BASE,
        runtime_token=RUNTIME_TOKEN,
        transport=httpx.MockTransport(_handler(sent)),
        vault_retry=RetryPolicy(max_retries=1, base_delay=0.0),
    )
    secret = Sensitive(CANARY)
    produced: list[object] = [secret]
    produced.append(
        client.vault.secrets.create(
            environment="e", name="crm", secret_type="api_key", provider_id="p", value=secret
        )
    )
    produced.append(client.vault.secrets.rotate("s-1", value=secret))
    for call in (
        lambda: client.vault.secrets.create(
            environment="e", name="echo", secret_type="api_key", provider_id="p", value=secret
        ),
        lambda: client.vault.secrets.add_version("s-1", value=secret),
    ):
        with pytest.raises(VaultError) as excinfo:
            call()
        produced.append(excinfo.value)
    produced.append(
        client.tools.execute(tool="t", binding="b", arguments={"k": "v"}, action_id=ACTION)
    )
    produced.append(
        client.vault.runtime_identities.issue(environment="e", agent_id="agent://a/b", label="l")
    )
    produced += [client, client.vault, client.vault.secrets, client.tools, client.vault.runtime]
    for obj in produced:
        assert not _leaks(_scan_targets(obj)), type(obj).__name__
    secret_requests = [r for r in sent if CANARY_ESCAPED.encode() in r.content]
    assert {r.url.path for r in secret_requests} == {
        "/v1/vault/secrets",
        "/v1/vault/secrets/s-1/rotations",
        "/v1/vault/secrets/s-1/versions",
    }
    assert all(r.method == "POST" for r in secret_requests)
    assert not any(
        CANARY_ESCAPED.encode() in r.content
        for r in sent
        if r.url.path.startswith("/v1/vault/runtime")
    )
    assert not any(CANARY in repr(r) + repr(dict(r.headers)) + str(r.url) for r in sent)
    records = [r.getMessage() + repr(r.args) + str(r.exc_text) for r in caplog.records]
    assert records, "the debug logs must have been captured"
    assert not _leaks(records)
    assert not _leaks([caplog.text])


async def test_the_async_path_leaks_nothing_either(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sent: list[httpx.Request] = []
    client = Client(api_key=API_KEY, base_url=BASE, transport=httpx.MockTransport(_handler(sent)))
    secret = Sensitive(CANARY)
    detail = await client.vault.secrets.acreate(
        environment="e", name="crm", secret_type="api_key", provider_id="p", value=secret
    )
    with pytest.raises(VaultError) as excinfo:
        await client.vault.secrets.aadd_version("s-1", value=secret)
    for obj in (detail, excinfo.value, secret):
        assert not _leaks(_scan_targets(obj))
    assert not _leaks([caplog.text])


def test_a_secret_bearing_failure_keeps_no_chained_exception_and_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = Client(api_key=API_KEY, base_url=BASE, transport=httpx.MockTransport(refuse))
    with pytest.raises(VaultError) as excinfo:
        client.vault.secrets.create(
            environment="e",
            name="n",
            secret_type="api_key",
            provider_id="p",
            value=Sensitive(CANARY),
        )
    error = excinfo.value
    assert (error.__cause__, error.__context__) == (None, None)
    assert not _leaks(_scan_targets(error))


def test_execute_results_and_errors_hold_the_business_data_only() -> None:
    result = ExecuteResult(action_id=ACTION, result={"ok": True}, receipt_id="r")
    assert set(type(result).model_fields) == {
        "action_id",
        "state",
        "result",
        "receipt_id",
        "status_code",
        "limitations",
        "replayed",
    }
    error = VaultOutcomeUnknown(ACTION, status_code=504)
    assert set(vars(error)) >= {"action_id", "status_code", "receipt_id", "limitations"}


def test_an_issued_token_is_masked_in_every_rendering_and_pickle() -> None:
    issued = IssuedIdentity(identity=RuntimeIdentity(id="i-1"), token="vrt_secret_token_to_hide")  # type: ignore[arg-type]
    blob = repr(issued) + str(issued) + issued.model_dump_json() + repr(issued.model_dump())
    assert "vrt_secret_token_to_hide" not in blob
    assert b"vrt_secret_token_to_hide" not in pickle.dumps(issued)
    assert json.loads(issued.model_dump_json())["token"] == "**********"
