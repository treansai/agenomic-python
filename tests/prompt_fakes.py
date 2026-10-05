from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts.local import LocalPromptEngine

WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"
OTHER_WORKSPACE = "5a5a5a5a-1111-4222-8333-444455556666"
AGENT = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
CHILD = "7f3c9a1e-0b2d-4c5e-8f6a-9b0c1d2e3f4a"
OTHER_AGENT = "3c4d5e6f-7a8b-4c9d-8e0f-1a2b3c4d5e6f"
NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)


def text_content(
    body: str, variables: Optional[dict[str, Any]] = None, **extra: Any
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "schema": "agenomic.prompt_content/v1",
        "template_format": "agenomic-fstring/v1",
        "renderer_version": "1",
        "kind": "text",
        "body": body,
        "variables": variables or {},
        "partials": {},
        "output_contract": None,
        "fragments": {},
    }
    content.update(extra)
    return content


def chat_content(
    body: list[dict[str, Any]], variables: Optional[dict[str, Any]] = None, **extra: Any
) -> dict[str, Any]:
    content = text_content("", variables, **extra)
    content["kind"] = "chat"
    content["body"] = body
    return content


def required(kind: str = "string") -> dict[str, Any]:
    return {"type": kind, "required": True}


def seeded_engine(workspace_id: str = WORKSPACE) -> LocalPromptEngine:
    engine = LocalPromptEngine(workspace_id)
    engine.create_prompt("prm_safety", kind="fragment", name="Safety")
    safety = engine.publish(
        "prm_safety",
        text_content("Never share internal notes."),
        parent_version=None,
        change_message="first",
    )
    engine.create_prompt("prm_planner", kind="chat", name="Planner")
    engine.publish(
        "prm_planner",
        chat_content(
            [
                {"role": "system", "content": "Plan for {customer}. {>safety}"},
                {"placeholder": "history", "optional": True},
                {"role": "user", "content": "{question}"},
            ],
            {
                "customer": required(),
                "history": {"type": "messages", "required": False},
                "question": required(),
            },
            fragments={
                "safety": {
                    "prompt_id": "prm_safety",
                    "version": 1,
                    "content_digest": safety.content_digest,
                }
            },
        ),
        parent_version=None,
        change_message="first",
    )
    engine.create_prompt("prm_writer", kind="text", name="Writer")
    engine.publish(
        "prm_writer",
        text_content("Write about {topic}.", {"topic": required()}),
        parent_version=None,
        change_message="first",
    )
    return engine


def release_with_child(engine: LocalPromptEngine, *, status: str = "approved") -> tuple[str, str]:
    child = engine.create_release(CHILD, {"writer.response": "prm_writer:1"})
    root = engine.create_release(
        AGENT, {"planner.instructions": "prm_planner:1"}, children={CHILD: child}, status=status
    )
    return root, child


def make_signed_bundle(
    *,
    status: str = "approved",
    signer: Optional[SigningKey] = None,
    workspace_id: str = WORKSPACE,
    agent_id: str = AGENT,
) -> tuple[dict[str, Any], SigningKey]:
    key = signer or SigningKey.generate("orgkey_test")
    engine = seeded_engine(workspace_id)
    if agent_id == AGENT:
        release, _ = release_with_child(engine, status=status)
    else:
        release = engine.create_release(agent_id, {"writer.main": "prm_writer:1"}, status=status)
    bundle = engine.export_bundle(agent_id, signer=key, release_id=release, now=NOW)
    return bundle, key


@dataclass
class FakePromptServer:
    engine: LocalPromptEngine
    api_key_scopes: list[str] = field(default_factory=lambda: ["read"])
    outage: Optional[str] = None
    fail_next: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.outage == "transport_error":
            raise httpx.ConnectError("registry unreachable", request=request)
        if self.outage == "http_503":
            return _error_response(ApiError("service_unavailable", 503, "registry down"))
        if self.outage == "http_403":
            return _error_response(ApiError("forbidden", 403, "access revoked"))
        if self.fail_next:
            status, headers = self.fail_next.pop(0)
            return httpx.Response(
                status, headers=headers, json={"error": {"code": "busy", "message": "retry later"}}
            )
        try:
            return self._route(request)
        except ApiError as error:
            return _error_response(error)

    def _route(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        for pattern, method, handler in self._routes():
            match = re.fullmatch(pattern, path)
            if match and request.method == method:
                return handler(request, body, *match.groups())
        return _error_response(ApiError("not_found", 404, f"{request.method} {path}"))

    def _routes(self) -> list[tuple[str, str, Callable[..., httpx.Response]]]:
        return [
            (r"/v1/whoami", "GET", self._whoami),
            (r"/v1/prompts/([^/]+)/versions/([0-9]+)", "GET", self._version),
            (r"/v1/prompts/([^/]+)/draft", "GET", self._get_draft),
            (r"/v1/prompts/([^/]+)/draft", "PUT", self._put_draft),
            (r"/v1/agents/([^/]+)/bindings", "POST", self._create_binding),
            (r"/v1/agents/([^/]+)/channels/([^/]+)", "GET", self._channel),
            (r"/v1/echo", "POST", self._echo),
            (r"/v1/text", "GET", self._text),
            (r"/v1/list", "GET", self._list),
        ]

    def _whoami(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(
            200, json={"org_id": self.engine.workspace_id, "api_key_scopes": self.api_key_scopes}
        )

    def _version(
        self, request: httpx.Request, body: Any, prompt_id: str, number: str
    ) -> httpx.Response:
        version = self.engine.get_version(prompt_id, int(number))
        records = version.closure_records()
        wire = [record.model_dump(mode="json") for record in records]
        main = next(item for item in wire if item["prompt_id"] == prompt_id)
        return httpx.Response(
            200, json={"version": main, "fragments": [item for item in wire if item is not main]}
        )

    def _get_draft(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        draft = self.engine.get_draft(prompt_id)
        return httpx.Response(200, headers={"ETag": f'"{draft["revision"]}"'}, json=draft)

    def _put_draft(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        header = request.headers.get("if-match")
        if header is None:
            raise ApiError("if_match_required", 400, "If-Match is required")
        match = re.fullmatch(r'"([0-9]+)"', header)
        if match is None:
            raise ApiError("validation_error", 400, "If-Match must carry the numeric revision")
        draft = self.engine.save_draft(
            prompt_id,
            body["content"],
            base_version=body.get("base_version"),
            expected_revision=int(match.group(1)),
        )
        return httpx.Response(200, headers={"ETag": f'"{draft["revision"]}"'}, json=draft)

    def _create_binding(self, request: httpx.Request, body: Any, agent_id: str) -> httpx.Response:
        selector = body.get("selector", {})
        binding, artifacts, created = self.engine.create_binding(
            agent_id,
            thread_key=body["thread_key"],
            scope=body["scope"],
            channel=selector.get("channel"),
            release_id=selector.get("release_id"),
        )
        return httpx.Response(
            201 if created else 200,
            json={"created": created, "binding": binding, "artifacts": artifacts},
        )

    def _channel(
        self, request: httpx.Request, body: Any, agent_id: str, name: str
    ) -> httpx.Response:
        channel = self.engine.get_channel(agent_id, name)
        return httpx.Response(200, headers={"ETag": f'"{channel["generation"]}"'}, json=channel)

    def _echo(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "body": body,
                "if_match": request.headers.get("if-match"),
                "idempotency_key": request.headers.get("idempotency-key"),
            },
        )

    def _text(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(200, text="plain text")

    def _list(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])


def _error_response(error: ApiError) -> httpx.Response:
    payload: dict[str, Any] = {
        "code": error.code,
        "message": error.message,
        "request_id": "6c1e9f4a-2a7b-4d3e-9b8f-0f3c2d1e4a5b",
    }
    if error.details:
        payload["details"] = error.details
    status = error.status or 400
    return httpx.Response(status, json={"error": payload})
