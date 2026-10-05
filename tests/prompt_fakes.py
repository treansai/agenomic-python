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
from agenomic.prompts.refs import parse_execution_ref

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


SCOPE_MESSAGE = "The API key scope does not allow this action."
SESSION_MESSAGE = "This action requires a signed-in user session."
API_KEY_ID = "6e7f8a9b-0c1d-4e2f-8a3b-4c5d6e7f8a9b"
REQUEST_ID = "6c1e9f4a-2a7b-4d3e-9b8f-0f3c2d1e4a5b"


def _stamp(moment: Optional[datetime]) -> Optional[str]:
    return None if moment is None else moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict(body: Any, allowed: set[str], not_null: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ApiError("validation_error", 400, "the body must be a JSON object")
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ApiError("validation_error", 400, f"unknown field `{unknown[0]}`")
    for key in not_null:
        if key in body and body[key] is None:
            raise ApiError("validation_error", 400, f"invalid type: null for `{key}`")
    return body


def _if_match(request: httpx.Request) -> int:
    header = request.headers.get("if-match")
    if header is None:
        raise ApiError("if_match_required", 400, "If-Match is required")
    match = re.fullmatch(r'"([0-9]+)"', header)
    if match is None:
        raise ApiError("validation_error", 400, "If-Match must carry the numeric revision")
    return int(match.group(1))


def _ref_status(error: ApiError) -> ApiError:
    status = 403 if error.code == "prompt_ref_cross_workspace" else 400
    return ApiError(error.code, status, error.message, error.details)


@dataclass
class FakePromptServer:
    engine: LocalPromptEngine
    api_key_scopes: Optional[list[str]] = field(default_factory=lambda: ["read"])
    outage: Optional[str] = None
    fail_next: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    signer: SigningKey = field(default_factory=lambda: SigningKey.generate("orgkey_fake"))
    rewrite: Optional[Callable[[httpx.Request, dict[str, Any]], dict[str, Any]]] = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def paths(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.outage == "transport_error":
            raise httpx.ConnectError("registry unreachable", request=request)
        if self.outage == "http_503":
            return _error_response(ApiError("service_unavailable", 503, "registry down"))
        if self.outage == "http_403":
            return _error_response(ApiError("forbidden", 403, "access revoked"))
        if self.outage == "http_404":
            return _error_response(ApiError("not_found", 404, "agent not found"))
        if self.fail_next:
            status, headers = self.fail_next.pop(0)
            return httpx.Response(
                status, headers=headers, json={"error": {"code": "busy", "message": "retry later"}}
            )
        try:
            response = self._route(request)
        except ApiError as error:
            return _error_response(error)
        if self.rewrite is not None and response.status_code < 300 and response.content:
            payload = self.rewrite(request, json.loads(response.content))
            return httpx.Response(response.status_code, headers=response.headers, json=payload)
        return response

    def _route(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        for pattern, method, handler in self._routes():
            match = re.fullmatch(pattern, path)
            if match and request.method == method:
                return handler(request, body, *match.groups())
        return _error_response(ApiError("not_found", 404, f"{request.method} {path}"))

    def _routes(self) -> list[tuple[str, str, Callable[..., httpx.Response]]]:
        agent = r"/v1/agents/([^/]+)"
        prompt = r"/v1/prompts/([^/]+)"
        return [
            (r"/v1/whoami", "GET", self._whoami),
            (r"/v1/prompts", "GET", self._list_prompts),
            (r"/v1/prompts", "POST", self._create_prompt),
            (r"/v1/prompts/resolve", "POST", self._resolve),
            (prompt + r"/versions", "GET", self._list_versions),
            (prompt + r"/versions", "POST", self._publish),
            (prompt + r"/versions/([0-9]+)", "GET", self._version),
            (prompt + r"/draft", "GET", self._get_draft),
            (prompt + r"/draft", "PUT", self._put_draft),
            (prompt + r"/aliases/([^/]+)", "GET", self._get_alias),
            (prompt + r"/aliases/([^/]+)", "PUT", self._move_alias),
            (agent + r"/bindings", "POST", self._create_binding),
            (agent + r"/bindings/([^/]+)", "GET", self._get_binding),
            (agent + r"/bindings/([^/]+)/children", "POST", self._child_binding),
            (agent + r"/resolve", "GET", self._resolve_agent),
            (agent + r"/prompt-bundle", "GET", self._export_bundle),
            (r"/v1/signing-keys/([^/]+)", "GET", self._signing_key),
            (agent + r"/channels", "GET", self._channels),
            (agent + r"/channels/([^/]+)", "GET", self._channel),
            (agent + r"/channels/([^/]+)/history", "GET", self._history),
            (agent + r"/channels/([^/]+)/move-preview", "GET", self._move_preview),
            (r"/v1/echo", "POST", self._echo),
            (r"/v1/text", "GET", self._text),
            (r"/v1/list", "GET", self._list),
        ]

    def _require_editor(self) -> None:
        scopes = self.api_key_scopes
        if scopes and not {"write", "admin"} & set(scopes):
            raise ApiError("api_key_scope_insufficient", 403, SCOPE_MESSAGE)

    def _whoami(self, request: httpx.Request, body: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "org_id": self.engine.workspace_id,
                "user_id": None,
                "api_key_id": API_KEY_ID,
                "api_key_name": "fake",
                "role": "maintainer",
                "auth_method": "api_key",
                "api_key_scopes": self.api_key_scopes,
            },
        )

    def wire_prompt(self, prompt_id: str) -> dict[str, Any]:
        prompt = self.engine.prompt(prompt_id)
        prompt.update(
            {
                "created_by": {"user_id": None, "api_key_id": API_KEY_ID},
                "archived_at": None,
                "uri_prefix": f"agenomic://{self.engine.workspace_id}/prompts/{prompt_id}",
            }
        )
        return prompt

    def wire_version(self, prompt_id: str, number: int, *, full: bool = True) -> dict[str, Any]:
        version = self.engine.get_version(prompt_id, number)
        workspace = self.engine.workspace_id
        wire: dict[str, Any] = {
            "prompt_id": prompt_id,
            "version": number,
            "ref": f"{prompt_id}:{number}",
            "canonical_uri": f"agenomic://{workspace}/prompts/{prompt_id}/versions/{number}",
            "content_digest": version.content_digest,
            "parent_version": version.parent_version,
            "change_message": version.change_message,
            "author": {"user_id": None, "api_key_id": API_KEY_ID},
            "provenance": {
                "source": "api",
                "draft_revision": None,
                "import_id": None,
                "item_id": None,
                "source_file": None,
                "source_line": None,
            },
            "created_at": _stamp(version.created_at),
            "variable_descriptions": {},
        }
        if full:
            wire["content"] = version.document
            wire["fragment_closure"] = [
                {
                    "prompt_id": record.prompt_id,
                    "version": record.version,
                    "content_digest": record.content_digest,
                }
                for record in version.closure_records()
                if record.ref != f"{prompt_id}:{number}"
            ]
        return wire

    def _page(self, request: httpx.Request, items: list[Any]) -> tuple[list[Any], Optional[str]]:
        limit = int(request.url.params.get("limit", "50"))
        start = int(request.url.params.get("cursor", "0"))
        end = start + limit
        return items[start:end], (str(end) if end < len(items) else None)

    def _list_prompts(self, request: httpx.Request, body: Any) -> httpx.Response:
        params = request.url.params
        status = params.get("status", "active")
        tags = [tag for tag in params.get("tags", "").split(",") if tag]
        found = []
        for prompt_id in sorted(self.engine._state["prompts"]):
            prompt = self.wire_prompt(prompt_id)
            if status != "all" and prompt["status"] != status:
                continue
            if "kind" in params and prompt["kind"] != params["kind"]:
                continue
            if "q" in params and params["q"] not in prompt_id + prompt["name"]:
                continue
            if any(tag not in prompt["tags"] for tag in tags):
                continue
            found.append(prompt)
        items, cursor = self._page(request, found)
        return httpx.Response(200, json={"prompts": items, "next_cursor": cursor})

    def _create_prompt(self, request: httpx.Request, body: Any) -> httpx.Response:
        self._require_editor()
        _strict(
            body,
            {
                "prompt_id",
                "kind",
                "name",
                "description",
                "owner",
                "tags",
                "initial_content",
                "change_message",
                "variable_descriptions",
            },
            ("tags", "variable_descriptions"),
        )
        self.engine.create_prompt(
            body["prompt_id"],
            kind=body["kind"],
            name=body["name"],
            description=body.get("description"),
            owner=body.get("owner"),
            tags=body.get("tags", []),
        )
        prompt = self.wire_prompt(body["prompt_id"])
        return httpx.Response(
            201,
            headers={"ETag": f'"{prompt["metadata_revision"]}"'},
            json={"prompt": prompt, "version": None},
        )

    def _resolve(self, request: httpx.Request, body: Any) -> httpx.Response:
        _strict(body, {"ref", "include"}, ("include",))
        try:
            parsed = parse_execution_ref(body["ref"], workspace_id=self.engine.workspace_id)
        except ApiError as error:
            raise _ref_status(error) from error
        version = self.engine.get(parsed)
        resolved = version.resolved_from
        number = version.ref.version
        form = "alias" if resolved is not None else "uri" if "://" in body["ref"] else "version"
        return httpx.Response(
            200,
            json={
                "input": body["ref"],
                "form": form,
                "prompt_id": version.ref.prompt_id,
                "version": number,
                "ref": str(version.ref),
                "canonical_uri": str(version.uri),
                "content_digest": version.content_digest,
                "alias": (
                    None
                    if resolved is None
                    else {"name": resolved.alias, "generation": resolved.generation}
                ),
                "kind": version.content.kind,
                "archived": False,
                "version_document": None,
            },
        )

    def _list_versions(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        prompt = self.engine._prompt(prompt_id)
        numbers = sorted((int(key) for key in prompt["versions"]), reverse=True)
        summaries = [self.wire_version(prompt_id, number, full=False) for number in numbers]
        items, cursor = self._page(request, summaries)
        return httpx.Response(200, json={"versions": items, "next_cursor": cursor})

    def _publish(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        self._require_editor()
        _strict(
            body,
            {
                "from_draft",
                "content",
                "parent_version",
                "change_message",
                "source",
                "variable_descriptions",
            },
            ("variable_descriptions",),
        )
        before = self.engine.prompt(prompt_id)["latest_version"]
        version = self.engine.publish(
            prompt_id,
            body["content"],
            parent_version=body.get("parent_version"),
            change_message=body.get("change_message") or "",
            variable_descriptions=body.get("variable_descriptions"),
        )
        created = self.engine.prompt(prompt_id)["latest_version"] != before
        return httpx.Response(
            201 if created else 200,
            json={
                "version": self.wire_version(prompt_id, version.ref.version),
                "created": created,
            },
        )

    def _version(
        self, request: httpx.Request, body: Any, prompt_id: str, number: str
    ) -> httpx.Response:
        main = self.wire_version(prompt_id, int(number))
        payload: dict[str, Any] = {"version": main}
        if request.url.params.get("include") == "fragments":
            payload["fragments"] = [
                self.wire_version(pin["prompt_id"], pin["version"])
                for pin in sorted(
                    main["fragment_closure"], key=lambda pin: (pin["prompt_id"], pin["version"])
                )
            ]
        return httpx.Response(200, json=payload)

    def _get_draft(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        draft = self.engine.get_draft(prompt_id)
        return httpx.Response(
            200, headers={"ETag": f'"{draft["revision"]}"'}, json={"draft": draft}
        )

    def _put_draft(self, request: httpx.Request, body: Any, prompt_id: str) -> httpx.Response:
        self._require_editor()
        expected = _if_match(request)
        _strict(body, {"base_version", "content", "origin"})
        draft = self.engine.save_draft(
            prompt_id,
            body["content"],
            base_version=body.get("base_version"),
            expected_revision=expected,
        )
        return httpx.Response(
            201 if expected == 0 else 200,
            headers={"ETag": f'"{draft["revision"]}"'},
            json={"draft": draft},
        )

    def _get_alias(
        self, request: httpx.Request, body: Any, prompt_id: str, alias: str
    ) -> httpx.Response:
        found = self.engine.get_alias(prompt_id, alias)
        return httpx.Response(
            200, headers={"ETag": f'"{found["generation"]}"'}, json={"alias": found}
        )

    def _move_alias(
        self, request: httpx.Request, body: Any, prompt_id: str, alias: str
    ) -> httpx.Response:
        raise ApiError("session_required", 403, SESSION_MESSAGE, {"reason": "alias_move"})

    def _governed(self, agent_id: str, release_id: str) -> None:
        release = self.engine.get_release(release_id)
        if release["status"] in ("approved", "production", "rejected", "rolled_back"):
            return
        targets = {
            state["release_id"]
            for state in self.engine._state["channels"].get(agent_id, {}).values()
        }
        if release_id not in targets:
            raise ApiError(
                "session_required", 403, SESSION_MESSAGE, {"reason": "ungoverned_release"}
            )

    def _create_binding(self, request: httpx.Request, body: Any, agent_id: str) -> httpx.Response:
        _strict(
            body,
            {
                "thread_key",
                "scope",
                "selector",
                "child_selectors",
                "runtime_client",
                "expect",
                "include",
            },
            ("child_selectors", "include"),
        )
        selector = _strict(body.get("selector") or {}, {"channel", "release_id"})
        if "release_id" in selector:
            self._governed(agent_id, selector["release_id"])
        expect = _strict(body.get("expect") or {}, {"prompt_manifest_digest"})
        runtime = _strict(
            body.get("runtime_client") or {}, {"sdk", "sdk_version", "adapter", "adapter_version"}
        )
        binding, artifacts, created = self.engine.create_binding(
            agent_id,
            thread_key=body["thread_key"],
            scope=body["scope"],
            channel=selector.get("channel"),
            release_id=selector.get("release_id"),
            expect_manifest_digest=expect.get("prompt_manifest_digest"),
            runtime_client=runtime or None,
        )
        payload: dict[str, Any] = {"created": created, "binding": binding}
        if "artifacts" in body.get("include", []):
            payload["artifacts"] = artifacts
        return httpx.Response(201 if created else 200, json=payload)

    def _get_binding(
        self, request: httpx.Request, body: Any, agent_id: str, binding_id: str
    ) -> httpx.Response:
        binding, artifacts = self.engine.get_binding(agent_id, binding_id)
        payload: dict[str, Any] = {"binding": binding}
        if request.url.params.get("include") == "artifacts":
            payload["artifacts"] = artifacts
        return httpx.Response(200, json=payload)

    def _child_binding(
        self, request: httpx.Request, body: Any, agent_id: str, parent_id: str
    ) -> httpx.Response:
        _strict(body, {"thread_key", "selector", "reason"})
        selector = _strict(body.get("selector") or {}, {"channel", "release_id"})
        if "release_id" not in selector:
            raise ApiError(
                "agent_selector_required",
                400,
                "a release_id is required",
                {"reason": "release_id_required"},
            )
        self._governed(agent_id, selector["release_id"])
        parent, _ = self.engine.get_binding(agent_id, parent_id)
        binding, _, created = self.engine.create_binding(
            agent_id,
            thread_key=body["thread_key"],
            scope=parent["scope"],
            release_id=selector["release_id"],
        )
        stored = self.engine._state["bindings"][agent_id][body["thread_key"]]
        stored["parent_binding_id"] = parent_id
        binding["parent_binding_id"] = parent_id
        return httpx.Response(201 if created else 200, json={"binding": binding})

    def _release_ref(self, release_id: Optional[str]) -> Optional[dict[str, Any]]:
        if release_id is None:
            return None
        release = self.engine.get_release(release_id)
        return {
            "release_id": release_id,
            "agent_id": release["agent_id"],
            "name": release["name"],
            "status": release["status"],
            "origin": "prompt_candidate",
            "base_release_id": None,
            "genome_version": release["genome_version"],
            "prompt_manifest_digest": release["prompt_manifest_digest"],
            "bundle_id": release["bundle_id"],
            "bundle_hash": release["bundle_hash"],
            "created_at": release["created_at"],
        }

    def _resolve_agent(self, request: httpx.Request, body: Any, agent_id: str) -> httpx.Response:
        params = request.url.params
        if "release_id" in params:
            self._governed(agent_id, params["release_id"])
        artifacts = self.engine.resolve(
            agent_id, channel=params.get("channel"), release_id=params.get("release_id")
        )
        source = artifacts["source"]
        release_id = artifacts["release"]["release_id"]
        return httpx.Response(
            200,
            json={
                "agent_id": agent_id,
                "selector": (
                    {"channel": source["channel"], "generation": source["channel_generation"]}
                    if "channel" in source
                    else source
                ),
                "release": self._release_ref(release_id),
                "genome_version": artifacts["release"]["genome_version"],
                "prompt_manifest_digest": artifacts["prompt_manifest_digest"],
                "runtime": {
                    "bundle_id": artifacts["release"]["bundle_id"],
                    "bundle_hash": artifacts["release"]["bundle_hash"],
                },
                "children": {
                    child_id: {
                        "release_id": child["release_id"],
                        "genome_version": child["genome_version"],
                        "source": "manifest",
                    }
                    for child_id, child in artifacts["children"].items()
                },
                "artifacts": artifacts,
                "resolved_at": _stamp(NOW),
            },
        )

    def _export_bundle(self, request: httpx.Request, body: Any, agent_id: str) -> httpx.Response:
        params = request.url.params
        if "release_id" in params:
            self._governed(agent_id, params["release_id"])
        bundle = self.engine.export_bundle(
            agent_id,
            signer=self.signer,
            channel=params.get("channel"),
            release_id=params.get("release_id"),
            expires_in_days=int(params.get("expires_in_days", "30")),
        )
        return httpx.Response(200, json=bundle)

    def _signing_key(self, request: httpx.Request, body: Any, key_id: str) -> httpx.Response:
        if key_id != self.signer.key_id:
            raise ApiError("not_found", 404, "signing key not found")
        return httpx.Response(
            200,
            json={
                "key_id": key_id,
                "algorithm": "ed25519",
                "public_key_pem": self.signer.public_pem(),
                "status": "active",
            },
        )

    def wire_channel(self, agent_id: str, name: str) -> dict[str, Any]:
        state = self.engine.get_channel(agent_id, name)
        state.pop("history")
        state["release"] = self._release_ref(state["release_id"])
        state["updated_by"] = None
        state["updated_at"] = None
        return state

    def _channels(self, request: httpx.Request, body: Any, agent_id: str) -> httpx.Response:
        names = {"production", *self.engine._state["channels"].get(agent_id, {})}
        return httpx.Response(
            200,
            json={
                "agent_id": agent_id,
                "channels": [self.wire_channel(agent_id, name) for name in sorted(names)],
            },
        )

    def _channel(
        self, request: httpx.Request, body: Any, agent_id: str, name: str
    ) -> httpx.Response:
        channel = self.wire_channel(agent_id, name)
        return httpx.Response(
            200, headers={"ETag": f'"{channel["generation"]}"'}, json={"channel": channel}
        )

    def _history(
        self, request: httpx.Request, body: Any, agent_id: str, name: str
    ) -> httpx.Response:
        after = int(request.url.params.get("after", "0"))
        limit = int(request.url.params.get("limit", "50"))
        events = [
            {
                **event,
                "actor": {"user_id": None, "api_key_id": None},
                "approval_ids": [],
                "evidence_refs": [],
                "reason": None,
            }
            for event in self.engine.get_channel(agent_id, name)["history"]
            if event["generation"] > after
        ]
        page = events[:limit]
        following = page[-1]["generation"] if len(page) >= limit else None
        return httpx.Response(200, json={"events": page, "next_after": following})

    def _move_preview(
        self, request: httpx.Request, body: Any, agent_id: str, name: str
    ) -> httpx.Response:
        params = dict(request.url.params)
        channel = self.wire_channel(agent_id, name)
        return httpx.Response(
            200,
            json={
                "agent_id": agent_id,
                "action": params.get("action", "promote"),
                "channel": {
                    "name": name,
                    "generation": channel["generation"],
                    "protected": channel["protected"],
                    "release_id": channel["release_id"],
                },
                "candidate": self._release_ref(params.get("release_id")),
                "actions": {"promote": False, "rollback": False, "reasons": ["session_required"]},
                "move_url": f"/agents/{agent_id}/channels/{name}",
                "query": params,
            },
        )

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
        "request_id": REQUEST_ID,
    }
    if error.details:
        payload["details"] = error.details
    status = error.status or 400
    return httpx.Response(status, json={"error": payload})
