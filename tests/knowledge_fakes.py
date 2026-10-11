from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from agenomic import Client

BASE = "https://cloud.test"
KB = "kb_customer_support"
DOC = "kdoc_01jb3m5q7s9v1x3z5b7d9f0001"
NEW_DOC = "kdoc_01jb3m5q7s9v1x3z5b7d9f0009"
SECTION = "sec_c41e8a5f2b9d7036"
AGENT = "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c"
BINDING = "kbnd_01jb3m5q7s9v1x3z5b7d9f0011"
JOB = "kjob_01jb3m5q7s9v1x3z5b7d9f0047"
EVENT = "kret_01jb3m5q7s9v1x3z5b7d9f0005"
PROMPT_BINDING = "bnd_01j9x4w6k2m8n0p3q5r7s9t1v3"
REQUEST_ID = "req_01knowledge"
FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"

Reply = Union[dict[str, Any], httpx.Response]
Handler = Callable[[httpx.Request], Reply]


def fixture(name: str) -> dict[str, Any]:
    return dict(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


def error_response(
    code: str, status: int, message: str = "refused", **details: Any
) -> httpx.Response:
    payload: dict[str, Any] = {"code": code, "message": message, "request_id": REQUEST_ID}
    if details:
        payload["details"] = details
    return httpx.Response(status, json={"error": payload})


def body_of(request: httpx.Request) -> Any:
    return json.loads(request.content) if request.content else None


def _document_detail() -> dict[str, Any]:
    written = fixture("document_write")
    listed = fixture("document_list")["documents"][0]
    return {"document": listed, "revision": {**written["revision"], "revision": 6}}


def _decision() -> dict[str, Any]:
    version = fixture("version_created")["version"]
    approval = fixture("version_detail")["approvals"][0]
    return {"version": {**version, "status": "approved"}, "approval": approval, "event": None}


def _binding_result() -> dict[str, Any]:
    knowledge = fixture("agent_knowledge")
    return {"binding": knowledge["bindings"][0], "knowledge": knowledge}


def _routes() -> list[tuple[str, str, Handler]]:
    kb = f"/v1/knowledge-bases/{KB}"
    doc = f"{kb}/documents/{DOC}"
    record = fixture("knowledge_base")["knowledge_base"]
    collection = fixture("collections")["collections"][1]
    return [
        ("GET", "/v1/knowledge-bases", lambda _: fixture("knowledge_base_list")),
        (
            "POST",
            "/v1/knowledge-bases",
            lambda _: httpx.Response(201, json={"knowledge_base": record}),
        ),
        ("GET", kb, lambda _: fixture("knowledge_base")),
        ("PATCH", kb, lambda _: {"knowledge_base": record}),
        ("DELETE", kb, lambda _: httpx.Response(204)),
        ("POST", f"{kb}/archive", lambda _: {"knowledge_base": {**record, "status": "archived"}}),
        ("POST", f"{kb}/restore", lambda _: {"knowledge_base": record}),
        ("GET", f"{kb}/collections", lambda _: fixture("collections")),
        (
            "POST",
            f"{kb}/collections",
            lambda _: httpx.Response(201, json={"collection": collection}),
        ),
        ("PATCH", f"{kb}/collections/faq", lambda _: {"collection": collection}),
        ("DELETE", f"{kb}/collections/faq", lambda _: httpx.Response(204)),
        ("GET", f"{kb}/documents", lambda _: fixture("document_list")),
        ("POST", f"{kb}/documents", lambda _: httpx.Response(202, json=fixture("document_write"))),
        (
            "POST",
            f"{kb}/documents/upload",
            lambda _: httpx.Response(202, json=fixture("document_write")),
        ),
        ("GET", doc, lambda _: _document_detail()),
        ("PATCH", doc, lambda _: _document_detail()),
        ("DELETE", doc, lambda _: httpx.Response(204)),
        ("POST", f"{doc}/restore", lambda _: _document_detail()),
        ("PUT", f"{doc}/content", lambda _: httpx.Response(202, json=fixture("document_write"))),
        ("POST", f"{doc}/upload", lambda _: httpx.Response(202, json=fixture("document_write"))),
        ("POST", f"{doc}/duplicate", lambda _: httpx.Response(202, json=fixture("document_write"))),
        ("GET", f"{doc}/revisions", lambda _: fixture("revisions")),
        (
            "GET",
            f"{doc}/content",
            lambda _: {
                "document_id": DOC,
                "revision": 6,
                "media_type": "text/markdown",
                "content_digest": None,
                "text": "# Refunds",
            },
        ),
        ("GET", f"{doc}/sections", lambda _: fixture("section_tree")),
        ("GET", f"{doc}/sections/{SECTION}", lambda _: fixture("section")),
        ("GET", f"{doc}/backlinks", lambda _: fixture("backlinks")),
        ("POST", f"{kb}/search", lambda _: fixture("search")),
        ("POST", f"{kb}/query", lambda _: fixture("query")),
        ("POST", f"{kb}/answer", lambda _: fixture("answer")),
        ("GET", f"{kb}/versions", lambda _: fixture("version_list")),
        ("POST", f"{kb}/versions", lambda _: httpx.Response(202, json=fixture("version_created"))),
        ("GET", f"{kb}/versions/4", lambda _: fixture("version_detail")),
        ("GET", f"{kb}/versions/4/diff", lambda _: fixture("version_diff")),
        ("GET", f"{kb}/versions/3/verify", lambda _: fixture("version_verify")),
        ("POST", f"{kb}/versions/4/approve", lambda _: _decision()),
        ("POST", f"{kb}/versions/4/reject", lambda _: _decision()),
        ("POST", f"{kb}/publish", lambda _: fixture("publication")),
        ("POST", f"{kb}/rollback", lambda _: fixture("publication")),
        ("GET", f"/v1/knowledge-jobs/{JOB}", lambda _: fixture("job")),
        (
            "GET",
            "/v1/knowledge/retrievals",
            lambda _: {"events": [fixture("retrieval_event")["event"]], "next_cursor": None},
        ),
        ("GET", f"/v1/knowledge/retrievals/{EVENT}", lambda _: fixture("retrieval_event")),
        ("GET", f"/v1/knowledge/retrievals/{EVENT}/snapshot", lambda _: fixture("snapshot")),
        ("GET", f"/v1/agents/{AGENT}/knowledge", lambda _: fixture("agent_knowledge")),
        ("PUT", f"/v1/agents/{AGENT}/knowledge", lambda _: fixture("agent_knowledge")),
        ("POST", f"/v1/agents/{AGENT}/knowledge/bindings", lambda _: _binding_result()),
        (
            "DELETE",
            f"/v1/agents/{AGENT}/knowledge/bindings/{BINDING}",
            lambda _: httpx.Response(204),
        ),
        ("POST", f"/v1/agents/{AGENT}/knowledge/search", lambda _: fixture("agent_search")),
    ]


class FakeKnowledgeApi:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.overrides: dict[tuple[str, str], Handler] = {}
        self._routes = {(method, path): handler for method, path, handler in _routes()}

    def override(self, method: str, path: str, handler: Handler) -> None:
        self.overrides[(method, path)] = handler

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        handler = self.overrides.get(key) or self._routes.get(key)
        if handler is None:
            return error_response("not_found", 404, f"no route {request.method} {request.url.path}")
        reply = handler(request)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> Client:
        return Client(api_key="agm_test", base_url=BASE, transport=self.transport())

    def last(self, method: Optional[str] = None, pattern: Optional[str] = None) -> httpx.Request:
        for request in reversed(self.requests):
            if method is not None and request.method != method:
                continue
            if pattern is not None and not re.search(pattern, request.url.path):
                continue
            return request
        raise AssertionError(f"no request {method} {pattern}")

    def paths(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]
