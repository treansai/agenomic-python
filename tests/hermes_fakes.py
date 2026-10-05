"""Offline fakes for the Hermes adapter tests.

* :class:`FakeAgenomic` - a threaded HTTP server speaking the runtime and
  supervisor wire shapes of agenomic-cloud ``docs/hermes/api.md``, with a
  scripted chat completions endpoint mounted at the Model Gateway path.
* :class:`FakeOpenAI` - a threaded OpenAI compatible server (stream and non
  stream) driven by the same :class:`ScriptedLLM`.
* :class:`FakeCtx` - a minimal Hermes ``PluginContext`` with a manager whose
  ``_hooks``/``_middleware`` registries mirror the real ones.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from agenomic.integrations.hermes.canonical import arguments_hash

RUNTIME = "/v1/hermes/runtime"
SUPERVISOR = "/v1/hermes/supervisor"


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


class ScriptedLLM:
    """Answers chat completions: first a scripted tool call, then a final text.

    ``tool_calls`` is a list of ``(name, arguments)`` emitted while the request
    has fewer tool results than calls already made (one call per turn).
    """

    def __init__(self, tool_calls: Optional[list[tuple[str, dict[str, Any]]]] = None) -> None:
        self.tool_calls = list(tool_calls or [])
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.requests.append(body)
        messages = body.get("messages") or []
        tool_results = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        has_tools = bool(body.get("tools"))
        if has_tools and len(tool_results) < len(self.tool_calls):
            name, args = self.tool_calls[len(tool_results)]
            return {
                "tool_calls": [
                    {
                        "id": f"call_{len(tool_results) + 1}_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ]
            }
        last = tool_results[-1].get("content") if tool_results else ""
        if isinstance(last, list):
            last = json.dumps(last)
        return {"content": f"final answer. last tool said: {str(last)[:300]}"}

    @staticmethod
    def completion(model: str, answer: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": answer.get("content")}
        finish = "stop"
        if answer.get("tool_calls"):
            message["tool_calls"] = answer["tool_calls"]
            finish = "tool_calls"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @staticmethod
    def chunks(model: str, answer: dict[str, Any]) -> list[dict[str, Any]]:
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }
        out: list[dict[str, Any]] = []
        if answer.get("tool_calls"):
            calls = [dict(c, index=i) for i, c in enumerate(answer["tool_calls"])]
            out.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "tool_calls": calls},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            out.append(
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
            )
        else:
            out.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer.get("content")},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            out.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        out.append(
            {
                **base,
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        return out


class _Server:
    def __init__(self, handler: Callable[[BaseHTTPRequestHandler, str, str, Any], None]) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                outer.on_request(self, method, self.path.split("?")[0], body)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                return  # clients probing and closing connections early are expected

        self.on_request = handler
        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def send_json(h: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    data = json.dumps(body).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def send_llm(h: BaseHTTPRequestHandler, llm: ScriptedLLM, body: dict[str, Any]) -> None:
    model = str(body.get("model") or "demo-model")
    answer = llm.respond(body)
    if body.get("stream"):
        payload = (
            b"".join(
                b"data: " + json.dumps(c).encode() + b"\n\n"
                for c in ScriptedLLM.chunks(model, answer)
            )
            + b"data: [DONE]\n\n"
        )
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Content-Length", str(len(payload)))
        h.end_headers()
        h.wfile.write(payload)
        return
    send_json(h, 200, ScriptedLLM.completion(model, answer))


class FakeOpenAI:
    """OpenAI compatible server at ``<url>/v1``."""

    def __init__(self, llm: Optional[ScriptedLLM] = None) -> None:
        self.llm = llm or ScriptedLLM()
        self.requests: list[Recorded] = []
        self.server = _Server(self._handle)

    @property
    def base_url(self) -> str:
        return self.server.url + "/v1"

    def _handle(self, h: BaseHTTPRequestHandler, method: str, path: str, body: Any) -> None:
        self.requests.append(Recorded(method, path, dict(h.headers), body))
        if method == "GET" and path.endswith("/models"):
            send_json(h, 200, {"object": "list", "data": [{"id": "demo-model", "object": "model"}]})
        elif method == "POST" and path.endswith("/chat/completions") and isinstance(body, dict):
            send_llm(h, self.llm, body)
        else:
            send_json(h, 404, {"error": {"code": "not_found", "message": path}})

    def close(self) -> None:
        self.server.close()


Decide = Callable[[dict[str, Any]], str]


@dataclass
class FakeAgenomic:
    """Runtime + supervisor API fake.

    ``decide(body) -> "allow" | "deny" | "require_approval"`` picks the
    authorize outcome; ``effective_state`` drives the mode.
    """

    effective_state: str = "enforce"
    decide: Decide = field(default=lambda body: "allow")
    authorize_delay_s: float = 0.0
    authorize_status: Optional[int] = None
    llm: ScriptedLLM = field(default_factory=ScriptedLLM)
    fail_events: int = 0
    heartbeat_interval_secs: int = 15
    auto_approve: bool = False

    def __post_init__(self) -> None:
        self.requests: list[Recorded] = []
        self.events: list[dict[str, Any]] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        self.pending_by_call: dict[str, str] = {}
        self.commands: list[dict[str, Any]] = []
        self.supervisor_commands: list[dict[str, Any]] = []
        self.acks: list[tuple[str, dict[str, Any]]] = []
        self.skills: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.server = _Server(self._handle)

    @property
    def url(self) -> str:
        return self.server.url

    @property
    def model_base_url(self) -> str:
        return self.url + RUNTIME + "/model/v1"

    def close(self) -> None:
        self.server.close()

    def calls(self, suffix: str, method: str = "POST") -> list[Recorded]:
        return [r for r in list(self.requests) if r.method == method and r.path.endswith(suffix)]

    def authorize_calls(self) -> list[Recorded]:
        return self.calls("/actions/authorize")

    def reports(self) -> list[Recorded]:
        return self.calls("/actions/report")

    def event_types(self) -> list[str]:
        return [str(e.get("type")) for e in list(self.events)]

    def approve(self, approval_id: str, status: str = "approved") -> None:
        self.approvals[approval_id]["status"] = status

    def _handle(self, h: BaseHTTPRequestHandler, method: str, path: str, body: Any) -> None:
        with self.lock:
            self.requests.append(Recorded(method, path, dict(h.headers), body))
        auth = h.headers.get("Authorization") or ""
        if path.startswith(SUPERVISOR):
            if not auth.startswith("Bearer agmhs_"):
                send_json(h, 401, {"error": {"code": "unauthorized", "message": "bad token"}})
                return
            self._supervisor(h, method, path[len(SUPERVISOR) :], body)
            return
        if not path.startswith(RUNTIME):
            send_json(h, 404, {"error": {"code": "not_found", "message": path}})
            return
        if not auth.startswith("Bearer agmhr_"):
            send_json(h, 401, {"error": {"code": "unauthorized", "message": "bad token"}})
            return
        self._runtime(h, method, path[len(RUNTIME) :], body if isinstance(body, dict) else {})

    def _supervisor(self, h: BaseHTTPRequestHandler, method: str, sub: str, body: Any) -> None:
        if sub == "/heartbeat":
            with self.lock:
                cmds, self.supervisor_commands = self.supervisor_commands, []
            send_json(h, 200, {"effective_state": self.effective_state, "commands": cmds})
        elif sub.startswith("/commands/") and sub.endswith("/ack"):
            cid = sub.split("/")[2]
            self.acks.append((cid, body))
            send_json(h, 200, {"id": cid, "status": body.get("status")})
        elif sub == "/skills/approved":
            send_json(h, 200, {"skills": self.skills})
        else:
            send_json(h, 404, {"error": {"code": "not_found", "message": sub}})

    def _runtime(
        self, h: BaseHTTPRequestHandler, method: str, sub: str, body: dict[str, Any]
    ) -> None:
        state = self.effective_state
        mode = state if state in ("observe", "shadow") else "enforce"
        if sub == "/hello":
            send_json(
                h,
                200,
                {
                    "instance_id": "inst-1",
                    "agent_ref": "agent://acme/hermes",
                    "environment": "test",
                    "requested_mode": mode,
                    "effective_state": state,
                    "protection": {},
                    "profile": {
                        "version": 1,
                        "digest": "sha256:x",
                        "document": {"protected_paths": []},
                    },
                    "heartbeat_interval_secs": self.heartbeat_interval_secs,
                    "catalog_digest": "sha256:y",
                },
            )
        elif sub == "/heartbeat":
            with self.lock:
                cmds = [c for c in self.commands if c.get("status") in ("requested", "received")]
            send_json(h, 200, {"effective_state": state, "requested_mode": mode, "commands": cmds})
        elif sub == "/tools/discovered":
            send_json(h, 200, {"entries": []})
        elif sub == "/sessions":
            sid = body.get("hermes_session_id")
            send_json(
                h,
                201,
                {
                    "session": {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(sid))),
                        "hermes_session_id": sid,
                        "status": "active",
                    },
                    "effective_state": state,
                    "mode": mode,
                    "protect_run_id": None,
                },
            )
        elif re.fullmatch(r"/sessions/[^/]+/end", sub):
            send_json(h, 200, {"session": {"status": body.get("status")}})
        elif re.fullmatch(r"/sessions/[^/]+/delegations", sub):
            if mode == "observe":
                send_json(h, 200, {"decision": "observe"})
            else:
                send_json(
                    h,
                    200,
                    {"decision": "allow", "delegation_id": str(uuid.uuid4()), "remaining": {}},
                )
        elif re.fullmatch(r"/sessions/[^/]+/actions/authorize", sub):
            self._authorize(h, body, mode)
        elif re.fullmatch(r"/sessions/[^/]+/actions/report", sub):
            permit = body.get("permit") or {}
            bound = (permit.get("document") or {}).get("arguments_hash")
            if bound and bound != arguments_hash(body.get("arguments")):
                send_json(
                    h, 403, {"error": {"code": "permit_invalid", "message": "permit does not bind"}}
                )
                return
            send_json(
                h,
                200,
                {
                    "record_id": (permit.get("document") or {}).get("record_id"),
                    "status": "reported",
                },
            )
        elif sub.startswith("/approvals/"):
            aid = sub.split("/")[2]
            approval = self.approvals.get(aid)
            if approval is None:
                send_json(h, 404, {"error": {"code": "not_found", "message": "approval"}})
            else:
                send_json(h, 200, approval)
        elif sub == "/proposals":
            send_json(h, 201, {"id": "prop-1", "status": "proposed", "proposer_kind": "runtime"})
        elif sub == "/events":
            if self.fail_events > 0:
                self.fail_events -= 1
                send_json(h, 503, {"error": {"code": "unavailable", "message": "try later"}})
                return
            events = body.get("events") or []
            with self.lock:
                seen = {e.get("event_id") for e in self.events}
                fresh = [e for e in events if e.get("event_id") not in seen]
                self.events.extend(fresh)
            send_json(
                h,
                200,
                {"accepted": len(fresh), "duplicates": len(events) - len(fresh), "rejected": []},
            )
        elif sub.startswith("/commands/") and sub.endswith("/ack"):
            cid = sub.split("/")[2]
            self.acks.append((cid, body))
            for c in self.commands:
                if c.get("id") == cid:
                    c["status"] = body.get("status")
            send_json(h, 200, {"id": cid, "status": body.get("status")})
        elif sub == "/model/v1/models" and method == "GET":
            send_json(h, 200, {"object": "list", "data": [{"id": "demo-model", "object": "model"}]})
        elif sub == "/model/v1/chat/completions":
            send_llm(h, self.llm, body)
        else:
            send_json(h, 404, {"error": {"code": "not_found", "message": sub}})

    def _authorize(self, h: BaseHTTPRequestHandler, body: dict[str, Any], mode: str) -> None:
        if self.authorize_delay_s:
            time.sleep(self.authorize_delay_s)
        if self.authorize_status is not None:
            send_json(
                h, self.authorize_status, {"error": {"code": "boom", "message": "server error"}}
            )
            return
        call_id = str(body.get("tool_call_id"))
        args_hash = arguments_hash(body.get("arguments"))
        base = {
            "effective_mode": mode,
            "counterfactual": None,
            "arguments_hash": args_hash,
            "logical_call_id": call_id,
            "attempt": body.get("attempt", 1),
            "action_id": f"act-{call_id}",
            "decision_id": f"dec-{uuid.uuid4().hex[:8]}",
            "reason_codes": [],
            "explanation": "",
            "record_id": None,
            "approval_id": None,
            "approval_expires_at": None,
            "permit": None,
        }
        if mode == "observe":
            send_json(h, 200, {**base, "decision": "observe"})
            return
        pending = self.pending_by_call.get(call_id)
        if pending is not None and self.approvals[pending]["status"] == "approved":
            self.approvals[pending]["status"] = "consumed"
            outcome = "allow"
        elif pending is not None and self.approvals[pending]["status"] == "pending":
            outcome = "require_approval"
        else:
            outcome = self.decide(body)
        if mode == "shadow":
            send_json(
                h,
                200,
                {
                    **base,
                    "decision": "allow",
                    "record_id": f"rec-{call_id}",
                    "counterfactual": {"outcome": outcome, "reason_codes": []},
                },
            )
            return
        if outcome == "allow":
            record = f"rec-{call_id}-{body.get('attempt', 1)}"
            permit = {
                "document": {
                    "schema_version": "agenomic.protect.permit/v1",
                    "record_id": record,
                    "tool": body.get("tool"),
                    "arguments_hash": args_hash,
                },
                "signature": {"alg": "ed25519", "value": "sig"},
            }
            send_json(h, 200, {**base, "decision": "allow", "record_id": record, "permit": permit})
        elif outcome == "require_approval":
            aid = pending or f"appr-{uuid.uuid4().hex[:8]}"
            self.approvals.setdefault(
                aid,
                {
                    "id": aid,
                    "status": "pending",
                    "expires_at": None,
                    "action_digest": args_hash,
                    "tool": body.get("tool"),
                },
            )
            self.pending_by_call[call_id] = aid
            send_json(h, 202, {**base, "decision": "require_approval", "approval_id": aid})
            if self.auto_approve:
                self.approvals[aid]["status"] = "approved"
        elif outcome == "invalid":
            send_json(h, 200, {"decision": "maybe"})
        else:
            send_json(
                h,
                403,
                {
                    **base,
                    "decision": "deny",
                    "reason_codes": ["policy_denied"],
                    "explanation": "writes are not allowed",
                },
            )


class FakeManager:
    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {}
        self._middleware: dict[str, list[Callable[..., Any]]] = {}
        self._plugin_tool_names: set[str] = set()


class FakeCtx:
    """Subset of ``hermes_cli.plugins.PluginContext`` the adapter uses."""

    def __init__(self, settings: Optional[dict[str, Any]] = None) -> None:
        self._manager = FakeManager()
        self.settings = settings or {}
        self.unload: list[Callable[[], None]] = []

    def register_hook(self, name: str, cb: Callable[..., Any]) -> None:
        self._manager._hooks.setdefault(name, []).append(cb)

    def register_middleware(self, kind: str, cb: Callable[..., Any]) -> None:
        self._manager._middleware.setdefault(kind, []).append(cb)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def on_unload(self, cb: Callable[[], None]) -> None:
        self.unload.append(cb)
