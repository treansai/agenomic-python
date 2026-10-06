from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, cast

import httpx
import pytest
from prompt_fakes import (
    AGENT,
    CHILD,
    OTHER_AGENT,
    OTHER_WORKSPACE,
    REQUEST_ID,
    SCOPE_MESSAGE,
    SESSION_MESSAGE,
    WORKSPACE,
    FakePromptServer,
    _if_match,
    _strict,
    chat_content,
    release_with_child,
    required,
    seeded_engine,
    text_content,
)

from agenomic import _transport
from agenomic._client import Client
from agenomic._transport import pool_for
from agenomic._version import __version__
from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts import (
    BundleTrust,
    PromptBindingError,
    PromptBundle,
    PromptCache,
    PromptConflictError,
    PromptImportError,
    PromptIntegrityError,
    PromptRefError,
    PromptUri,
    PromptVersionRef,
    ResolvedFrom,
    prompt_digest,
    thread_key,
)
from agenomic.prompts.importer import complete_content, load_prompts_file, plan_summary
from agenomic.prompts.resources import (
    Channel,
    ChannelEvent,
    ChannelMovePreview,
    Draft,
    ImportPlan,
    Page,
    PinnedRefs,
    PromptAlias,
    PromptSummary,
    RenderResult,
)

BASE = "https://api.test"
IMPORTS = Path(__file__).parent / "fixtures" / "prompt_imports"
SCHEMAS = Path(__file__).parent / "schemas" / "v0.4"
AT = "2026-10-05T12:00:00Z"
OBSERVATION_KEYS = {
    "slot_path",
    "node_path",
    "prompt_ref",
    "content_digest",
    "rendered_hash",
    "overlay",
    "alias",
    "alias_generation",
    "unmanaged",
    "role_layout",
    "count",
    "first_at",
    "last_at",
}
PROMPTS_FILE = f"""schema: agenomic.prompts_file/v1
agent_id: {AGENT}
prompts:
  - prompt_id: prm_support_writer
    kind: text
    name: Support writer
    tags: [yes, support]
    expected_latest_version: null
    change_message: first
    content:
      kind: text
      body: |-
        Answer {{question}} politely.
      variables:
        question: {{ type: string, required: true }}
slots:
  - slot_path: writer.response
    usage: system
    prompt_id: prm_support_writer
"""


@dataclass
class ImportServer(FakePromptServer):
    imports: dict[str, dict[str, Any]] = field(default_factory=dict)
    replays: dict[str, str] = field(default_factory=dict)
    journal: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    slot_revision: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)

    def _routes(self) -> list[tuple[str, str, Callable[..., httpx.Response]]]:
        return [
            (r"/v1/prompts/imports", "POST", self._create_import),
            (r"/v1/prompts/imports/([^/]+)/apply", "POST", self._apply_import),
            (r"/v1/prompts/declarations/plan", "POST", self._plan_declarations),
            (r"/v1/prompts/declarations/apply", "POST", self._apply_declarations),
            (r"/v1/agents/([^/]+)/bindings/([^/]+)/usage", "POST", self._usage),
            *super()._routes(),
        ]

    def _plan(
        self, kind: str, digest: str, items: list[Any], **header: Optional[str]
    ) -> dict[str, Any]:
        plan: dict[str, Any] = {
            "schema": "agenomic.prompt_import_plan/v1",
            "plan_id": header.get("plan_id"),
            "workspace_id": self.engine.workspace_id,
            "agent_id": header.get("agent_id"),
            "source": {"kind": kind, "digest": digest},
            "created_at": header.get("created_at"),
            "items": items,
            "summary": plan_summary(items),
        }
        plan["plan_digest"] = prompt_digest(plan)
        return plan

    def _action(self, prompt_id: str, digest: str) -> tuple[str, Optional[int], Optional[int]]:
        if prompt_id not in self.engine._state["prompts"]:
            return "create_prompt", None, None
        latest = self.engine.prompt(prompt_id)["latest_version"]
        if latest is not None and self.engine.get_version(prompt_id, latest).content_digest == (
            digest
        ):
            return "reuse_version", latest, latest
        return "create_version", latest, None

    def _write(self, item: dict[str, Any], prompt_id: str, action: str) -> dict[str, Any]:
        if action == "reuse_version":
            return {"outcome": "unchanged", "version": item["base_version"]}
        if action == "create_prompt":
            self.engine.create_prompt(prompt_id, kind=item["prompt_kind"], name=prompt_id)
        latest = self.engine.prompt(prompt_id)["latest_version"]
        version = self.engine.publish(
            prompt_id, item["content"], parent_version=latest, change_message="import"
        )
        outcome = "created" if action == "create_prompt" else "versioned"
        return {"outcome": outcome, "version": version.ref.version}

    def _replayed(self, key: str, digest: str) -> Optional[httpx.Response]:
        journaled = self.journal.get(key)
        if journaled is None:
            return None
        if journaled[0] != digest:
            raise ApiError("idempotency_key_reused", 409, "the key names another request")
        return httpx.Response(200, json={**journaled[1], "replayed": True})

    def _slots(self, request: httpx.Request, writes: bool) -> None:
        if writes and _if_match(request) != self.slot_revision:
            raise ApiError(
                "agent_prompt_slots_conflict", 409, "slots moved", {"current": self.slot_revision}
            )

    def _create_import(self, request: httpx.Request, body: Any) -> httpx.Response:
        _strict(body, {"report", "agent_id", "options"})
        self._require_editor()
        key = prompt_digest({"request": body})
        if key in self.replays:
            record = self.imports[self.replays[key]]["record"]
            return httpx.Response(200, json={"replayed": True, "import": record})
        report = body["report"]
        items = []
        for candidate in report["candidates"]:
            proposal, status = candidate["proposal"], candidate["status"]
            action, base, existing = "skip", None, None
            if status in ("unsupported", "blocked_secret"):
                action = "blocked"
            elif status == "supported":
                action, base, existing = self._action(
                    proposal["prompt_id"], candidate["content_digest"]
                )
            items.append(
                {
                    "item_id": candidate["candidate_id"],
                    "action": action,
                    "prompt_id": proposal["prompt_id"],
                    "prompt_kind": proposal["prompt_kind"],
                    "base_version": base,
                    "content": candidate["content"],
                    "content_digest": candidate["content_digest"],
                    "existing_version_with_same_digest": existing,
                    "slot": {
                        "slot_path": proposal["slot_path"],
                        "node_path": proposal["node_path"],
                        "subagent_id": None,
                        "usage": proposal["usage"],
                        "status": "unresolved" if status == "unresolved" else "discovered",
                    },
                    "issues": candidate["issues"],
                    "secret_findings": candidate["secret_findings"],
                    "provenance": {
                        "source_file": candidate["source"]["path"],
                        "source_line": candidate["source"]["line"],
                    },
                }
            )
        import_id = f"imp_01j9x1k3m5n7p9q1r3s5t7v9{len(self.imports):02d}"
        plan = self._plan(
            "discovery_report",
            prompt_digest(report),
            items,
            plan_id=import_id,
            agent_id=body.get("agent_id"),
            created_at=AT,
        )
        record = {
            "import_id": import_id,
            "status": "planned",
            "report_digest": prompt_digest(report),
            "expires_at": "2026-10-12T12:00:00Z",
            "plan": plan,
        }
        self.imports[import_id] = {"record": record, "applied": False}
        self.replays[key] = import_id
        return httpx.Response(201, json={"replayed": False, "import": record})

    def _apply_import(self, request: httpx.Request, body: Any, import_id: str) -> httpx.Response:
        allowed = {"idempotency_key", "plan_digest", "agent_id", "mode", "declare_slots", "items"}
        _strict(body, allowed, ("items",))
        self._require_editor()
        decisions = {}
        for decision in body["items"]:
            _strict(
                decision,
                {
                    "item_id",
                    "action",
                    "prompt_id",
                    "base_version",
                    "slot_path",
                    "subagent_id",
                    "override",
                },
            )
            decisions[decision["item_id"]] = decision
        stored = self.imports.get(import_id)
        if stored is None:
            raise ApiError("prompt_import_not_found", 404, "no such import")
        declare = body.get("declare_slots", False)
        if declare:
            _if_match(request)
        replay = self._replayed(body["idempotency_key"], prompt_digest(body))
        if replay is not None:
            return replay
        if stored["applied"]:
            raise ApiError("prompt_import_already_applied", 409, "the plan was applied")
        plan = stored["record"]["plan"]
        if body["plan_digest"] != plan["plan_digest"]:
            raise ApiError(
                "prompt_import_plan_stale",
                409,
                "plan_digest differs from the stored plan",
                {"current": plan["plan_digest"]},
            )
        self._slots(request, declare)
        if set(decisions) != {item["item_id"] for item in plan["items"]}:
            raise ApiError("validation_error", 400, "every plan item is listed once")
        results = []
        for item in plan["items"]:
            decision = decisions[item["item_id"]]
            action = decision["action"]
            if action == "skip":
                results.append({"item_id": item["item_id"], "outcome": "skipped"})
                continue
            if item["action"] in ("blocked", "skip") and "override" not in decision:
                raise ApiError(
                    "prompt_import_item_blocked",
                    409,
                    "the item cannot be imported",
                    {"item_id": item["item_id"]},
                )
            written = self._write(item, decision["prompt_id"], action)
            results.append(
                {
                    "item_id": item["item_id"],
                    "prompt_id": decision["prompt_id"],
                    "content_digest": item["content_digest"],
                    **written,
                }
            )
        slots = None
        if declare:
            self.slot_revision += 1
            agent = body.get("agent_id") or plan["agent_id"]
            slots = {"agent_id": agent, "revision": self.slot_revision}
        stored["applied"] = True
        answer = {"import_id": import_id, "replayed": False, "results": results, "slots": slots}
        self.journal[body["idempotency_key"]] = (prompt_digest(body), answer)
        return httpx.Response(200, json=answer)

    def _declared(self, document: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        items = []
        declared = document.get("slots") or []
        for entry in document["prompts"]:
            content = complete_content(entry["content"])
            action, base, existing = self._action(entry["prompt_id"], prompt_digest(content))
            if "expected_latest_version" in entry and entry["expected_latest_version"] != base:
                raise ApiError("prompt_import_plan_stale", 409, "expected_latest_version moved", {})
            slot = next((s for s in declared if s.get("prompt_id") == entry["prompt_id"]), None)
            items.append(
                {
                    "item_id": "cand_"
                    + hashlib.sha256(entry["prompt_id"].encode()).hexdigest()[:16],
                    "action": action,
                    "prompt_id": entry["prompt_id"],
                    "prompt_kind": entry["kind"],
                    "base_version": base,
                    "content": content,
                    "content_digest": prompt_digest(content),
                    "existing_version_with_same_digest": existing,
                    "slot": None
                    if slot is None
                    else {
                        "slot_path": slot["slot_path"],
                        "node_path": slot.get("node_path"),
                        "subagent_id": None,
                        "usage": slot.get("usage", "other"),
                        "status": "managed",
                    },
                    "issues": [],
                    "secret_findings": [],
                    "provenance": {"source_file": None, "source_line": None},
                }
            )
        plan = self._plan(
            "prompts_file", prompt_digest(document), items, agent_id=document.get("agent_id")
        )
        slots = None
        if isinstance(document.get("slots"), list) and isinstance(document.get("agent_id"), str):
            slots = {
                "agent_id": document["agent_id"],
                "revision": self.slot_revision,
                "added": [slot["slot_path"] for slot in declared],
                "removed": [],
                "changed": [],
            }
        return plan, slots

    def _plan_declarations(self, request: httpx.Request, body: Any) -> httpx.Response:
        _strict(body, {"document"})
        self._require_editor()
        plan, slots = self._declared(body["document"])
        return httpx.Response(200, json={"plan": plan, "slots": slots})

    def _apply_declarations(self, request: httpx.Request, body: Any) -> httpx.Response:
        _strict(body, {"idempotency_key", "document", "plan_digest"})
        self._require_editor()
        document = body["document"]
        writes = isinstance(document.get("slots"), list) and isinstance(
            document.get("agent_id"), str
        )
        if writes:
            _if_match(request)
        replay = self._replayed(body["idempotency_key"], prompt_digest(body))
        if replay is not None:
            return replay
        plan, _ = self._declared(document)
        if plan["plan_digest"] != body["plan_digest"]:
            raise ApiError(
                "prompt_import_plan_stale",
                409,
                "the plan computed now differs from the cited plan_digest",
                {"plan_digest": plan["plan_digest"]},
            )
        self._slots(request, writes)
        results = [
            {
                "item_id": item["item_id"],
                "prompt_id": item["prompt_id"],
                **self._write(item, item["prompt_id"], item["action"]),
            }
            for item in plan["items"]
        ]
        slots = None
        if writes:
            self.slot_revision += 1
            slots = {"agent_id": document["agent_id"], "revision": self.slot_revision}
        answer = {
            "replayed": False,
            "plan_digest": body["plan_digest"],
            "results": results,
            "slots": slots,
        }
        self.journal[body["idempotency_key"]] = (prompt_digest(body), answer)
        return httpx.Response(200, json=answer)

    def _usage(
        self, request: httpx.Request, body: Any, agent_id: str, binding_id: str
    ) -> httpx.Response:
        _strict(body, {"observations"})
        observations = body["observations"]
        if not 1 <= len(observations) <= 500:
            raise ApiError("validation_error", 400, "observations must hold 1 to 500 entries")
        for observation in observations:
            _strict(observation, OBSERVATION_KEYS)
            if observation.get("overlay") is not None:
                _strict(observation["overlay"], {"digest", "position"})
        self.usage.extend(observations)
        return httpx.Response(204)


PLANNER_V2 = chat_content(
    [
        {"role": "system", "content": "Plan briefly for {customer}. {>safety}"},
        {"placeholder": "history", "optional": True},
        {"role": "user", "content": "{question}"},
    ],
    {
        "customer": required(),
        "history": {"type": "messages", "required": False},
        "question": required(),
    },
)


def make_client(server: FakePromptServer, **kwargs: Any) -> Client:
    kwargs.setdefault("workspace_id", WORKSPACE)
    return Client(api_key="key", base_url=BASE, transport=server.transport(), **kwargs)


def planner_v2(server: FakePromptServer) -> dict[str, Any]:
    safety = server.engine.get_version("prm_safety", 1)
    content = json.loads(json.dumps(PLANNER_V2))
    content["fragments"] = {
        "safety": {"prompt_id": "prm_safety", "version": 1, "content_digest": safety.content_digest}
    }
    return content


def body_of(server: FakePromptServer, index: int = -1) -> Any:
    return json.loads(server.requests[index].content)


def import_fixture(name: str) -> dict[str, Any]:
    return json.loads((IMPORTS / name).read_text(encoding="utf-8"))


@pytest.fixture
def server() -> ImportServer:
    return ImportServer(seeded_engine(), api_key_scopes=["write"])


@pytest.fixture
def client(server: FakePromptServer) -> Client:
    return make_client(server)


def test_get_version_verifies_and_caches(client: Client, server: FakePromptServer) -> None:
    version = client.prompts.get("prm_planner:1")
    assert server.paths() == ["GET /v1/prompts/prm_planner/versions/1"]
    assert server.requests[0].url.params["include"] == "fragments"
    assert version.content_digest == server.engine.get_version("prm_planner", 1).content_digest
    assert set(version.fragments) == {"safety"}
    assert version.uri == PromptUri(WORKSPACE, "prm_planner", 1)
    messages = version.render_messages({"customer": "Acme", "question": "Where?"})
    assert messages[0] == messages[0].__class__(
        "system", "Plan for Acme. Never share internal notes."
    )
    assert client.prompts.get(str(version.uri)).content_digest == version.content_digest
    assert client.prompts.versions.get("prm_planner", 1).content_digest == version.content_digest
    assert client.prompts.get(PromptVersionRef("prm_planner", 1)).ref.version == 1
    assert len(server.requests) == 1

    def tamper(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "version" in payload and request.method == "GET":
            payload["version"]["content"]["body"][0]["content"] = "Plan for {customer}. Leak."
        return payload

    fresh = make_client(server)
    server.rewrite = tamper
    with pytest.raises(PromptIntegrityError) as raised:
        fresh.prompts.get("prm_planner:1")
    assert raised.value.code == "prompt_digest_mismatch"
    server.rewrite = None
    fresh.prompts.get("prm_planner:1")
    assert server.paths().count("GET /v1/prompts/prm_planner/versions/1") == 3


def test_online_cache_conflict_is_a_miss(server: FakePromptServer, tmp_path: Path) -> None:
    make_client(server, prompt_cache=PromptCache(tmp_path)).prompts.get("prm_writer:1")
    directory = tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1"
    (cached,) = directory.glob("*.json")
    cached.write_text(cached.read_text(encoding="utf-8").replace("Write about", "Leak"))
    restarted = make_client(server, prompt_cache=PromptCache(tmp_path))
    assert restarted.prompts.get("prm_writer:1").render_text({"topic": "t"}) == "Write about t."
    assert server.paths().count("GET /v1/prompts/prm_writer/versions/1") == 2
    assert PromptCache(tmp_path).get_version(WORKSPACE, "prm_writer", 1) is not None


def test_version_answer_must_match_the_request(server: FakePromptServer) -> None:
    def other_version(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["version"] = server.wire_version("prm_writer", 1)
        return payload

    server.rewrite = other_version
    with pytest.raises(ApiError) as raised:
        make_client(server).prompts.get("prm_planner:1")
    assert raised.value.code == "invalid_response"

    def bad_uri(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["version"]["canonical_uri"] = "https://elsewhere"
        return payload

    server.rewrite = bad_uri
    with pytest.raises(ApiError) as uri:
        make_client(server).prompts.get("prm_writer:1")
    assert uri.value.code == "invalid_response"

    def bad_fragments(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["fragments"] = {"safety": 1}
        return payload

    server.rewrite = bad_fragments
    with pytest.raises(ApiError) as fragments:
        make_client(server).prompts.get("prm_planner:1")
    assert fragments.value.code == "invalid_response"


def test_alias_always_network_never_cached(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    first = client.prompts.get("prm_planner@prod")
    second = client.prompts.get("prm_planner@prod")
    assert first.ref == second.ref == PromptVersionRef("prm_planner", 1)
    assert first.resolved_from == ResolvedFrom("prod", 1)
    resolves = [request for request in server.requests if request.url.path == "/v1/prompts/resolve"]
    assert len(resolves) == 2
    assert json.loads(resolves[0].content) == {"ref": "prm_planner@prod"}
    assert "idempotency-key" not in resolves[0].headers
    assert server.paths().count("GET /v1/prompts/prm_planner/versions/1") == 1
    engine.publish("prm_planner", planner_v2(server), parent_version=1, change_message="two")
    engine.move_alias("prm_planner", "prod", version=2, expected_generation=1)
    moved = client.prompts.resolve("prm_planner@prod")
    assert moved.ref.version == 2
    assert moved.resolved_from == ResolvedFrom("prod", 2)
    assert client.prompts.get("prm_planner:1").resolved_from is None

    def wrong_digest(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path == "/v1/prompts/resolve":
            payload["content_digest"] = "sha256:" + "0" * 64
        return payload

    server.rewrite = wrong_digest
    with pytest.raises(PromptIntegrityError) as raised:
        client.prompts.get("prm_planner@prod")
    assert raised.value.code == "prompt_digest_mismatch"

    def no_alias(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path == "/v1/prompts/resolve":
            payload["alias"] = None
        return payload

    server.rewrite = no_alias
    with pytest.raises(ApiError) as invalid:
        client.prompts.get("prm_planner@prod")
    assert invalid.value.code == "invalid_response"


def test_draft_save_if_match_conflict(client: Client, server: FakePromptServer) -> None:
    content = text_content("Draft about {topic}.", {"topic": required()})
    saved = client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    put = server.requests[-1]
    assert put.method == "PUT"
    assert put.headers["if-match"] == '"0"'
    assert body_of(server) == {"base_version": 1, "content": content}
    assert isinstance(saved, Draft)
    assert (saved.revision, saved.base_version, saved.origin) == (1, 1, "editor")
    with pytest.raises(PromptConflictError) as raised:
        client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    assert raised.value.code == "prompt_draft_conflict"
    assert raised.value.status == 409
    assert raised.value.details["current"] == 1
    assert raised.value.request_id == REQUEST_ID
    assert client.prompts.drafts.get("prm_writer").revision == 1
    updated = client.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=1)
    assert updated.revision == 2

    def stale_etag(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["draft"]["revision"] = 7
        return payload

    server.rewrite = stale_etag
    with pytest.raises(ApiError) as mismatch:
        client.prompts.drafts.get("prm_writer")
    assert mismatch.value.code == "invalid_response"


def test_error_codes_map_to_classes(client: Client, server: FakePromptServer) -> None:
    with pytest.raises(ApiError) as missing:
        client.prompts.get("prm_planner:9")
    assert type(missing.value) is ApiError
    assert (missing.value.code, missing.value.status) == ("prompt_version_not_found", 404)
    with pytest.raises(PromptConflictError) as stale:
        client.prompts.publish(
            "prm_writer", text_content("Other."), parent_version=None, change_message="x"
        )
    assert stale.value.code == "prompt_version_conflict"
    root, _ = release_with_child(server.engine)
    key = thread_key(WORKSPACE, "conflict")
    client.bindings.create(AGENT, thread_key=key, scope="thread", release_id=root)
    with pytest.raises(PromptBindingError) as conflict:
        client.bindings.create(AGENT, thread_key=key, scope="execution", release_id=root)
    assert conflict.value.code == "execution_binding_conflict"
    assert conflict.value.details["release_id"] == root
    server.engine.set_release_status(root, "rejected")
    with pytest.raises(PromptBindingError) as unbindable:
        client.bindings.create(
            AGENT, thread_key=thread_key(WORKSPACE, "x"), scope="thread", release_id=root
        )
    assert unbindable.value.code == "release_not_bindable"


def test_read_key_publish_refused_verbatim(client: Client, server: FakePromptServer) -> None:
    server.api_key_scopes = ["read"]
    calls: list[Callable[[], Any]] = [
        lambda: client.prompts.create("prm_new", name="New", kind="text"),
        lambda: client.prompts.publish(
            "prm_writer",
            text_content("Other {topic}.", {"topic": required()}),
            parent_version=1,
            change_message="x",
        ),
        lambda: client.prompts.drafts.save(
            "prm_writer", text_content("x"), base_version=1, expected_revision=0
        ),
        lambda: client.prompts.import_report(import_fixture("discovery-report.json")),
        lambda: client.prompts.apply_import(
            "imp_01j9x1k3m5n7p9q1r3s5t7v9w1", plan_digest="sha256:" + "0" * 64, items=[]
        ),
        lambda: client.prompts.plan_declarations(PROMPTS_FILE),
        lambda: client.prompts.apply_declarations(
            PROMPTS_FILE, plan_digest="sha256:" + "0" * 64, expected_slots_revision=0
        ),
    ]
    for call in calls:
        with pytest.raises(ApiError) as raised:
            call()
        assert type(raised.value) is ApiError
        assert (raised.value.code, raised.value.status) == ("api_key_scope_insufficient", 403)
        assert raised.value.message == SCOPE_MESSAGE
        assert raised.value.request_id == REQUEST_ID
    assert "prm_new" not in server.engine._state["prompts"]
    assert "prm_support_writer" not in server.engine._state["prompts"]
    assert server.engine.prompt("prm_writer")["latest_version"] == 1
    assert len(server.requests) == 7
    for scopes in (["read"], ["write"], []):
        server.api_key_scopes = scopes
        with pytest.raises(ApiError) as alias:
            client.prompts.aliases.move("prm_writer", "prod", version=1, expected_generation=0)
        assert (alias.value.code, alias.value.status) == ("session_required", 403)
        assert alias.value.message == SESSION_MESSAGE
        assert server.requests[-1].headers["if-match"] == '"0"'
    server.api_key_scopes = ["read"]
    assert client.prompts.get("prm_writer:1").ref.version == 1


def test_alias_in_managed_run_refused(client: Client, server: FakePromptServer) -> None:
    config = pytest.importorskip("langchain_core.runnables.config")
    server.engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    local = Client(workspace_id=WORKSPACE)
    token = config.var_child_runnable_config.set(
        {"configurable": {"__agenomic_prompt_set": object()}}
    )
    try:
        calls: list[Callable[[], Any]] = [
            lambda: client.prompts.get("prm_planner@prod"),
            lambda: client.prompts.resolve("prm_planner@prod"),
            lambda: client.prompts.pin(["prm_planner:1", "prm_planner@prod"]),
            lambda: local.prompts.get("prm_planner@prod"),
        ]
        for call in calls:
            with pytest.raises(PromptRefError) as raised:
                call()
            assert raised.value.code == "alias_in_managed_run"
        assert all(request.url.path != "/v1/prompts/resolve" for request in server.requests)
        assert client.prompts.get("prm_planner:1").ref.version == 1
    finally:
        config.var_child_runnable_config.reset(token)
    other = config.var_child_runnable_config.set({"configurable": {"thread_id": "t"}})
    try:
        assert client.prompts.get("prm_planner@prod").resolved_from == ResolvedFrom("prod", 1)
    finally:
        config.var_child_runnable_config.reset(other)


def test_pin_resolves_alias_once(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    pinned = client.prompts.pin(["prm_planner@prod", "prm_writer:1", "prm_planner@prod"])
    assert isinstance(pinned, PinnedRefs)
    assert list(pinned) == ["prm_planner@prod", "prm_writer:1"]
    assert len(pinned) == 2
    assert sum(request.url.path == "/v1/prompts/resolve" for request in server.requests) == 1
    engine.publish("prm_planner", planner_v2(server), parent_version=1, change_message="two")
    engine.move_alias("prm_planner", "prod", version=2, expected_generation=1)
    assert pinned["prm_planner@prod"].ref.version == 1
    assert pinned["prm_planner@prod"].resolved_from == ResolvedFrom("prod", 1)
    with pytest.raises(TypeError):
        client.prompts.pin("prm_planner@prod")


def test_cross_workspace_uri_never_reaches_the_server(
    client: Client, server: FakePromptServer
) -> None:
    foreign = PromptUri(OTHER_WORKSPACE, "prm_planner", 1)
    for ref in (str(foreign), foreign):
        with pytest.raises(PromptRefError) as raised:
            client.prompts.get(ref)
        assert raised.value.code == "prompt_ref_cross_workspace"
    assert server.requests == []


def test_workspace_is_learned_once_and_checked(server: FakePromptServer) -> None:
    learned = make_client(server, workspace_id=None)
    assert learned.workspace_id is None
    learned.prompts.get("prm_writer:1")
    learned.prompts.get("prm_planner:1")
    assert learned.workspace_id == WORKSPACE
    assert learned.whoami()["api_key_scopes"] == ["write"]
    assert server.paths().count("GET /v1/whoami") == 1
    mismatched = make_client(server, workspace_id=OTHER_WORKSPACE)
    with pytest.raises(PromptRefError) as canonical:
        mismatched.prompts.get("prm_writer:1")
    assert canonical.value.code == "workspace_mismatch"
    with pytest.raises(PromptRefError) as identity:
        mismatched.whoami()
    assert identity.value.code == "workspace_mismatch"
    for call in (lambda: mismatched.workspace_id, mismatched.whoami):
        with pytest.raises(PromptRefError):
            call()
    with pytest.raises(ValueError):
        Client(workspace_id="NOT-A-UUID")

    def no_org(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["org_id"] = "nope"
        return payload

    server.rewrite = no_org
    with pytest.raises(ApiError) as invalid:
        make_client(server, workspace_id=None).whoami()
    assert invalid.value.code == "invalid_response"


def test_create_publish_list_and_versions(client: Client, server: FakePromptServer) -> None:
    created = client.prompts.create(
        "prm_notes", name="Notes", kind="text", description="Notes", tags=["support"]
    )
    assert body_of(server) == {
        "prompt_id": "prm_notes",
        "kind": "text",
        "name": "Notes",
        "tags": ["support"],
        "description": "Notes",
    }
    assert isinstance(created, PromptSummary)
    assert (created.metadata_revision, created.latest_version) == (1, None)
    content = text_content("Note on {topic}.", {"topic": required()})
    first = client.prompts.publish(
        "prm_notes",
        content,
        parent_version=None,
        change_message="first",
        variable_descriptions={"topic": "What the note covers."},
    )
    assert body_of(server) == {
        "content": content,
        "parent_version": None,
        "change_message": "first",
        "variable_descriptions": {"topic": "What the note covers."},
    }
    assert server.requests[-1].headers.get("idempotency-key") is None
    assert first.ref == PromptVersionRef("prm_notes", 1)
    again = client.prompts.publish(
        "prm_notes", content, parent_version=None, change_message="first"
    )
    assert again.content_digest == first.content_digest
    assert "variable_descriptions" not in body_of(server)
    count = len(server.requests)
    assert client.prompts.get("prm_notes:1").ref.version == 1
    assert len(server.requests) == count
    second = client.prompts.publish(
        "prm_planner", planner_v2(server), parent_version=1, change_message="two"
    )
    assert second.ref.version == 2
    assert set(second.fragments) == {"safety"}
    assert server.paths()[-1] == "GET /v1/prompts/prm_planner/versions/2"
    page = client.prompts.list(query="prm", tags=["support"], kind="text", limit=1)
    assert dict(server.requests[-1].url.params) == {
        "status": "active",
        "limit": "1",
        "q": "prm",
        "kind": "text",
        "tags": "support",
    }
    assert isinstance(page, Page)
    assert [item.prompt_id for item in page.items] == ["prm_notes"]
    assert page.next_cursor is None
    first_page = client.prompts.list(limit=2)
    assert first_page.next_cursor == "2"
    rest = client.prompts.list(limit=2, cursor=first_page.next_cursor, status="all")
    assert [item.prompt_id for item in first_page.items + rest.items] == [
        "prm_notes",
        "prm_planner",
        "prm_safety",
        "prm_writer",
    ]
    versions = client.prompts.versions.list("prm_planner", limit=1)
    assert [item.version for item in versions.items] == [2]
    assert versions.next_cursor == "1"
    older = client.prompts.versions.list("prm_planner", cursor=versions.next_cursor)
    assert [item.version for item in older.items] == [1]


def test_render_returns_the_rendered_hash(client: Client) -> None:
    text = client.prompts.render("prm_writer:1", {"topic": "tides"})
    assert isinstance(text, RenderResult)
    assert (text.kind, text.text, text.messages) == ("text", "Write about tides.", None)
    assert text.ref == PromptVersionRef("prm_writer", 1)
    assert text.rendered_hash.startswith("sha256:")
    chat = client.prompts.render("prm_planner:1", {"customer": "Acme", "question": "Why?"})
    assert chat.kind == "chat"
    assert chat.messages is not None
    assert len(chat.messages) == 2


def test_bindings_create_get_and_counterfactual(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    root, child = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    key = thread_key(WORKSPACE, "thread-1")
    binding, bundle, created = client.bindings.create(
        AGENT, thread_key=key, scope="thread", channel="production"
    )
    assert body_of(server) == {
        "thread_key": key,
        "scope": "thread",
        "selector": {"channel": "production"},
        "runtime_client": {
            "sdk": "agenomic-python",
            "sdk_version": __version__,
            "adapter": None,
            "adapter_version": None,
        },
        "include": ["artifacts"],
    }
    assert "idempotency-key" not in server.requests[-1].headers
    assert created
    assert binding.resolved_from == {"channel": "production", "generation": 1}
    assert bundle.prompt_manifest_digest == binding.prompt_manifest_digest
    assert bundle.version("planner.instructions").ref == PromptVersionRef("prm_planner", 1)
    assert bundle.version("writer.response", agent_id=CHILD).ref.prompt_id == "prm_writer"
    again, _, created_again = client.bindings.create(
        AGENT,
        thread_key=key,
        scope="thread",
        channel="production",
        child_selectors={CHILD: {"channel": "production"}},
        expect_manifest_digest=binding.prompt_manifest_digest,
        runtime_client={"adapter": "langgraph", "adapter_version": "1.2.11"},
    )
    sent = body_of(server)
    assert sent["child_selectors"] == {CHILD: {"channel": "production"}}
    assert sent["expect"] == {"prompt_manifest_digest": binding.prompt_manifest_digest}
    assert sent["runtime_client"]["adapter"] == "langgraph"
    assert not created_again
    assert again.binding_id == binding.binding_id
    fetched, fetched_bundle = client.bindings.get(AGENT, binding.binding_id)
    assert server.requests[-1].url.params["include"] == "artifacts"
    assert fetched == binding
    assert fetched_bundle.prompt_bundle_digest == bundle.prompt_bundle_digest
    cf_key = "cf:sha256:" + "4" * 64
    counterfactual = client.bindings.counterfactual(
        AGENT, binding.binding_id, thread_key=cf_key, release_id=root
    )
    assert body_of(server) == {"thread_key": cf_key, "selector": {"release_id": root}}
    assert counterfactual.parent_binding_id == binding.binding_id
    for selector in ({}, {"channel": "production", "release_id": root}):
        with pytest.raises(ValueError):
            client.bindings.create(AGENT, thread_key=key, scope="thread", **selector)
    with pytest.raises(ValueError):
        client.bindings.create(
            AGENT, thread_key=key, scope="thread", channel="production", runtime_client={"x": "y"}
        )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p, root, child: p["binding"].update(agent_id=OTHER_AGENT), "binding_mismatch"),
        (lambda p, root, child: p["binding"].update(thread_key="other"), "binding_mismatch"),
        (lambda p, root, child: p["binding"].update(release_id=child), "binding_mismatch"),
        (
            lambda p, root, child: p["binding"]["children"][CHILD].update(
                prompt_manifest_digest="sha256:" + "0" * 64
            ),
            "manifest_digest_mismatch",
        ),
        (
            lambda p, root, child: p["binding"]["children"].pop(CHILD),
            "manifest_digest_mismatch",
        ),
        (lambda p, root, child: p.pop("artifacts"), "invalid_response"),
        (lambda p, root, child: p.update(created="yes"), "invalid_response"),
        (lambda p, root, child: p["binding"].pop("binding_id"), "invalid_response"),
    ],
)
def test_binding_answers_are_checked(
    server: FakePromptServer, mutate: Callable[[dict[str, Any], str, str], Any], code: str
) -> None:
    root, child = release_with_child(server.engine)

    def rewrite(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        mutate(payload, root, child)
        return payload

    server.rewrite = rewrite
    with pytest.raises(ApiError) as raised:
        make_client(server).bindings.create(
            AGENT, thread_key=thread_key(WORKSPACE, "t"), scope="thread", release_id=root
        )
    assert raised.value.code == code


def test_channels_are_read_only(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    engine.move_channel(AGENT, "staging", root, expected_generation=0)
    channels = client.channels.list(AGENT)
    assert [channel.name for channel in channels] == ["production", "staging"]
    assert all(isinstance(channel, Channel) for channel in channels)
    production = client.channels.get(AGENT, "production")
    assert (production.generation, production.protected, production.release_id) == (1, True, root)
    assert production.release is not None
    assert production.release["status"] == "production"
    assert client.channels.get(OTHER_AGENT, "production").materialized is False
    history = client.channels.history(AGENT, "production")
    assert isinstance(history[0], ChannelEvent)
    assert (history[0].action, history[0].to_release_id) == ("promote", root)
    assert client.channels.history(AGENT, "production", after=1) == []
    assert server.requests[-1].url.params["after"] == "1"
    preview = client.channels.move_preview(AGENT, "production", release_id=root)
    assert isinstance(preview, ChannelMovePreview)
    assert preview.actions["promote"] is False
    assert dict(server.requests[-1].url.params) == {"action": "promote", "release_id": root}
    client.channels.move_preview(AGENT, "production", action="rollback", release_id=root)
    assert dict(server.requests[-1].url.params) == {"action": "rollback", "to_release_id": root}
    with pytest.raises(ValueError):
        client.channels.move_preview(AGENT, "production")
    with pytest.raises(ValueError):
        client.channels.move_preview(AGENT, "production", action="delete", release_id=root)
    assert not hasattr(client.channels, "promote")
    assert not hasattr(client.channels, "rollback")

    def stale(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "channel" in payload:
            payload["channel"]["generation"] = 9
        return payload

    server.rewrite = stale
    with pytest.raises(ApiError) as mismatch:
        client.channels.get(AGENT, "production")
    assert mismatch.value.code == "invalid_response"


def test_channel_history_follows_next_after(client: Client, server: FakePromptServer) -> None:
    engine = server.engine
    first, _ = release_with_child(engine)
    second = engine.create_release(AGENT, {"planner.instructions": "prm_planner:1"})
    for generation in range(120):
        target = first if generation % 2 == 0 else second
        engine.move_channel(AGENT, "staging", target, expected_generation=generation)
    history = client.channels.history(AGENT, "staging")
    assert [event.generation for event in history] == list(range(1, 121))
    pages = [request.url.params["after"] for request in server.requests]
    assert pages == ["0", "50", "100"]
    assert [
        event.generation for event in client.channels.history(AGENT, "staging", after=110)
    ] == list(range(111, 121))

    def stuck(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["next_after"] = int(request.url.params["after"])
        return payload

    server.rewrite = stuck
    with pytest.raises(ApiError) as raised:
        client.channels.history(AGENT, "staging")
    assert raised.value.code == "invalid_response"


def test_resolve_agent_and_export_bundle(
    client: Client, server: FakePromptServer, tmp_path: Path
) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    resolved = client.prompts.resolve_agent(AGENT, channel="production")
    assert dict(server.requests[-1].url.params) == {"channel": "production"}
    assert not resolved.signed
    assert resolved.release_id == root
    target = tmp_path / "bundle.json"
    exported = client.prompts.export_bundle(
        AGENT, channel="production", expires_in_days=7, path=target
    )
    assert server.paths()[-2:] == [
        f"GET /v1/agents/{AGENT}/prompt-bundle",
        f"GET /v1/signing-keys/{server.signer.key_id}",
    ]
    assert dict(server.requests[-2].url.params) == {"channel": "production", "expires_in_days": "7"}
    assert exported.signed
    assert exported.governance is not None
    assert exported.governance["approved"] is True
    trust = BundleTrust.from_pems({server.signer.key_id: server.signer.public_pem()})
    reloaded = PromptBundle.load(
        target, expected_workspace_id=WORKSPACE, expected_agent_id=AGENT, trust=trust
    )
    assert reloaded.prompt_bundle_digest == exported.prompt_bundle_digest
    count = len(server.requests)
    pinned = client.prompts.export_bundle(AGENT, release_id=root, trust=trust)
    assert pinned.release_id == root
    assert len(server.requests) == count + 1
    stranger = SigningKey.generate("orgkey_other")
    with pytest.raises(PromptIntegrityError) as untrusted:
        client.prompts.export_bundle(
            AGENT,
            release_id=root,
            trust=BundleTrust.from_pems({stranger.key_id: stranger.public_pem()}),
        )
    assert untrusted.value.code == "bundle_untrusted_key"
    candidate = engine.create_release(
        AGENT, {"planner.instructions": "prm_planner:1"}, status="awaiting_approval"
    )
    engine.move_channel(AGENT, "staging", candidate, expected_generation=0)
    staged = client.prompts.export_bundle(AGENT, channel="staging")
    assert staged.governance is not None
    assert staged.governance["approved"] is False
    with pytest.raises(PromptBindingError) as ungoverned:
        client.prompts.resolve_agent(
            AGENT, release_id=engine.create_release(AGENT, {}, status="awaiting_approval")
        )
    assert ungoverned.value.code == "session_required"

    def unsigned(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.endswith("/prompt-bundle"):
            payload.pop("issuer")
        if request.url.path.startswith("/v1/signing-keys/"):
            payload["key_id"] = "orgkey_swapped"
        return payload

    server.rewrite = unsigned
    with pytest.raises(PromptIntegrityError) as missing:
        client.prompts.export_bundle(AGENT, channel="production")
    assert missing.value.code == "bundle_signature_invalid"

    def swapped(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.path.startswith("/v1/signing-keys/"):
            payload["key_id"] = "orgkey_swapped"
        return payload

    server.rewrite = swapped
    with pytest.raises(ApiError) as invalid:
        client.prompts.export_bundle(AGENT, channel="production")
    assert invalid.value.code == "invalid_response"

    def other_agent(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["agent_id"] = OTHER_AGENT
        return payload

    server.rewrite = other_agent
    with pytest.raises(ApiError) as foreign:
        client.prompts.resolve_agent(AGENT, channel="production")
    assert foreign.value.code == "invalid_response"


def test_local_mode_uses_the_engine(client: Client) -> None:
    local = Client(workspace_id=WORKSPACE)
    engine = local.prompts.local
    assert engine.workspace_id == local.workspace_id == WORKSPACE
    assert not local.is_cloud
    created = local.prompts.create("prm_writer", name="Writer", kind="text")
    assert isinstance(created, PromptSummary)
    content = text_content("Write about {topic}.", {"topic": required()})
    version = local.prompts.publish(
        "prm_writer", content, parent_version=None, change_message="first"
    )
    assert local.prompts.get("prm_writer:1").content_digest == version.content_digest
    assert local.prompts.versions.get("prm_writer", 1).ref.version == 1
    assert local.prompts.render("prm_writer:1", {"topic": "tides"}).text == "Write about tides."
    draft = local.prompts.drafts.save("prm_writer", content, base_version=1, expected_revision=0)
    assert local.prompts.drafts.get("prm_writer").revision == draft.revision == 1
    alias = local.prompts.aliases.move("prm_writer", "prod", version=1, expected_generation=0)
    assert isinstance(alias, PromptAlias)
    assert local.prompts.aliases.get("prm_writer", "prod").generation == 1
    assert local.prompts.get("prm_writer@prod").resolved_from == ResolvedFrom("prod", 1)
    release = engine.create_release(AGENT, {"writer.main": "prm_writer:1"})
    engine.move_channel(AGENT, "production", release, expected_generation=0)
    binding, bundle, created_binding = local.bindings.create(
        AGENT, thread_key=thread_key(WORKSPACE, "t"), scope="thread", channel="production"
    )
    assert created_binding
    assert bundle.version("writer.main").ref.prompt_id == "prm_writer"
    assert local.bindings.get(AGENT, binding.binding_id)[0] == binding
    assert local.channels.get(AGENT, "production").generation == 1
    assert local.channels.history(AGENT, "production")[0].action == "promote"
    assert local.channels.history(AGENT, "production", after=1) == []
    assert local.prompts.resolve_agent(AGENT, channel="production").release_id == release
    unsupported: list[Callable[[], Any]] = [
        local.prompts.list,
        lambda: local.prompts.versions.list("prm_writer"),
        lambda: local.prompts.export_bundle(AGENT, channel="production"),
        lambda: local.channels.list(AGENT),
        lambda: local.channels.move_preview(AGENT, "production", release_id=release),
        lambda: local.bindings.counterfactual(
            AGENT, binding.binding_id, thread_key="cf:x", release_id=release
        ),
        lambda: local.bindings.create(
            AGENT,
            thread_key="t2",
            scope="thread",
            channel="production",
            child_selectors={CHILD: {"channel": "production"}},
        ),
        local.whoami,
        lambda: client.prompts.local,
        lambda: local.prompts.import_report(import_fixture("discovery-report.json")),
        lambda: local.prompts.apply_import("imp_x", plan_digest="sha256:x", items=[]),
        lambda: local.prompts.plan_declarations(PROMPTS_FILE),
        lambda: local.prompts.apply_declarations(PROMPTS_FILE, plan_digest="sha256:x"),
        lambda: local.prompts.register_runtime(AGENT, {"writer.system": object()}),
        lambda: local.bindings.report_usage(AGENT, binding.binding_id, []),
    ]
    for call in unsupported:
        with pytest.raises(ApiError) as raised:
            call()
        assert raised.value.code == "cloud_required"


def test_from_env_and_lifecycle(
    server: FakePromptServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENOMIC_ENDPOINT", BASE + "/")
    monkeypatch.setenv("AGENOMIC_API_KEY", "env-key")
    monkeypatch.setenv("AGENOMIC_WORKSPACE_ID", WORKSPACE)
    monkeypatch.setenv("AGENOMIC_PROMPT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENOMIC_TIMEOUT", "5")
    configured = Client.from_env(transport=server.transport())
    assert (configured.base_url, configured.api_key, configured.workspace_id) == (
        BASE,
        "env-key",
        WORKSPACE,
    )
    assert configured._timeout == 5.0
    configured.prompts.get("prm_writer:1")
    assert server.requests[-1].headers["authorization"] == "Bearer env-key"
    assert (tmp_path / "v1" / WORKSPACE / "prompts" / "prm_writer" / "1").is_dir()
    assert Client.from_env(api_key="explicit", transport=server.transport()).api_key == "explicit"
    monkeypatch.setenv("AGENOMIC_TIMEOUT", "soon")
    with pytest.raises(ValueError):
        Client.from_env()
    for variable in (
        "AGENOMIC_ENDPOINT",
        "AGENOMIC_API_KEY",
        "AGENOMIC_WORKSPACE_ID",
        "AGENOMIC_PROMPT_CACHE_DIR",
        "AGENOMIC_TIMEOUT",
    ):
        monkeypatch.delenv(variable)
    assert not Client.from_env().is_cloud
    with make_client(server) as managed:
        managed.prompts.get("prm_writer:1")
        http = pool_for(managed).sync()
    assert http.is_closed


async def test_async_twins(server: FakePromptServer, tmp_path: Path) -> None:
    engine = server.engine
    root, _ = release_with_child(engine)
    engine.move_channel(AGENT, "production", root, expected_generation=0)
    engine.move_alias("prm_planner", "prod", version=1, expected_generation=0)
    async with make_client(server, workspace_id=None) as client:
        assert (await client.awhoami())["org_id"] == WORKSPACE
        version = await client.prompts.aget("prm_planner:1")
        assert (await client.prompts.aresolve("prm_planner@prod")).ref == version.ref
        assert len(await client.prompts.apin(["prm_planner@prod"])) == 1
        assert (
            await client.prompts.arender("prm_writer:1", {"topic": "t"})
        ).text == "Write about t."
        assert (await client.prompts.alist()).items
        assert (await client.prompts.versions.alist("prm_planner")).items[0].version == 1
        assert (await client.prompts.versions.aget("prm_writer", 1)).ref.version == 1
        created = await client.prompts.acreate("prm_async", name="Async", kind="text")
        assert created.prompt_id == "prm_async"
        published = await client.prompts.apublish(
            "prm_async", text_content("Hello."), parent_version=None, change_message="first"
        )
        assert published.ref.version == 1
        draft = await client.prompts.drafts.asave(
            "prm_async", text_content("Hi."), base_version=1, expected_revision=0
        )
        assert (await client.prompts.drafts.aget("prm_async")).revision == draft.revision
        assert (await client.prompts.aliases.aget("prm_planner", "prod")).generation == 1
        with pytest.raises(ApiError):
            await client.prompts.aliases.amove(
                "prm_planner", "prod", version=1, expected_generation=1
            )
        key = thread_key(WORKSPACE, "async")
        binding, _, created_binding = await client.bindings.acreate(
            AGENT, thread_key=key, scope="thread", channel="production"
        )
        assert created_binding
        assert (await client.bindings.aget(AGENT, binding.binding_id))[0] == binding
        child = await client.bindings.acounterfactual(
            AGENT, binding.binding_id, thread_key="cf:async", release_id=root
        )
        assert child.parent_binding_id == binding.binding_id
        assert (await client.prompts.aresolve_agent(AGENT, channel="production")).release_id == root
        exported = await client.prompts.aexport_bundle(
            AGENT, channel="production", path=tmp_path / "async.json"
        )
        assert exported.signed
        assert (tmp_path / "async.json").is_file()
        assert [channel.name for channel in await client.channels.alist(AGENT)] == ["production"]
        assert (await client.channels.aget(AGENT, "production")).generation == 1
        assert len(await client.channels.ahistory(AGENT, "production")) == 1
        preview = await client.channels.amove_preview(AGENT, "production", release_id=root)
        assert preview.agent_id == AGENT


def report_errors(report: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    documents = [json.loads(path.read_text()) for path in sorted(SCHEMAS.glob("*.json"))]
    registry = Registry().with_resources(
        (document["$id"], Resource.from_contents(document)) for document in documents
    )
    schema = next(d for d in documents if d["$id"].endswith("prompt-discovery-report.schema.json"))
    validator = Draft202012Validator(schema, registry=registry)
    return [f"{error.json_path}: {error.message}" for error in validator.iter_errors(report)]


def test_import_report_then_apply_cites_the_plan_digest(
    client: Client, server: ImportServer
) -> None:
    report = import_fixture("discovery-report.json")
    plan = client.prompts.import_report(report, agent_id=AGENT)
    assert server.paths() == ["POST /v1/prompts/imports"]
    assert body_of(server) == {"report": report, "agent_id": AGENT}
    assert "idempotency-key" not in server.requests[-1].headers
    assert isinstance(plan, ImportPlan)
    assert (plan.import_id, plan.status, plan.replayed) == (plan.plan["plan_id"], "planned", False)
    assert plan.report_digest == prompt_digest(report)
    assert plan.plan_digest == plan.plan["plan_digest"]
    assert {item["action"] for item in plan.items} == {"create_prompt", "skip", "blocked"}
    replay = client.prompts.import_report(report, agent_id=AGENT)
    assert replay.replayed
    assert replay.plan == plan.plan
    import_id = cast_str(plan.import_id)
    with pytest.raises(ValueError):
        client.prompts.apply_import(
            import_id, plan_digest=plan.plan_digest, items=plan.decisions(), declare_slots=True
        )
    assert len(server.requests) == 2
    with pytest.raises(PromptImportError) as stale:
        client.prompts.apply_import(
            import_id, plan_digest="sha256:" + "0" * 64, items=plan.decisions()
        )
    assert stale.value.code == "prompt_import_plan_stale"
    assert stale.value.details["current"] == plan.plan_digest
    result = client.prompts.apply_import(
        import_id,
        plan_digest=plan.plan_digest,
        items=plan.decisions(),
        declare_slots=True,
        expected_slots_revision=0,
        agent_id=AGENT,
    )
    sent = body_of(server)
    assert set(sent) == {
        "idempotency_key",
        "plan_digest",
        "mode",
        "declare_slots",
        "items",
        "agent_id",
    }
    assert sent["plan_digest"] == plan.plan_digest
    assert sent["idempotency_key"].startswith("import-apply-")
    assert sent["idempotency_key"] != body_of(server, -2)["idempotency_key"]
    assert sent["items"] == plan.decisions()
    assert server.requests[-1].headers["if-match"] == '"0"'
    assert "idempotency-key" not in server.requests[-1].headers
    assert result["replayed"] is False
    assert result["slots"] == {"agent_id": AGENT, "revision": 1}
    outcomes = {entry["item_id"]: entry["outcome"] for entry in result["results"]}
    assert sorted(set(outcomes.values())) == ["created", "skipped"]
    assert server.engine.prompt("prm_support_planner")["latest_version"] == 1
    again = client.prompts.apply_import(
        import_id,
        plan_digest=plan.plan_digest,
        items=plan.decisions(),
        declare_slots=True,
        expected_slots_revision=0,
        agent_id=AGENT,
        idempotency_key=sent["idempotency_key"],
    )
    assert again["replayed"] is True
    assert again["results"] == result["results"]
    with pytest.raises(PromptImportError) as applied:
        client.prompts.apply_import(import_id, plan_digest=plan.plan_digest, items=plan.decisions())
    assert applied.value.code == "prompt_import_already_applied"
    assert "if-match" not in server.requests[-1].headers
    assert client.prompts.import_report(report).items[0]["action"] == "reuse_version"


def cast_str(value: Optional[str]) -> str:
    assert value is not None
    return value


def test_import_answers_are_verified(client: Client, server: ImportServer) -> None:
    report = import_fixture("discovery-report.json")

    def tamper(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "import" in payload:
            payload["import"]["plan"]["items"][0]["prompt_id"] = "prm_other"
        return payload

    server.rewrite = tamper
    with pytest.raises(PromptIntegrityError) as tampered:
        client.prompts.import_report(report, agent_id=AGENT)
    assert tampered.value.code == "prompt_digest_mismatch"

    def other_import(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "import" in payload:
            payload["import"]["import_id"] = "imp_01j9x1k3m5n7p9q1r3s5t7v9zz"
        if "results" in payload:
            payload["import_id"] = "imp_01j9x1k3m5n7p9q1r3s5t7v9zz"
        return payload

    def no_record(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        payload["import"] = "planned"
        return payload

    def other_report(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "import" in payload:
            plan = payload["import"]["plan"]
            plan["source"]["digest"] = "sha256:" + "0" * 64
            del plan["plan_digest"]
            plan["plan_digest"] = prompt_digest(plan)
        return payload

    server.rewrite = other_report
    with pytest.raises(ApiError) as other_source:
        client.prompts.import_report(report, agent_id=AGENT)
    assert other_source.value.code == "invalid_response"
    server.rewrite = no_record
    with pytest.raises(ApiError) as shapeless:
        client.prompts.import_report(report, agent_id=AGENT)
    assert shapeless.value.code == "invalid_response"
    server.rewrite = other_import
    with pytest.raises(ApiError) as mismatch:
        client.prompts.import_report(report, agent_id=AGENT)
    assert mismatch.value.code == "invalid_response"
    with pytest.raises(ApiError) as other_agent:
        client.prompts.import_report(report, agent_id=OTHER_AGENT)
    assert other_agent.value.code == "invalid_response"
    server.rewrite = None
    plan = client.prompts.import_report(report)
    server.rewrite = other_import
    with pytest.raises(ApiError) as result:
        client.prompts.apply_import(
            cast_str(plan.import_id), plan_digest=plan.plan_digest, items=plan.decisions()
        )
    assert result.value.code == "invalid_response"
    broken = json.loads(json.dumps(report))
    broken["root"]["label"] = "/home/dev/support"
    count = len(server.requests)
    with pytest.raises(PromptImportError):
        client.prompts.import_report(broken)
    assert len(server.requests) == count


def test_declarations_convert_yaml_client_side(
    client: Client, server: ImportServer, tmp_path: Path
) -> None:
    plan = client.prompts.plan_declarations(PROMPTS_FILE)
    request = server.requests[-1]
    sent = body_of(server)
    assert server.paths() == ["POST /v1/prompts/declarations/plan"]
    assert set(sent) == {"document"}
    assert sent["document"] == load_prompts_file(PROMPTS_FILE)
    assert sent["document"]["prompts"][0]["tags"] == ["yes", "support"]
    assert sent["document"]["prompts"][0]["content"]["body"] == "Answer {question} politely."
    assert b"|-" not in request.content
    assert request.headers["content-type"] == "application/json"
    assert "idempotency-key" not in request.headers
    assert plan.plan["source"] == {
        "kind": "prompts_file",
        "digest": prompt_digest(sent["document"]),
    }
    assert (plan.import_id, plan.plan["plan_id"]) == (None, None)
    assert plan.slots == {
        "agent_id": AGENT,
        "revision": 0,
        "added": ["writer.response"],
        "removed": [],
        "changed": [],
    }
    path = tmp_path / "prompts.yaml"
    path.write_text(PROMPTS_FILE, encoding="utf-8")
    assert client.prompts.plan_declarations(path).plan_digest == plan.plan_digest
    assert client.prompts.plan_declarations(sent["document"]).plan_digest == plan.plan_digest
    count = len(server.requests)
    with pytest.raises(ValueError):
        client.prompts.apply_declarations(PROMPTS_FILE, plan_digest=plan.plan_digest)
    for broken in (
        "prompts: [",
        PROMPTS_FILE.replace("kind: text\n    name", "kind: 1.5\n    name"),
    ):
        with pytest.raises(PromptImportError):
            client.prompts.plan_declarations(broken)
    assert len(server.requests) == count
    with pytest.raises(PromptImportError) as stale:
        client.prompts.apply_declarations(
            PROMPTS_FILE, plan_digest="sha256:" + "0" * 64, expected_slots_revision=0
        )
    assert stale.value.code == "prompt_import_plan_stale"
    assert stale.value.details["plan_digest"] == plan.plan_digest
    result = client.prompts.apply_declarations(
        path, plan_digest=plan.plan_digest, expected_slots_revision=plan.slots["revision"]
    )
    applied = body_of(server)
    assert set(applied) == {"idempotency_key", "document", "plan_digest"}
    assert applied["document"] == sent["document"]
    assert applied["plan_digest"] == plan.plan_digest
    assert applied["idempotency_key"].startswith("declarations-apply-")
    assert server.requests[-1].headers["if-match"] == '"0"'
    assert "idempotency-key" not in server.requests[-1].headers
    assert result["results"][0]["outcome"] == "created"
    assert result["slots"] == {"agent_id": AGENT, "revision": 1}
    again = client.prompts.apply_declarations(
        path,
        plan_digest=plan.plan_digest,
        expected_slots_revision=0,
        idempotency_key=applied["idempotency_key"],
    )
    assert again["replayed"] is True
    unslotted = load_prompts_file(PROMPTS_FILE)
    del unslotted["slots"]
    unslotted["prompts"][0]["expected_latest_version"] = 1
    reused = client.prompts.plan_declarations(unslotted)
    assert reused.slots is None
    assert reused.items[0]["action"] == "reuse_version"
    done = client.prompts.apply_declarations(unslotted, plan_digest=reused.plan_digest)
    assert "if-match" not in server.requests[-1].headers
    assert done["results"][0]["outcome"] == "unchanged"

    def other_document(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if "plan" in payload:
            plan = payload["plan"]
            plan["source"]["digest"] = "sha256:" + "1" * 64
            plan["plan_digest"] = prompt_digest(
                {key: value for key, value in plan.items() if key != "plan_digest"}
            )
        if "results" in payload:
            payload["plan_digest"] = "sha256:" + "2" * 64
        return payload

    server.rewrite = other_document
    with pytest.raises(ApiError) as mismatch:
        client.prompts.plan_declarations(unslotted)
    assert mismatch.value.code == "invalid_response"
    with pytest.raises(ApiError) as result_mismatch:
        client.prompts.apply_declarations(unslotted, plan_digest=reused.plan_digest)
    assert result_mismatch.value.code == "invalid_response"


def test_register_runtime_builds_a_report_and_only_plans(
    client: Client, server: ImportServer
) -> None:
    prompts = pytest.importorskip("langchain_core.prompts")
    secret = "AKIA" + "Z7" * 8
    slots = {
        "writer.instructions": prompts.PromptTemplate.from_template("Write about {topic}."),
        "planner.system": prompts.ChatPromptTemplate.from_messages(
            [
                ("system", "Plan for {customer}."),
                prompts.MessagesPlaceholder("history", optional=True),
            ]
        ),
        "triage.router": prompts.PromptTemplate.from_template(
            "Route {{x}}", template_format="mustache"
        ),
        "billing.system": prompts.PromptTemplate.from_template(f"Sign with {secret}."),
        "dated.system": prompts.PromptTemplate.from_template(
            "Today is {day}.", partial_variables={"day": lambda: "monday"}
        ),
        "price.system": prompts.PromptTemplate.from_template("Price {amount:.2f}"),
    }
    plan = client.prompts.register_runtime(AGENT, slots)
    assert server.paths() == ["POST /v1/prompts/imports"]
    assert secret not in server.requests[-1].content.decode("utf-8")
    sent = body_of(server)
    assert set(sent) == {"report", "agent_id"}
    assert sent["agent_id"] == AGENT
    report = sent["report"]
    assert report_errors(report) == []
    assert report["root"] == {"label": "runtime_registration", "vcs": None}
    assert report["files"] == []
    candidates = {entry["proposal"]["slot_path"]: entry for entry in report["candidates"]}
    assert list(candidates) == sorted(slots)
    planner = candidates["planner.system"]
    assert (planner["status"], planner["construct"]) == (
        "supported",
        "langchain.chat_prompt_template",
    )
    assert planner["proposal"] == {
        "prompt_id": "prm_planner_system",
        "prompt_kind": "chat",
        "slot_path": "planner.system",
        "node_path": None,
        "usage": "system",
    }
    assert planner["source"]["path"] == "planner.system"
    assert planner["content"]["body"][1] == {"placeholder": "history", "optional": True}
    assert candidates["writer.instructions"]["proposal"]["usage"] == "instructions"
    triage = candidates["triage.router"]
    assert (triage["status"], triage["content"], triage["proposal"]["usage"]) == (
        "unsupported",
        None,
        "other",
    )
    assert [issue["code"] for issue in triage["issues"]] == ["unsupported_template_format"]
    billing = candidates["billing.system"]
    assert (billing["status"], billing["content"]) == ("blocked_secret", None)
    assert billing["secret_findings"] == [
        {"pattern": "aws_access_key", "line": 1, "column": 1, "length": 20}
    ]
    assert "callable_partial" in [issue["code"] for issue in candidates["dated.system"]["issues"]]
    price = candidates["price.system"]
    assert price["status"] == "unsupported"
    assert price["issues"][0]["code"] == "format_spec"
    assert price["issues"][0]["severity"] == "error"
    assert price["issues"][0]["message"] == "template syntax error: format_spec"
    assert plan.plan["source"]["kind"] == "discovery_report"
    actions = {item["slot"]["slot_path"]: item["action"] for item in plan.items}
    assert actions == {
        "billing.system": "blocked",
        "dated.system": "blocked",
        "planner.system": "create_prompt",
        "price.system": "blocked",
        "triage.router": "blocked",
        "writer.instructions": "create_prompt",
    }
    assert "prm_planner_system" not in server.engine._state["prompts"]
    client.prompts.register_runtime(AGENT, slots)
    assert [entry["candidate_id"] for entry in body_of(server)["report"]["candidates"]] == [
        entry["candidate_id"] for entry in report["candidates"]
    ]
    count = len(server.requests)
    with pytest.raises(ValueError):
        client.prompts.register_runtime(AGENT, {})
    with pytest.raises(ValueError):
        client.prompts.register_runtime(AGENT, {"Planner": slots["planner.system"]})
    with pytest.raises(TypeError):
        client.prompts.register_runtime(AGENT, {"planner.system": "Plan for {customer}."})
    assert len(server.requests) == count
    server.api_key_scopes = ["read"]
    with pytest.raises(ApiError) as refused:
        client.prompts.register_runtime(
            AGENT, {"writer.instructions": slots["writer.instructions"]}
        )
    assert (refused.value.code, refused.value.status) == ("api_key_scope_insufficient", 403)
    assert refused.value.message == SCOPE_MESSAGE
    assert len(server.requests) == count + 1


def test_runtime_prompt_ids_follow_the_grammar(client: Client, server: ImportServer) -> None:
    prompts = pytest.importorskip("langchain_core.prompts")
    template = prompts.PromptTemplate.from_template("Write about {topic}.")
    client.prompts.register_runtime(
        AGENT, {"a_b.c": template, "a.b_c": template, "a__.b_": template}
    )
    report = body_of(server)["report"]
    assert report_errors(report) == []
    assert [entry["proposal"]["prompt_id"] for entry in report["candidates"]] == [
        "prm_a_b_c",
        "prm_a_b",
        "prm_a_b_c_2",
    ]


def test_report_usage_sends_refs_and_hashes_only(
    client: Client, server: ImportServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.api_key_scopes = ["read"]
    path = f"POST /v1/agents/{AGENT}/bindings/bnd_x/usage"
    managed = {
        "slot_path": "planner.instructions",
        "node_path": "planner",
        "prompt_ref": "prm_planner:1",
        "content_digest": "sha256:" + "a" * 64,
        "rendered_hash": "sha256:" + "b" * 64,
        "overlay": {"digest": "blake3:" + "c" * 64, "position": "prepended"},
        "alias": "prod",
        "alias_generation": 1,
        "count": 3,
        "first_at": AT,
        "last_at": AT,
    }
    unmanaged = {
        "slot_path": None,
        "node_path": "researcher",
        "unmanaged": True,
        "rendered_hash": "sha256:" + "d" * 64,
        "role_layout": ["system", "user"],
        "count": 1,
        "first_at": AT,
        "last_at": AT,
    }
    assert client.bindings.report_usage(AGENT, "bnd_x", [managed, unmanaged]) is None
    assert server.paths() == [path]
    assert body_of(server) == {"observations": [managed, unmanaged]}
    assert "idempotency-key" not in server.requests[-1].headers
    refused: list[Any] = [
        {**unmanaged, "rendered_text": "hello there"},
        {**unmanaged, "rendered_hash": "blake3:" + "e" * 64},
        {**managed, "overlay": {**managed["overlay"], "text": "hello there"}},
        {**managed, "overlay": "prepended"},
    ]
    for observation in refused:
        with pytest.raises(ValueError):
            client.bindings.report_usage(AGENT, "bnd_x", [unmanaged, observation])
    with pytest.raises(TypeError):
        client.bindings.report_usage(AGENT, "bnd_x", cast(Any, ["hello there"]))
    client.bindings.report_usage(AGENT, "bnd_x", [])
    assert server.paths() == [path]
    assert all(b"hello there" not in request.content for request in server.requests)
    batch = [{**unmanaged, "count": number + 1} for number in range(501)]
    monkeypatch.setattr(_transport, "_sleep", lambda seconds: None)
    server.fail_next = [(503, {})]
    client.bindings.report_usage(AGENT, "bnd_x", batch)
    sizes = [len(json.loads(request.content)["observations"]) for request in server.requests[1:]]
    assert sizes == [500, 500, 1]
    assert len(server.usage) == 2 + 501


async def test_import_and_usage_async_twins(server: ImportServer, tmp_path: Path) -> None:
    async with make_client(server) as client:
        report = import_fixture("discovery-report.json")
        plan = await client.prompts.aimport_report(report, agent_id=AGENT)
        applied = await client.prompts.aapply_import(
            cast_str(plan.import_id), plan_digest=plan.plan_digest, items=plan.decisions()
        )
        assert body_of(server)["idempotency_key"].startswith("import-apply-")
        assert {entry["outcome"] for entry in applied["results"]} == {"created", "skipped"}
        notes = PROMPTS_FILE.replace("prm_support_writer", "prm_support_notes")
        declared = await client.prompts.aplan_declarations(notes)
        assert declared.slots is not None
        result = await client.prompts.aapply_declarations(
            notes,
            plan_digest=declared.plan_digest,
            expected_slots_revision=declared.slots["revision"],
        )
        assert result["slots"]["revision"] == 1
        observation = {
            "unmanaged": True,
            "rendered_hash": "sha256:" + "d" * 64,
            "count": 1,
            "first_at": AT,
            "last_at": AT,
        }
        assert await client.bindings.areport_usage(AGENT, "bnd_x", [observation]) is None
        assert server.usage == [observation]
        prompts = pytest.importorskip("langchain_core.prompts")
        registered = await client.prompts.aregister_runtime(
            AGENT, {"notes.system": prompts.PromptTemplate.from_template("Take notes.")}
        )
        assert registered.items[0]["action"] == "create_prompt"


def test_bind_langgraph_never_registers_prompts() -> None:
    pytest.importorskip("langgraph")
    from langgraph_world import World, thread, two_node_graph

    world = World.create()
    world.server.api_key_scopes = ["read"]
    managed = world.bind(two_node_graph())
    assert managed.invoke({"log": []}, thread("registration"))["log"]
    paths = world.server.paths()
    assert any(path.endswith("/bindings") for path in paths)
    assert not [
        path
        for path in paths
        if "/v1/prompts/imports" in path
        or "/v1/prompts/declarations" in path
        or path.endswith("/usage")
    ]
