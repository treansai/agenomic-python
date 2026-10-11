from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
from knowledge_fakes import (
    AGENT,
    BASE,
    BINDING,
    DOC,
    EVENT,
    JOB,
    KB,
    PROMPT_BINDING,
    SECTION,
    FakeKnowledgeApi,
    body_of,
    error_response,
    fixture,
)

from agenomic import Client
from agenomic._transport import api_request
from agenomic.exceptions import ApiError
from agenomic.knowledge import (
    UNSET,
    AgentSearchResponse,
    KnowledgeBase,
    SearchResponse,
    normalize_version,
    version_number,
)
from agenomic.knowledge import resources as knowledge_resources


def test_normalize_version_accepts_numbers_tags_and_names() -> None:
    assert normalize_version(3) == 3
    assert normalize_version("3") == 3
    assert normalize_version("v3") == 3
    assert normalize_version("published") == "published"
    assert normalize_version("draft") == "draft"
    assert normalize_version("2147483647") == 2147483647
    assert version_number("v12") == 12
    for bad in (0, -1, True, "v0", "03", "V3", "latest", "", "3.0", 2147483648, "2147483648"):
        with pytest.raises(ValueError):
            normalize_version(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        version_number("published")


def test_knowledge_bases_list_get_create_update_archive_restore_delete() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        page = cloud.knowledge.list(q="support", tag="support", sort="name", limit=10)
        assert [item.kb_id for item in page.items] == [KB, "kb_compliance"]
        assert page.next_cursor is not None
        kb = cloud.knowledge.get(KB)
        assert isinstance(kb, KnowledgeBase)
        assert kb.record is not None
        assert kb.record.published_version == 3
        assert kb.published_version == 3
        assert kb.detail is not None
        assert kb.detail.stats is not None
        assert kb.detail.stats.retrievals_24h == 342
        assert kb.detail.published is not None
        assert kb.detail.published.status == "approved"
        created = cloud.knowledge.create(
            KB,
            "Customer Support",
            description="Refund policies",
            tags=["support"],
            labels={"domain": "support"},
            settings={"retrieval": {"top_k": 8}},
        )
        assert created.kb_id == KB
        updated = cloud.knowledge.update(KB, if_match=5, name="Support EU", description=None)
        assert updated.metadata_revision == 5
        assert cloud.knowledge.archive(KB, reason="replaced").status == "archived"
        assert cloud.knowledge.restore(KB).status == "active"
        assert cloud.knowledge.delete(KB) is None
    listing = api.last("GET", r"^/v1/knowledge-bases$")
    assert dict(listing.url.params) == {
        "q": "support",
        "tag": "support",
        "sort": "name",
        "limit": "10",
    }
    create = api.last("POST", r"^/v1/knowledge-bases$")
    assert body_of(create) == {
        "kb_id": KB,
        "name": "Customer Support",
        "description": "Refund policies",
        "tags": ["support"],
        "labels": {"domain": "support"},
        "settings": {"retrieval": {"top_k": 8}},
    }
    assert "idempotency-key" not in create.headers
    assert create.headers["authorization"] == "Bearer agm_test"
    patch = api.last("PATCH")
    assert patch.headers["if-match"] == '"5"'
    assert body_of(patch) == {"name": "Support EU", "description": None}
    assert body_of(api.last("POST", "/archive$")) == {"reason": "replaced"}
    assert body_of(api.last("POST", "/restore$")) == {}
    assert api.last("DELETE").url.path == f"/v1/knowledge-bases/{KB}"


def test_update_tri_state_omits_unset_members() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        cloud.knowledge.update(KB, if_match=1, owner="team-support", description=UNSET)
    assert body_of(api.last("PATCH")) == {"owner": "team-support"}


def test_collections_crud() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        page = cloud.knowledge.collections.list(KB)
        assert [item.collection_id for item in page.items] == [
            "escalations",
            "faq",
            "product_documentation",
        ]
        created = cloud.knowledge.collections.create(
            KB, "faq", "FAQ", classification="public", restricted_roles=["owner"]
        )
        assert created.collection_id == "faq"
        cloud.knowledge.collections.update(KB, "faq", if_match=2, classification=None)
        cloud.knowledge.collections.delete(KB, "faq")
    assert body_of(api.last("POST", "/collections$")) == {
        "collection_id": "faq",
        "name": "FAQ",
        "classification": "public",
        "restricted_roles": ["owner"],
    }
    patch = api.last("PATCH", "/collections/faq$")
    assert patch.headers["if-match"] == '"2"'
    assert body_of(patch) == {"classification": None}
    assert api.last("DELETE").url.path.endswith("/collections/faq")


def test_documents_list_with_tree_and_filters() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        listed = cloud.knowledge.documents.list(
            KB, path_prefix="faq/", collection="faq", tree=True, limit=2
        )
    assert [item.document_id for item in listed.documents] == [
        DOC,
        "kdoc_01jb3m5q7s9v1x3z5b7d9f0003",
    ]
    assert listed.tree is not None
    assert any(node.children for node in listed.tree)
    assert listed.next_cursor is not None
    params = dict(api.last("GET", "/documents$").url.params)
    assert params == {"path_prefix": "faq/", "collection": "faq", "view": "tree", "limit": "2"}


def test_document_create_inline_and_write_parsing() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        written = cloud.knowledge.documents.create(
            KB,
            "faq/gift-cards.md",
            "# Gift Cards\n\nGift cards never expire.\n",
            media_type="text/markdown",
            collection="faq",
            tags=["gift-cards"],
            metadata={"owner_team": "support-ops"},
            classification="public",
            change_message="Add the gift card FAQ",
        )
    assert written.created is True
    assert written.document.path == "faq/gift-cards.md"
    assert written.revision is not None
    assert written.revision.revision == 1
    assert written.job is not None
    assert written.job.kind == "ingest_document"
    assert body_of(api.last("POST", "/documents$")) == {
        "path": "faq/gift-cards.md",
        "content": "# Gift Cards\n\nGift cards never expire.\n",
        "media_type": "text/markdown",
        "collection": "faq",
        "tags": ["gift-cards"],
        "metadata": {"owner_team": "support-ops"},
        "classification": "public",
        "change_message": "Add the gift card FAQ",
    }


def test_upload_bytes_sends_raw_body_and_headers() -> None:
    api = FakeKnowledgeApi()
    data = b"%PDF-1.7 binary \x00\xff"
    with api.client() as cloud:
        written = cloud.knowledge.documents.upload(
            KB,
            data,
            path="policies/refund policy é.pdf",
            content_type="application/pdf",
            collection="faq",
            tags=["refunds", "policy"],
            classification="internal",
            change_message="Initial import",
        )
    assert written.document.document_id
    request = api.last("POST", "/documents/upload$")
    assert request.content == data
    assert request.headers["content-type"] == "application/pdf"
    raw_path = request.headers["x-agenomic-document-path"]
    assert raw_path == "policies/refund%20policy%20%C3%A9.pdf"
    assert unquote(raw_path) == "policies/refund policy é.pdf"
    assert request.headers["x-agenomic-collection"] == "faq"
    assert request.headers["x-agenomic-tags"] == "refunds,policy"
    assert request.headers["x-agenomic-classification"] == "internal"
    assert request.headers["x-agenomic-change-message"] == "Initial%20import"
    assert "idempotency-key" not in request.headers


def test_upload_headers_are_percent_encoded_utf8() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        cloud.knowledge.documents.upload(
            KB,
            b"# Remboursements\n",
            path="politiques/remboursements été.md",
            collection="politiques générales",
            tags=["remboursé", "a/b", "fin d'année"],
            classification="interne",
            change_message="Révision des remboursements",
        )
    request = api.last("POST", "/documents/upload$")
    headers = request.headers
    assert headers["x-agenomic-document-path"] == "politiques/remboursements%20%C3%A9t%C3%A9.md"
    assert headers["x-agenomic-collection"] == "politiques%20g%C3%A9n%C3%A9rales"
    assert headers["x-agenomic-tags"] == "rembours%C3%A9,a%2Fb,fin%20d%27ann%C3%A9e"
    assert headers["x-agenomic-classification"] == "interne"
    change = headers["x-agenomic-change-message"]
    assert change == "R%C3%A9vision%20des%20remboursements"
    assert unquote(change) == "Révision des remboursements"
    assert [unquote(tag) for tag in headers["x-agenomic-tags"].split(",")] == [
        "remboursé",
        "a/b",
        "fin d'année",
    ]
    assert all(value.isascii() for value in headers.values())


def test_upload_file_path_defaults_path_and_content_type(tmp_path: Path) -> None:
    file = tmp_path / "refunds.md"
    file.write_bytes(b"# Refunds\n")
    api = FakeKnowledgeApi()

    async def run() -> None:
        async with api.client() as cloud:
            await cloud.knowledge.documents.aupload(KB, file)
            await cloud.knowledge.documents.aupload(KB, str(file), path="faq/refunds.md")

    asyncio.run(run())
    first, second = [r for r in api.requests if r.url.path.endswith("/documents/upload")]
    assert first.content == b"# Refunds\n"
    assert first.headers["content-type"] == "application/octet-stream"
    assert first.headers["x-agenomic-document-path"] == "refunds.md"
    assert second.headers["x-agenomic-document-path"] == "faq/refunds.md"


def test_upload_revision_and_put_content_send_if_match() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        cloud.knowledge.documents.upload_revision(
            KB, DOC, b"# Refunds v7\n", if_match=6, content_type="text/markdown"
        )
        cloud.knowledge.documents.put_content(
            KB, DOC, "# Refunds\n", if_match=6, change_message="Fix typo"
        )
    upload = api.last("POST", f"{DOC}/upload$")
    assert upload.headers["if-match"] == '"6"'
    assert upload.content == b"# Refunds v7\n"
    assert upload.headers["content-type"] == "text/markdown"
    assert "x-agenomic-document-path" not in upload.headers
    put = api.last("PUT", "/content$")
    assert put.headers["if-match"] == '"6"'
    assert body_of(put) == {"content": "# Refunds\n", "change_message": "Fix typo"}


def test_document_reads_and_writes() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        detail = cloud.knowledge.documents.get(KB, DOC)
        assert detail.document.document_id == DOC
        assert detail.revision is not None
        assert detail.revision.revision == 6
        updated = cloud.knowledge.documents.update(
            KB, DOC, if_match=3, path="faq/refunds.md", title=None, tags=["refunds"]
        )
        assert updated.document_id == DOC
        duplicated = cloud.knowledge.documents.duplicate(KB, DOC, path="faq/refunds-eu.md")
        assert duplicated.created is True
        assert cloud.knowledge.documents.delete(KB, DOC) is None
        assert cloud.knowledge.documents.restore(KB, DOC).document_id == DOC
        revisions = cloud.knowledge.documents.revisions(KB, DOC, limit=5)
        assert revisions.document_id == DOC
        assert revisions.revisions
        text = cloud.knowledge.documents.text(KB, DOC, revision=6)
        assert text.text == "# Refunds"
        tree = cloud.knowledge.documents.sections(KB, DOC, version="v4", include_content=True)
        assert tree.version == 4
        assert tree.sections[0].children
        section = cloud.knowledge.documents.section(
            KB, DOC, SECTION, version=3, include=["children"]
        )
        assert section.section.heading == "Refund Policy"
        assert section.children
        backlinks = cloud.knowledge.documents.backlinks(KB, DOC, version="published")
        assert backlinks.backlinks[0].target_section_id
    patch = api.last("PATCH")
    assert patch.headers["if-match"] == '"3"'
    assert body_of(patch) == {"path": "faq/refunds.md", "title": None, "tags": ["refunds"]}
    assert body_of(api.last("POST", "/duplicate$")) == {"path": "faq/refunds-eu.md"}
    assert dict(api.last("GET", "/content$").url.params) == {"format": "text", "revision": "6"}
    assert dict(api.last("GET", "/sections$").url.params) == {"version": "4", "include": "content"}
    assert dict(api.last("GET", f"/sections/{SECTION}$").url.params) == {
        "version": "3",
        "include": "children",
    }
    assert dict(api.last("GET", "/backlinks$").url.params) == {"version": "published"}


def test_search_query_answer_bodies_and_parsing() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        found = cloud.knowledge.search(
            KB,
            "How long do customers have to request a refund?",
            version="v3",
            mode="hybrid",
            top_k=5,
            filters={"collections": ["faq"], "metadata": {"owner_team": "support-ops"}},
            rerank="lexical",
            max_context_tokens=2000,
            include_context=True,
            expand="parent",
        )
        assert isinstance(found, SearchResponse)
        assert found.results[0].citation.section_id == SECTION
        assert found.results[1].risk.level == "medium"
        assert found.context is not None
        assert "<knowledge_evidence" in found.context
        assert found.retrieval.version_manifest_digest is not None
        queried = cloud.knowledge.query(KB, 'get "Refund Policy" from "Refunds and Returns"')
        assert queried.matches[0].match_kind == "exact"
        operation = cloud.knowledge.query(
            KB,
            operation={"op": "list_children", "section": "Refunds and Returns"},
            version=3,
        )
        assert operation.sections
        answered = cloud.knowledge.answer(KB, "refund window?", version=3, top_k=5)
        assert answered.abstained is False
        assert answered.citations
    search, text_query, op_query = (
        body_of(r) for r in api.requests if r.url.path.endswith(("/search", "/query"))
    )
    assert search == {
        "query": "How long do customers have to request a refund?",
        "version": 3,
        "mode": "hybrid",
        "top_k": 5,
        "filters": {"collections": ["faq"], "metadata": {"owner_team": "support-ops"}},
        "rerank": "lexical",
        "max_context_tokens": 2000,
        "expand": "parent",
        "include_context": True,
    }
    assert text_query == {"query": 'get "Refund Policy" from "Refunds and Returns"'}
    assert op_query == {
        "operation": {"op": "list_children", "section": "Refunds and Returns"},
        "version": 3,
    }
    assert body_of(api.last("POST", "/answer$")) == {
        "query": "refund window?",
        "version": 3,
        "top_k": 5,
    }


def test_abstained_answer_parses() -> None:
    api = FakeKnowledgeApi()
    api.override("POST", f"/v1/knowledge-bases/{KB}/answer", lambda _: fixture("answer_abstained"))
    with api.client() as cloud:
        answered = cloud.knowledge.answer(KB, "warranty on drones?")
    assert answered.abstained is True
    assert answered.answer is None
    assert answered.reason


def test_handle_search_get_document_and_section_by_id() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        kb = cloud.knowledge.get(KB)
        found = kb.search("refund window", version="published", mode="keyword")
        assert found.retrieval.kb_id == KB
        assert kb.get_document(DOC).document.document_id == DOC
        section = kb.get_section(document_id=DOC, section=SECTION, version=3)
        assert section.section_id == SECTION
        assert section.content is not None
        versions = kb.versions()
        assert [v.version for v in versions.items] == [4, 3]
        published = kb.publish("v4", if_match=4, reason="Refund window extended to 30 days")
        assert published.event is not None
        assert published.event.to_version == 4
    assert body_of(api.last("POST", "/search$")) == {
        "query": "refund window",
        "version": "published",
        "mode": "keyword",
        "top_k": 5,
    }
    publish = api.last("POST", "/publish$")
    assert publish.headers["if-match"] == '"4"'
    assert body_of(publish) == {"version": 4, "reason": "Refund window extended to 30 days"}


def test_handle_get_section_by_heading_resolves_through_query() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        kb = cloud.knowledge.base(KB)
        assert kb.detail is None
        section = kb.get_section(document_id=DOC, section="Refund Policy", version="v3")
        assert section.section_id == SECTION
        by_name = kb.get_section(document="Refunds and Returns", section="Refund Policy")
        assert by_name.heading == "Refund Policy"
    assert api.paths() == [
        f"POST /v1/knowledge-bases/{KB}/query",
        f"POST /v1/knowledge-bases/{KB}/query",
    ]
    first, second = (body_of(r) for r in api.requests)
    assert first == {
        "operation": {"op": "get_section", "document": DOC, "section": "Refund Policy"},
        "version": 3,
    }
    assert second["operation"]["document"] == "Refunds and Returns"
    assert "version" not in second


def test_handle_get_section_refuses_a_match_of_another_document() -> None:
    api = FakeKnowledgeApi()
    other = "kdoc_01jb3m5q7s9v1x3z5b7d9f0003"
    with api.client() as cloud, pytest.raises(ApiError) as missing:
        cloud.knowledge.base(KB).get_section(document_id=other, section="Refund Policy")
    assert missing.value.code == "knowledge_section_not_found"
    assert api.paths() == [f"POST /v1/knowledge-bases/{KB}/query"]
    assert body_of(api.requests[0])["operation"]["document"] == other


def test_versions_create_get_diff_verify_decide_publish_rollback() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        created = cloud.knowledge.versions.create(
            KB,
            change_message="Extend the refund window",
            expected_draft_revision=61,
            idempotency_key="ci-build-2026-10-08",
        )
        assert created.version.version == 4
        assert created.job is not None
        detail = cloud.knowledge.versions.get(KB, "v4")
        assert detail.manifest["schema"] == "agenomic.knowledge_version_manifest/v1"
        assert detail.approvals
        diff = cloud.knowledge.versions.diff(KB, 4, against="3")
        assert diff.from_version is not None
        assert diff.from_version.version == 3
        assert diff.to_version.version == 4
        assert diff.summary.documents_modified >= 0
        verified = cloud.knowledge.versions.verify(KB, 3)
        assert verified.valid is True
        approved = cloud.knowledge.versions.approve(KB, 4, reason="Checked with legal")
        assert approved.version.status == "approved"
        cloud.knowledge.versions.reject(KB, 4)
        cloud.knowledge.versions.publish(KB, 4, if_match=4)
        rolled = cloud.knowledge.rollback(KB, reason="Customs answer was wrong", if_match=5)
        assert rolled.knowledge_base.kb_id == KB
        cloud.knowledge.versions.rollback(KB, reason="again", if_match=6, to_version="v3")
    create = api.last("POST", "/versions$")
    assert body_of(create) == {
        "change_message": "Extend the refund window",
        "expected_draft_revision": 61,
        "idempotency_key": "ci-build-2026-10-08",
    }
    assert "idempotency-key" not in create.headers
    assert dict(api.last("GET", "/diff$").url.params) == {"against": "3"}
    assert body_of(api.last("POST", "/approve$")) == {"reason": "Checked with legal"}
    assert body_of(api.last("POST", "/reject$")) == {}
    rollbacks = [r for r in api.requests if r.url.path.endswith("/rollback")]
    assert rollbacks[0].headers["if-match"] == '"5"'
    assert body_of(rollbacks[0]) == {"reason": "Customs answer was wrong"}
    assert body_of(rollbacks[1]) == {"reason": "again", "to_version": 3}
    assert rollbacks[1].headers["if-match"] == '"6"'


def test_jobs_get_and_wait_for(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeKnowledgeApi()
    states = iter(["pending", "running", "succeeded"])

    def job_reply(_: httpx.Request) -> dict[str, Any]:
        job = fixture("job")
        job["job"]["status"] = next(states)
        return job

    api.override("GET", f"/v1/knowledge-jobs/{JOB}", job_reply)
    sleeps: list[float] = []
    monkeypatch.setattr(knowledge_resources, "_sleep", sleeps.append)
    with api.client() as cloud:
        job = cloud.knowledge.jobs.wait_for(JOB, timeout=30, poll_interval=0.5)
    assert job.status == "succeeded"
    assert job.terminal
    assert sleeps == [0.5, 0.5]


def test_wait_for_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeKnowledgeApi()
    clock = iter([0.0, 0.4, 1.2])
    monkeypatch.setattr(knowledge_resources, "_monotonic", lambda: next(clock))
    monkeypatch.setattr(knowledge_resources, "_sleep", lambda _: None)
    with api.client() as cloud, pytest.raises(ApiError) as timeout:
        cloud.knowledge.jobs.wait_for(JOB, timeout=1, poll_interval=0.5)
    assert timeout.value.code == "knowledge_job_timeout"
    assert timeout.value.details == {"job_id": JOB, "status": "running"}


def test_async_wait_for(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeKnowledgeApi()
    states = iter(["running", "failed"])

    def job_reply(_: httpx.Request) -> dict[str, Any]:
        job = fixture("job")
        job["job"]["status"] = next(states)
        job["job"]["error_code"] = "parse_failed"
        return job

    api.override("GET", f"/v1/knowledge-jobs/{JOB}", job_reply)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(knowledge_resources, "_asleep", fake_sleep)

    async def run() -> str:
        async with api.client() as cloud:
            job = await cloud.knowledge.jobs.await_for(JOB, poll_interval=2)
            return job.status

    assert asyncio.run(run()) == "failed"
    assert sleeps == [2]


def test_retrievals_get_list_and_snapshot() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        event = cloud.knowledge.retrievals.get(EVENT)
        assert event.query_text is None
        assert event.query_digest.startswith("sha256:")
        page = cloud.knowledge.retrievals.list(kb_id=KB, agent_id=AGENT, limit=20)
        assert page.items[0].event_id == EVENT
        snapshot = cloud.knowledge.retrievals.snapshot(EVENT)
        assert snapshot.verified is True
        assert snapshot.evidence
    assert dict(api.last("GET", "/retrievals$").url.params) == {
        "kb_id": KB,
        "agent_id": AGENT,
        "limit": "20",
    }


def test_agent_knowledge_get_put_attach_detach_and_search() -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        knowledge = cloud.knowledge.agents.get(AGENT)
        assert knowledge.config.revision == 4
        assert knowledge.manifest.digest.startswith("sha256:")
        assert knowledge.bindings[0].version == 3
        replaced = cloud.knowledge.agents.put(
            AGENT,
            if_match=4,
            enabled=True,
            retrieval={"top_k": 5},
            bindings=[
                {"knowledge_base": KB, "version": "v3", "collections": ["faq"]},
                {"knowledge_base": "kb_compliance", "version": "published"},
            ],
        )
        assert replaced.agent_id == AGENT
        attached = cloud.knowledge.agents.attach(
            AGENT, KB, version=3, access="read", max_classification="internal", enabled=True
        )
        assert attached.binding.binding_id == BINDING
        assert cloud.knowledge.agents.detach(AGENT, BINDING) is None
        found = cloud.knowledge.agent_search(
            AGENT,
            "Can a customer be verified with an email address only?",
            knowledge_base=KB,
            top_k=5,
            mode="hybrid",
            filters={"collections": ["faq"]},
            include_context=True,
            execution={"binding_id": PROMPT_BINDING, "run_id": "run_1"},
        )
        assert isinstance(found, AgentSearchResponse)
        assert found.execution.resolved_via == "release_genome"
        assert {item.kb_id for item in found.retrievals} == {"kb_compliance", KB}
    put = api.last("PUT")
    assert put.headers["if-match"] == '"4"'
    assert body_of(put) == {
        "enabled": True,
        "bindings": [
            {"knowledge_base": KB, "version": 3, "collections": ["faq"]},
            {"knowledge_base": "kb_compliance", "version": "published"},
        ],
        "retrieval": {"top_k": 5},
    }
    assert body_of(api.last("POST", "/bindings$")) == {
        "knowledge_base": KB,
        "version": 3,
        "access": "read",
        "max_classification": "internal",
        "enabled": True,
    }
    assert api.last("DELETE").url.path.endswith(f"/bindings/{BINDING}")
    assert body_of(api.last("POST", "/knowledge/search$")) == {
        "query": "Can a customer be verified with an email address only?",
        "knowledge_base": KB,
        "top_k": 5,
        "mode": "hybrid",
        "filters": {"collections": ["faq"]},
        "include_context": True,
        "execution": {"binding_id": PROMPT_BINDING, "run_id": "run_1"},
    }


def test_async_twins_follow_the_same_flows() -> None:
    api = FakeKnowledgeApi()

    async def run() -> list[str]:
        async with api.client() as cloud:
            kb = await cloud.knowledge.aget(KB)
            await kb.asearch("refund", version=3)
            await kb.aget_document(DOC)
            await kb.aget_section(document_id=DOC, section="Refund Policy")
            await kb.aversions()
            await kb.apublish(4, if_match=4)
            await cloud.knowledge.alist()
            await cloud.knowledge.collections.alist(KB)
            await cloud.knowledge.documents.alist(KB)
            await cloud.knowledge.documents.asections(KB, DOC)
            await cloud.knowledge.aquery(KB, "search refund")
            await cloud.knowledge.aanswer(KB, "refund?")
            await cloud.knowledge.versions.aget(KB, 4)
            await cloud.knowledge.jobs.aget(JOB)
            await cloud.knowledge.retrievals.asnapshot(EVENT)
            await cloud.knowledge.agents.aget(AGENT)
            response = await cloud.knowledge.aagent_search(AGENT, "verify customer")
            return [item.kb_id for item in response.results]

    assert asyncio.run(run()) == ["kb_compliance", KB]
    assert len(api.requests) == 17


def test_server_errors_map_to_api_error_with_code_and_details() -> None:
    api = FakeKnowledgeApi()
    api.override(
        "POST",
        f"/v1/knowledge-bases/{KB}/search",
        lambda _: error_response("knowledge_access_denied", 403, "denied", reason="classification"),
    )
    api.override(
        "POST",
        f"/v1/knowledge-bases/{KB}/publish",
        lambda _: error_response("knowledge_publication_conflict", 412, current=5),
    )
    with api.client() as cloud:
        with pytest.raises(ApiError) as denied:
            cloud.knowledge.search(KB, "secret")
        assert denied.value.code == "knowledge_access_denied"
        assert denied.value.status == 403
        assert denied.value.reason == "classification"
        assert denied.value.request_id == "req_01knowledge"
        with pytest.raises(ApiError) as conflict:
            cloud.knowledge.publish(KB, 4, if_match=4)
        assert conflict.value.code == "knowledge_publication_conflict"
        assert conflict.value.details["current"] == 5
        with pytest.raises(ApiError) as missing:
            cloud.knowledge.get("kb_unknown")
        assert missing.value.code == "not_found"


def test_search_retries_unavailable_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenomic import _transport

    monkeypatch.setattr(_transport, "_sleep", lambda _: None)
    api = FakeKnowledgeApi()
    replies = iter([httpx.Response(503), httpx.Response(200, json=fixture("search"))])
    api.override("POST", f"/v1/knowledge-bases/{KB}/search", lambda _: next(replies))
    with api.client() as cloud:
        assert cloud.knowledge.search(KB, "refund").results
    assert len(api.requests) == 2


def test_answer_is_sent_once_on_ambiguous_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenomic import _transport

    monkeypatch.setattr(_transport, "_sleep", lambda _: None)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("the answer took too long", request=request)

    answer = f"/v1/knowledge-bases/{KB}/answer"
    for reply in (
        lambda _: httpx.Response(502),
        lambda _: httpx.Response(504),
        lambda _: error_response("internal_error", 500),
        timeout,
    ):
        api = FakeKnowledgeApi()
        api.override("POST", answer, reply)
        with api.client() as cloud, pytest.raises(ApiError):
            cloud.knowledge.answer(KB, "refund window?")
        assert api.paths() == [f"POST {answer}"]


def test_async_answer_is_sent_once_on_ambiguous_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agenomic import _transport

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(_transport, "_asleep", no_sleep)
    answer = f"/v1/knowledge-bases/{KB}/answer"
    api = FakeKnowledgeApi()
    api.override("POST", answer, lambda _: httpx.Response(503))

    async def run() -> None:
        async with api.client() as cloud:
            with pytest.raises(ApiError):
                await cloud.knowledge.aanswer(KB, "refund window?")

    asyncio.run(run())
    assert api.paths() == [f"POST {answer}"]


def test_idempotent_reads_still_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenomic import _transport

    monkeypatch.setattr(_transport, "_sleep", lambda _: None)
    api = FakeKnowledgeApi()
    replies = iter([httpx.Response(502), httpx.Response(200, json=fixture("knowledge_base"))])
    api.override("GET", f"/v1/knowledge-bases/{KB}", lambda _: next(replies))
    with api.client() as cloud:
        assert cloud.knowledge.get(KB).kb_id == KB
    assert api.paths() == [f"GET /v1/knowledge-bases/{KB}"] * 2


def test_invalid_response_shapes_are_refused() -> None:
    api = FakeKnowledgeApi()
    api.override("POST", f"/v1/knowledge-bases/{KB}/search", lambda _: {"results": "nope"})
    api.override("GET", f"/v1/knowledge-jobs/{JOB}", lambda _: {"job": {"job_id": "kjob_other"}})
    api.override(
        "GET",
        f"/v1/knowledge-bases/{KB}",
        lambda _: {**fixture("knowledge_base"), "knowledge_base": {"kb_id": "kb_other"}},
    )
    api.override("GET", f"/v1/knowledge-bases/{KB}/collections", lambda _: {"items": []})
    with api.client() as cloud:
        for call in (
            lambda: cloud.knowledge.search(KB, "refund"),
            lambda: cloud.knowledge.jobs.get(JOB),
            lambda: cloud.knowledge.get(KB),
            lambda: cloud.knowledge.collections.list(KB),
        ):
            with pytest.raises(ApiError) as invalid:
                call()
            assert invalid.value.code == "invalid_response"


def test_arguments_checked_before_any_request(tmp_path: Path) -> None:
    api = FakeKnowledgeApi()
    with api.client() as cloud:
        kb = cloud.knowledge.base(KB)
        for call in (
            lambda: cloud.knowledge.get("Bad Id"),
            lambda: cloud.knowledge.search(KB, " "),
            lambda: cloud.knowledge.search(KB, "x", top_k=51),
            lambda: cloud.knowledge.search(KB, "x", mode="fuzzy"),  # type: ignore[arg-type]
            lambda: cloud.knowledge.search(KB, "x", version="latest"),
            lambda: cloud.knowledge.search(KB, "x", filters={"author": "x"}),
            lambda: cloud.knowledge.query(KB),
            lambda: cloud.knowledge.query(KB, "x", operation={"op": "search", "text": "x"}),
            lambda: cloud.knowledge.query(KB, operation={"op": "drop"}),
            lambda: cloud.knowledge.documents.upload(KB, b"bytes"),
            lambda: cloud.knowledge.documents.upload(KB, b"x", path="a.md", tags=["a,b"]),
            lambda: cloud.knowledge.documents.upload(KB, b"x", path="a.md", tags=[" "]),
            lambda: cloud.knowledge.documents.put_content(KB, DOC, "x", if_match=0),
            lambda: cloud.knowledge.versions.get(KB, "published"),
            lambda: cloud.knowledge.publish(KB, "draft", if_match=1),
            lambda: cloud.knowledge.agents.attach(AGENT, KB, version="draft"),
            lambda: cloud.knowledge.agent_search(AGENT, "x", execution={"agent_id": "x"}),
            lambda: cloud.knowledge.list(limit=201),
            lambda: cloud.knowledge.jobs.wait_for(JOB, poll_interval=0),
            lambda: kb.get_section(section="Refund Policy"),
            lambda: kb.get_section(document="Refunds", section=SECTION),
        ):
            with pytest.raises(ValueError):
                call()
        with pytest.raises(TypeError):
            cloud.knowledge.documents.upload(KB, 12, path="a.md")  # type: ignore[arg-type]
        with pytest.raises(FileNotFoundError):
            cloud.knowledge.documents.upload(KB, tmp_path / "missing.md")
    assert api.requests == []


def test_local_mode_raises_cloud_required() -> None:
    local = Client()
    for call in (
        lambda: local.knowledge.get(KB),
        lambda: local.knowledge.search(KB, "refund"),
        lambda: local.knowledge.base(KB).search("refund"),
        lambda: local.knowledge.jobs.get(JOB),
        lambda: local.knowledge.agent_search(AGENT, "refund"),
    ):
        with pytest.raises(ApiError) as refused:
            call()
        assert refused.value.code == "cloud_required"


def test_raw_bytes_transport_keeps_json_callers_unchanged() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True}, headers={"etag": '"7"'})

    with Client(api_key="k", base_url=BASE, transport=httpx.MockTransport(handler)) as cloud:
        response = api_request(cloud, "POST", "/v1/x", {"a": 1}, if_match=3)
        assert response.etag == 7
        assert response.body == {"ok": True}
    assert seen[0].headers["content-type"] == "application/json"
    assert seen[0].headers["if-match"] == '"3"'


def test_knowledge_import_stays_framework_free() -> None:
    code = (
        "import sys, agenomic, agenomic.knowledge, agenomic.integrations\n"
        "from agenomic import Client\n"
        "Client(api_key='k', base_url='https://x').knowledge\n"
        "loaded = [m for m in ('langgraph', 'langchain_core') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "assert 'agenomic.knowledge._langchain' not in sys.modules\n"
        "tool = agenomic.integrations.knowledge_tool\n"
        "assert 'langchain_core' not in sys.modules\n"
        "assert callable(tool)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
