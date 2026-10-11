from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest
from knowledge_fakes import fixture

from agenomic import Client
from agenomic.canonical.recorder import start_run
from agenomic.knowledge import AgentSearchResponse, SearchResponse
from agenomic.tracking import TRACKING_EVENT_TYPES

MANIFEST = "sha256:" + "5" * 64


def test_knowledge_retrieve_is_a_tracking_event_type() -> None:
    assert "knowledge.retrieve" in TRACKING_EVENT_TYPES
    session = Client().tracking.start(agent="agent://acme/support")
    refs = SearchResponse.model_validate(fixture("search")).knowledge_refs()
    event = session.event("knowledge.retrieve", metadata={"knowledge_refs": refs["citations"]})
    assert event["type"] == "knowledge.retrieve"
    session.stop()


def test_knowledge_refs_carry_references_and_digests_only() -> None:
    response = SearchResponse.model_validate(fixture("search"))
    refs = response.knowledge_refs()
    assert refs["agent_manifest_digest"] is None
    assert refs["retrievals"] == [
        {
            "event_id": "kret_01jb3m5q7s9v1x3z5b7d9f0001",
            "kb_id": "kb_customer_support",
            "version": 3,
            "version_manifest_digest": "sha256:" + "3" * 64,
            "index_config_digest": "sha256:" + "1" * 64,
            "mode": "hybrid",
        }
    ]
    assert set(refs["citations"][0]) == {
        "uri",
        "kb_id",
        "version",
        "document_id",
        "document_revision",
        "section_id",
        "chunk_id",
        "content_digest",
    }
    dumped = json.dumps(refs)
    for result in response.results:
        assert result.text not in dumped
    agent = AgentSearchResponse.model_validate(fixture("agent_search")).knowledge_refs()
    assert agent["agent_manifest_digest"] == MANIFEST
    assert len(agent["retrievals"]) == 2


def test_log_knowledge_records_a_schema_valid_event(
    v03_errors: Callable[[dict[str, Any]], list[str]],
) -> None:
    run = start_run("agent://acme/support")
    response = AgentSearchResponse.model_validate(fixture("agent_search"))
    leaky = [{**item, "text": "secret body"} for item in response.knowledge_refs()["citations"]]
    run.log_knowledge(
        retrievals=response.knowledge_refs()["retrievals"],
        citations=leaky,
        query="Can a customer be verified with an email address only?",
        agent_manifest_digest=MANIFEST,
    )
    trace = run.complete_run(output={"answer": "no"})
    assert v03_errors(trace) == []
    event = trace["events"][1]
    assert event["type"] == "knowledge.retrieve"
    assert event["actor"]["kind"] == "agent"
    payload = event["redacted_payload"]
    expected = b"Can a customer be verified with an email address only?"
    assert payload["query_digest"] == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert payload["agent_manifest_digest"] == MANIFEST
    assert "secret body" not in json.dumps(trace)
    assert all("text" not in item for item in payload["citations"])
    assert trace["components"]["knowledge_version"] == MANIFEST


def test_knowledge_manifest_digest_component() -> None:
    placeholder = start_run("agent://acme/support").complete_run()["components"]
    assert placeholder["knowledge_version"] == (
        "sha256:" + hashlib.sha256(b"knowledge").hexdigest()
    )
    pinned = start_run("agent://acme/support", knowledge_manifest_digest=MANIFEST)
    assert pinned.complete_run()["components"]["knowledge_version"] == MANIFEST
    with pytest.raises(ValueError):
        start_run("agent://acme/support", knowledge_manifest_digest="sha256:short")
    run = start_run("agent://acme/support", knowledge_manifest_digest=MANIFEST)
    run.log_knowledge(retrievals=[], agent_manifest_digest=MANIFEST)
    with pytest.raises(ValueError):
        run.log_knowledge(retrievals=[], agent_manifest_digest="sha256:" + "6" * 64)
    with pytest.raises(ValueError):
        run.log_knowledge(retrievals=[], agent_manifest_digest="nope")
