from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import httpx
import pytest
from experiment_fakes import SPEC_DIGEST, SPEC_FIXTURE, FakeExperimentApi

from agenomic import Client
from agenomic.exceptions import ApiError
from agenomic.experiments import spec_digest

BASE = "https://cloud.test"
DRAFT = {"name": "planner v8", "agent_id": "2b1e5c3a-8d4f-4e6a-9b0c-1d2e3f4a5b6c", "level": "agent"}
AUTHORIZATION = {
    "paid_model_usage": {
        "acknowledged": True,
        "max_total_tokens": 2000000,
        "max_cost_micros": None,
    },
    "live_tools": None,
    "tool_plan_hashes": {},
    "acknowledged_confounders": [],
}


def client(api: FakeExperimentApi) -> Client:
    return Client(api_key="agm_test", base_url=BASE, transport=api.transport())


def test_spec_digest_matches_spec_vector() -> None:
    spec = json.loads(SPEC_FIXTURE.read_text(encoding="utf-8"))
    assert spec_digest(spec) == SPEC_DIGEST
    spec["identity"]["arm_keys"]["cand_1"] = "arm_00000000"
    assert spec_digest(spec) == SPEC_DIGEST
    spec["repetitions"] = 6
    assert spec_digest(spec) != SPEC_DIGEST
    spec["analysis"]["alpha"] = 0.05
    with pytest.raises(ValueError):
        spec_digest(spec)


def test_create_update_preflight_launch_flow() -> None:
    api = FakeExperimentApi()
    with client(api) as cloud:
        created = cloud.experiments.create(DRAFT)
        assert created["revision"] == 1
        updated = cloud.experiments.update(created["experiment_id"], DRAFT, expected_revision=1)
        assert updated["revision"] == 2
        preflight = cloud.experiments.preflight("exp_01fake", expected_revision=2)
        assert preflight["spec_digest"] == SPEC_DIGEST
        fetched = cloud.experiments.get("exp_01fake")
        assert fetched["spec_digest"] == SPEC_DIGEST
        launched = cloud.experiments.launch(
            "exp_01fake",
            expected_revision=2,
            spec_digest=preflight["spec_digest"],
            authorization=AUTHORIZATION,
            idempotency_key="launch-exp-01fake-1",
        )
        assert launched["replayed"] is False
        assert launched["trials_planned"] == 8
        replay = cloud.experiments.launch(
            "exp_01fake",
            expected_revision=2,
            spec_digest=SPEC_DIGEST,
            authorization=AUTHORIZATION,
            idempotency_key="launch-exp-01fake-1",
        )
        assert replay["replayed"] is True
        assert (
            cloud.experiments.cancel("exp_01fake", reason="enough")["experiment"]["status"]
            == "cancelling"
        )
        assert cloud.experiments.results("exp_01fake")["interim"] is True
        events = cloud.experiments.events("exp_01fake", after=3, limit=10)
        assert events["next_after"] == 4
    launch_requests = [r for r in api.requests if r.url.path.endswith("/launch")]
    assert all("idempotency-key" not in r.headers for r in launch_requests)
    assert json.loads(launch_requests[0].content)["idempotency_key"] == "launch-exp-01fake-1"
    put = next(r for r in api.requests if r.method == "PUT")
    assert put.headers["if-match"] == '"1"'
    preflight_request = next(r for r in api.requests if r.url.path.endswith("/preflight"))
    assert preflight_request.headers["if-match"] == '"2"'
    events_request = next(r for r in api.requests if r.url.path.endswith("/events"))
    assert events_request.url.params["after"] == "3"
    assert events_request.url.params["limit"] == "10"


def test_async_twins_follow_the_same_flow() -> None:
    api = FakeExperimentApi()

    async def run() -> dict[str, object]:
        async with client(api) as cloud:
            created = await cloud.experiments.acreate(DRAFT)
            await cloud.experiments.aupdate(created["experiment_id"], DRAFT, expected_revision=1)
            preflight = await cloud.experiments.apreflight("exp_01fake", expected_revision=2)
            await cloud.experiments.alaunch(
                "exp_01fake",
                expected_revision=2,
                spec_digest=preflight["spec_digest"],
                authorization=AUTHORIZATION,
                idempotency_key="launch-async-1",
            )
            await cloud.experiments.acancel("exp_01fake", reason="stop")
            await cloud.experiments.aresults("exp_01fake")
            await cloud.experiments.aevents("exp_01fake")
            return await cloud.experiments.aget("exp_01fake")

    assert asyncio.run(run())["status"] == "cancelling"


def test_revision_conflict_and_tampered_spec_are_refused() -> None:
    api = FakeExperimentApi()
    with client(api) as cloud:
        cloud.experiments.create(DRAFT)
        with pytest.raises(ApiError) as conflict:
            cloud.experiments.update("exp_01fake", DRAFT, expected_revision=7)
        assert conflict.value.code == "experiment_revision_conflict"
        assert conflict.value.details["current"] == 1
        api.tamper_spec = True
        cloud.experiments.preflight("exp_01fake", expected_revision=1)
        with pytest.raises(ApiError) as tampered:
            cloud.experiments.get("exp_01fake")
        assert tampered.value.code == "experiment_spec_digest_mismatch"
        with pytest.raises(ApiError) as missing:
            cloud.experiments.get("exp_unknown")
        assert missing.value.code == "experiment_not_found"


def test_arguments_checked_before_any_request() -> None:
    api = FakeExperimentApi()
    with client(api) as cloud:
        for call in (
            lambda: cloud.experiments.launch(
                "exp_01fake",
                expected_revision=1,
                spec_digest="sha256:short",
                authorization={},
                idempotency_key="k",
            ),
            lambda: cloud.experiments.launch(
                "exp_01fake",
                expected_revision=1,
                spec_digest=SPEC_DIGEST,
                authorization={},
                idempotency_key="has space",
            ),
            lambda: cloud.experiments.update("exp_01fake", DRAFT, expected_revision=0),
            lambda: cloud.experiments.events("exp_01fake", after=-1),
            lambda: cloud.experiments.events("exp_01fake", limit=501),
            lambda: cloud.experiments.get(""),
        ):
            with pytest.raises(ValueError):
                call()
    assert api.requests == []


def test_local_mode_raises_cloud_required() -> None:
    local = Client()
    with pytest.raises(ApiError) as refused:
        local.experiments.create(DRAFT)
    assert refused.value.code == "cloud_required"
    with pytest.raises(ApiError):
        asyncio.run(local.experiments.aget("exp_01fake"))


def test_invalid_response_shapes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"items": []})
        if request.url.path.endswith("/preflight"):
            return httpx.Response(200, json={"preflight": {"spec_digest": "md5:x"}})
        return httpx.Response(200, headers={"etag": '"3"'}, json={"experiment": {"revision": 2}})

    with Client(api_key="agm_test", base_url=BASE, transport=httpx.MockTransport(handler)) as cloud:
        for call in (
            lambda: cloud.experiments.get("exp_1"),
            lambda: cloud.experiments.events("exp_1"),
            lambda: cloud.experiments.preflight("exp_1", expected_revision=1),
        ):
            with pytest.raises(ApiError) as invalid:
                call()
            assert invalid.value.code == "invalid_response"


def test_rmp_session_carries_candidate_release_id() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"session": {"session_id": "rmp_01"}})

    cloud = Client(api_key="agm_test", base_url=BASE, transport=httpx.MockTransport(handler))
    cloud.rmp.start(
        agent="agent://acme/support", candidate_release_id="a2b3c4d5-0000-4000-8000-000000000001"
    )
    assert seen[0]["candidate_release_id"] == "a2b3c4d5-0000-4000-8000-000000000001"
    local = Client().rmp.start(agent="agent://acme/support", candidate_release_id="rel")
    assert local["candidate_release_id"] == "rel"


def test_experiments_import_stays_framework_free() -> None:
    code = (
        "import sys, agenomic, agenomic.experiments\n"
        "loaded = [m for m in ('langgraph', 'langchain_core') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "assert hasattr(agenomic.experiments, 'ExperimentRunner')\n"
        "assert 'langgraph' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    with pytest.raises(AttributeError):
        _ = __import__("agenomic.experiments").experiments.missing_name


def test_case_documents_parse_with_spec_fixture() -> None:
    from agenomic.experiments import ExperimentCase

    path = SPEC_FIXTURE.parent / "node-state-case.json"
    case = ExperimentCase.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert case.kind == "node_state"
    assert case.context == {"locale": "fr"}
    assert case.initial_state is not None
    assert case.initial_state["confidence"] == 0.75
    assert case.is_snapshot is False
    assert case.provenance is None
