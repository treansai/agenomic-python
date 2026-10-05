from __future__ import annotations

import copy
import hashlib
import json
import operator
import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_world import World
from prompt_fakes import AGENT, REQUEST_ID
from typing_extensions import Annotated, TypedDict

from agenomic.canonical.hashing import canonical_json
from agenomic.exceptions import ApiError
from agenomic.experiments import TrialContext
from agenomic.experiments.runner import ExperimentRunner, GraphTarget, local_assignment
from agenomic.prompts.secrets import scan

TOKEN = "agr_" + "0123456789abcdef" * 4
BASE = "https://cloud.test"
SPEC_FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "prompt-only-spec.json"
SPEC_DIGEST = "sha256:a5157683831c40ec51d389136f70d041aeb527ee7a53e249378488aace313b05"
_RUNNER = "/v1/experiment-runner"


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]


NodeFn = Callable[[TrialContext, dict[str, Any], RunnableConfig], Awaitable[dict[str, Any]]]


def agent_case(case_id: str = "case-1", text: str = "hi", **extra: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": case_id,
        "kind": "agent_input",
        "input": {"messages": [{"role": "user", "content": text}]},
        "expected": None,
        "initial_state": None,
        "turns": [],
        "tags": [],
    }
    case.update(extra)
    return case


def single_node(node: NodeFn) -> Callable[[TrialContext], Any]:
    def factory(ctx: TrialContext) -> Any:
        async def plan(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            return await node(ctx, state, config)

        builder = StateGraph(GraphState)
        builder.add_node("plan", plan)
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=ctx.checkpointer, store=ctx.store)

    return factory


def runtime_digest(world: World) -> str:
    return str(world.engine.get_release(world.releases["v1"])["bundle_hash"])


def make_runner(
    server: FakeRunnerServer,
    factory: Callable[[TrialContext], Any],
    *,
    runner_options: Optional[dict[str, Any]] = None,
    **target_options: Any,
) -> ExperimentRunner:
    target = GraphTarget(
        factory=factory, runtime_digest=runtime_digest(server.world), **target_options
    )
    return ExperimentRunner(
        targets={AGENT: target},
        token=TOKEN,
        base_url=BASE,
        transport=server.transport(),
        **(runner_options or {}),
    )


def error_response(
    code: str, status: int, message: str = "refused", **details: Any
) -> httpx.Response:
    payload: dict[str, Any] = {"code": code, "message": message, "request_id": REQUEST_ID}
    if details:
        payload["details"] = details
    return httpx.Response(status, json={"error": payload})


def strict(body: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict) or set(body) - allowed:
        raise ApiError("validation_error", 400, f"unknown members {sorted(set(body) - allowed)}")
    return body


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass
class FakeTrial:
    assignment: dict[str, Any]
    status: str = "queued"
    lease_token: Optional[str] = None
    accepted_lease: Optional[str] = None
    result_digest: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    accepts: int = 0
    duplicates: int = 0
    heartbeats: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    releases: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    stop_requested: bool = False
    deadline_exceeded: bool = False

    @property
    def view(self) -> dict[str, Any]:
        return self.assignment["view"]


@dataclass
class ToolRecord:
    tool: str
    arguments: Any
    response: dict[str, Any]
    report: Optional[dict[str, Any]] = None


class FakeRunnerServer:
    def __init__(self, world: World, *, heartbeat_interval: float = 0.02) -> None:
        self.world = world
        self.heartbeat_interval = heartbeat_interval
        self.trials: dict[str, FakeTrial] = {}
        self.queue: list[str] = []
        self.requests: list[httpx.Request] = []
        self.hellos: list[dict[str, Any]] = []
        self.fixtures: dict[tuple[str, str], Any] = {}
        self.tool_records: dict[tuple[str, str, int], ToolRecord] = {}
        self.in_progress = 0
        self.drop_result_responses = 0
        self.refuse_result: Optional[tuple[str, int, dict[str, Any]]] = None
        self.on_heartbeat: Optional[Callable[[FakeTrial], None]] = None
        self.on_claim: Optional[Callable[[FakeTrial], None]] = None
        self.drop_tool_call_responses = 0
        self.drop_report_responses = 0
        self.fail_hellos = 0
        self.demand_hello = 0
        self.require_hello = True
        self._lock = threading.RLock()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def add_trial(
        self, release: str, case: dict[str, Any], *, agent_id: str = AGENT, **options: Any
    ) -> str:
        options.setdefault("heartbeat_interval_seconds", self.heartbeat_interval)
        assignment = local_assignment(
            self.world.engine,
            agent_id=agent_id,
            release_id=self.world.releases[release],
            case=case,
            **options,
        )
        trial_id = str(assignment["trial_id"])
        self.trials[trial_id] = FakeTrial(assignment)
        self.queue.append(trial_id)
        return trial_id

    def raw_bodies(self) -> list[bytes]:
        return [request.content for request in self.requests]

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return error_response("runner_token_required", 401)
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        routes: list[tuple[str, Callable[..., httpx.Response]]] = [
            (rf"{_RUNNER}/hello", self._hello),
            (rf"{_RUNNER}/claims", self._claim),
            (rf"{_RUNNER}/trials/([^/]+)/heartbeat", self._heartbeat),
            (rf"{_RUNNER}/trials/([^/]+)/tool-calls", self._tool_call),
            (rf"{_RUNNER}/trials/([^/]+)/tool-calls/([^/]+)/report", self._report),
            (rf"{_RUNNER}/trials/([^/]+)/result", self._result),
            (rf"{_RUNNER}/trials/([^/]+)/failure", self._failure),
            (rf"{_RUNNER}/trials/([^/]+)/release", self._release),
        ]
        for pattern, handler in routes:
            match = re.fullmatch(pattern, path)
            if match and request.method == "POST":
                try:
                    with self._lock:
                        return handler(request, body, *match.groups())
                except ApiError as error:
                    return error_response(error.code, error.status, error.message, **error.details)
        return error_response("not_found", 404)

    def _leased(self, trial_id: str, token: Any) -> FakeTrial:
        trial = self.trials.get(trial_id)
        if trial is None:
            raise ApiError("not_found", 404, "unknown trial")
        if trial.status != "leased" or trial.lease_token != token:
            raise ApiError("experiment_lease_stale", 409, "the lease is not current")
        return trial

    def _hello(self, request: httpx.Request, body: Any) -> httpx.Response:
        if body.get("schema") != "agenomic.experiment_runner_hello/v1":
            raise ApiError("validation_error", 400, "not a hello document")
        if self.hellos and self.fail_hellos > 0:
            self.fail_hellos -= 1
            raise ApiError("service_unavailable", 503, "the gateway is restarting")
        self.hellos.append(body)
        return httpx.Response(
            200,
            json={
                "runner_id": "exr_01fake",
                "heartbeat_interval_seconds": 30,
                "claim_wait_max_seconds": 25,
                "server_time": "2026-10-05T12:00:00Z",
            },
        )

    def _claim(self, request: httpx.Request, body: Any) -> httpx.Response:
        strict(body, {"wait_seconds"})
        if self.require_hello and not self.hellos:
            raise ApiError("experiment_runner_hello_required", 409, "send a hello first")
        if self.demand_hello > 0:
            self.demand_hello -= 1
            raise ApiError("experiment_runner_hello_required", 409, "the hello is too old")
        if not self.queue:
            return httpx.Response(204)
        trial = self.trials[self.queue.pop(0)]
        trial.status = "leased"
        trial.lease_token = str(trial.assignment["lease_token"])
        if self.on_claim is not None:
            self.on_claim(trial)
        return httpx.Response(200, json={"assignment": copy.deepcopy(trial.assignment)})

    def _heartbeat(self, request: httpx.Request, body: Any, trial_id: str) -> httpx.Response:
        strict(body, {"lease_token"})
        trial = self._leased(trial_id, body.get("lease_token"))
        trial.heartbeats += 1
        if self.on_heartbeat is not None:
            self.on_heartbeat(trial)
        return httpx.Response(
            200,
            json={
                "lease_until": "2026-10-05T12:02:00Z",
                "cancel_requested": trial.cancel_requested,
                "stop_requested": trial.stop_requested,
                "deadline_exceeded": trial.deadline_exceeded,
            },
        )

    def _tool_call(self, request: httpx.Request, body: Any, trial_id: str) -> httpx.Response:
        strict(
            body,
            {"lease_token", "logical_call_id", "attempt", "tool", "arguments", "parent_call_id"},
        )
        trial = self._leased(trial_id, body.get("lease_token"))
        if self.in_progress > 0:
            self.in_progress -= 1
            raise ApiError("experiment_tool_call_in_progress", 409, "the call is still pending")
        mode = trial.view["tools"]["mode"]
        if mode == "none":
            raise ApiError("experiment_tool_mode_none", 409, "the experiment declares no tools")
        key = (trial_id, str(body["logical_call_id"]), int(body["attempt"]))
        record = self.tool_records.get(key)
        if record is not None:
            if record.tool != body["tool"] or digest(record.arguments) != digest(body["arguments"]):
                raise ApiError("experiment_tool_call_id_reused", 409, "the id names another call")
            if record.report is not None:
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "result": _envelope(record.report["value"]),
                        "replayed": True,
                    },
                )
            return httpx.Response(200, json={**record.response, "replayed": True})
        response = self._answer(trial, mode, str(body["tool"]), body["arguments"])
        self.tool_records[key] = ToolRecord(str(body["tool"]), body["arguments"], response)
        if self.drop_tool_call_responses > 0:
            self.drop_tool_call_responses -= 1
            raise httpx.ConnectError("response lost", request=request)
        return httpx.Response(200, json=response)

    def _answer(self, trial: FakeTrial, mode: str, tool: str, arguments: Any) -> dict[str, Any]:
        if mode == "mock":
            return {
                "status": "ok",
                "result": _envelope({"mock": tool, "arguments": arguments}),
                "replayed": False,
            }
        if mode == "recorded":
            found = self.fixtures.get((tool, canonical_json(arguments)))
            if found is None:
                return {
                    "status": "recorded_fixture_miss",
                    "tool": tool,
                    "arguments_hash": "blake3:"
                    + hashlib.sha256(canonical_json(arguments).encode()).hexdigest(),
                    "on_fixture_miss": trial.view["tools"]["on_fixture_miss"],
                    "message": "No recorded response matches this tool and these canonical arguments.",
                }
            return {"status": "ok", "result": _envelope(found), "replayed": False}
        return {
            "status": "authorized",
            "record_id": str(uuid.uuid4()),
            "permit": {"document": {"tool": tool}, "signature": {"alg": "ed25519"}},
            "decision": {"outcome": "allow"},
        }

    def _report(
        self, request: httpx.Request, body: Any, trial_id: str, logical_call_id: str
    ) -> httpx.Response:
        strict(body, {"lease_token", "attempt", "value", "is_error", "duration_ms", "permit"})
        self._leased(trial_id, body.get("lease_token"))
        record = self.tool_records.get((trial_id, logical_call_id, int(body["attempt"])))
        if record is None or record.response.get("status") != "authorized":
            raise ApiError("not_found", 404, "no authorized call")
        settled = {"value": body["value"], "is_error": body["is_error"]}
        if record.report is not None:
            if digest(record.report) != digest(settled):
                raise ApiError("experiment_result_conflict", 409, "another value was reported")
            return httpx.Response(200, json={"recorded": True, "duplicate": True})
        record.report = settled
        if self.drop_report_responses > 0:
            self.drop_report_responses -= 1
            raise httpx.ConnectError("response lost", request=request)
        return httpx.Response(
            200, json={"recorded": True, "record_id": record.response["record_id"]}
        )

    def _result(self, request: httpx.Request, body: Any, trial_id: str) -> httpx.Response:
        strict(body, {"lease_token", "result"})
        if scan(request.content.decode("utf-8")):
            raise ApiError("experiment_result_secret_detected", 400, "a secret pattern matched")
        if self.refuse_result is not None:
            code, status, details = self.refuse_result
            self.refuse_result = None
            raise ApiError(code, status, "refused", details)
        trial = self.trials.get(trial_id)
        if trial is None:
            raise ApiError("not_found", 404, "unknown trial")
        result = body["result"]
        result_digest = digest(result)
        token = body.get("lease_token")
        if trial.status == "completed" and trial.accepted_lease == token:
            if trial.result_digest == result_digest:
                trial.duplicates += 1
                return httpx.Response(200, json={"accepted": True, "duplicate": True})
            raise ApiError("experiment_result_conflict", 409, "another result was accepted")
        self._leased(trial_id, token)
        if result.get("runner_view_digest") != trial.assignment["runner_view_digest"]:
            raise ApiError(
                "experiment_result_invalid", 400, "view digest", {"reason": "view_digest"}
            )
        prompts = trial.view["prompts"]["prompts"]
        for call in result.get("model_calls", []):
            ref = call.get("prompt_ref")
            if ref is not None and (
                ref not in prompts or prompts[ref]["content_digest"] != call.get("content_digest")
            ):
                raise ApiError(
                    "experiment_result_invalid",
                    400,
                    "outside",
                    {"reason": "prompt_outside_manifest"},
                )
        trial.status = "completed"
        trial.accepted_lease = token
        trial.result_digest = result_digest
        trial.result = result
        trial.accepts += 1
        if self.drop_result_responses > 0:
            self.drop_result_responses -= 1
            raise httpx.ConnectError("response lost", request=request)
        return httpx.Response(200, json={"accepted": True, "duplicate": False})

    def _failure(self, request: httpx.Request, body: Any, trial_id: str) -> httpx.Response:
        strict(body, {"lease_token", "error_class", "error_code", "message"})
        trial = self._leased(trial_id, body.get("lease_token"))
        trial.failures.append(body)
        trial.status = "failed"
        return httpx.Response(200, json={"trial_status": "failed", "retry_at": None})

    def _release(self, request: httpx.Request, body: Any, trial_id: str) -> httpx.Response:
        strict(body, {"lease_token", "reason"})
        trial = self._leased(trial_id, body.get("lease_token"))
        trial.releases.append(body)
        trial.status = {"cancelled": "cancelled", "stopped": "completed"}.get(
            body["reason"], "queued"
        )
        return httpx.Response(204)


def _envelope(value: Any) -> dict[str, Any]:
    return {
        "result": value,
        "agenomic": {
            "record_id": str(uuid.uuid4()),
            "status": "success",
            "provenance": {"source": "static"},
            "external_state": "none",
        },
    }


class FakeExperimentApi:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.experiments: dict[str, dict[str, Any]] = {}
        self.launches: dict[str, dict[str, Any]] = {}
        self.tamper_spec = False

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        method = request.method
        if path == "/v1/experiments" and method == "POST":
            experiment_id = "exp_01fake"
            experiment = {
                "experiment_id": experiment_id,
                "revision": 1,
                "status": "draft",
                "draft": body,
            }
            self.experiments[experiment_id] = experiment
            return httpx.Response(201, headers={"etag": '"1"'}, json={"experiment": experiment})
        match = re.fullmatch(r"/v1/experiments/([^/]+)(?:/([a-z]+))?", path)
        if match is None or match.group(1) not in self.experiments:
            return error_response("experiment_not_found", 404)
        experiment = self.experiments[match.group(1)]
        action = match.group(2)
        if_match = request.headers.get("if-match")
        if action is None and method == "GET":
            return httpx.Response(
                200,
                headers={"etag": f'"{experiment["revision"]}"'},
                json={"experiment": experiment},
            )
        if action is None and method == "PUT" or action == "preflight":
            if if_match is None:
                return error_response("if_match_required", 400)
            if if_match != f'"{experiment["revision"]}"':
                return error_response(
                    "experiment_revision_conflict", 409, current=experiment["revision"]
                )
            if action is None:
                experiment.update(revision=experiment["revision"] + 1, status="draft", draft=body)
                return httpx.Response(
                    200,
                    headers={"etag": f'"{experiment["revision"]}"'},
                    json={"experiment": experiment},
                )
            spec = json.loads(SPEC_FIXTURE.read_text(encoding="utf-8"))
            if self.tamper_spec:
                spec["repetitions"] = 6
            experiment.update(status="preflight_passed", spec=spec, spec_digest=SPEC_DIGEST)
            return httpx.Response(
                200,
                json={
                    "preflight": {
                        "status": "passed",
                        "revision": experiment["revision"],
                        "spec_digest": SPEC_DIGEST,
                        "checks": [],
                    }
                },
            )
        if action == "launch" and method == "POST":
            key = body["idempotency_key"]
            previous = self.launches.get(key)
            if previous is not None:
                if previous != body:
                    return error_response("idempotency_key_reused", 409)
                return httpx.Response(
                    200, json={"replayed": True, "experiment": experiment, "trials_planned": 8}
                )
            if body["spec_digest"] != experiment.get("spec_digest"):
                return error_response("experiment_preflight_stale", 409)
            self.launches[key] = body
            experiment["status"] = "queued"
            return httpx.Response(
                201, json={"replayed": False, "experiment": experiment, "trials_planned": 8}
            )
        if action == "cancel" and method == "POST":
            experiment["status"] = "cancelling"
            return httpx.Response(200, json={"experiment": experiment})
        if action == "results" and method == "GET":
            return httpx.Response(
                200, json={"experiment_id": match.group(1), "interim": True, "verdict": None}
            )
        if action == "events" and method == "GET":
            after = int(request.url.params.get("after", "0"))
            events = [
                {
                    "sequence": after + 1,
                    "kind": "experiment.launched",
                    "message": "",
                    "trial_id": None,
                    "payload": {},
                    "occurred_at": "2026-10-05T12:00:00Z",
                }
            ]
            return httpx.Response(200, json={"events": events, "next_after": after + 1})
        return error_response("not_found", 404)
