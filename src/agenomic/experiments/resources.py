from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

from agenomic._transport import ApiResponse, segment
from agenomic.exceptions import ApiError
from agenomic.experiments.models import spec_digest
from agenomic.prompts.resources import Call, Flow, arun_flow, run_flow

if TYPE_CHECKING:
    from agenomic._client import Client

__all__ = ["ExperimentsResource"]

_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}", re.ASCII)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)


def _cloud_required(operation: str) -> ApiError:
    return ApiError(
        "cloud_required",
        0,
        f"{operation} needs Agenomic Cloud; experiments run only on customer runners",
    )


def _member(response: ApiResponse, key: str) -> dict[str, Any]:
    value = response.body.get(key)
    if not isinstance(value, dict):
        raise ApiError("invalid_response", response.status, f"the response carries no {key} object")
    return value


def _verified(response: ApiResponse) -> dict[str, Any]:
    experiment = _member(response, "experiment")
    revision = experiment.get("revision")
    if response.etag is not None and isinstance(revision, int) and revision != response.etag:
        raise ApiError(
            "invalid_response", response.status, "the experiment revision differs from the ETag"
        )
    spec = experiment.get("spec")
    digest = experiment.get("spec_digest")
    if isinstance(spec, Mapping) and isinstance(digest, str):
        actual = spec_digest(spec)
        if actual != digest:
            raise ApiError(
                "experiment_spec_digest_mismatch",
                response.status,
                "the frozen spec does not hash to its spec_digest",
                {"expected": digest, "actual": actual},
            )
    return experiment


def _path(experiment_id: str, *rest: str) -> str:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    return "/".join(("/v1/experiments", segment(experiment_id), *rest))


def _revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("expected_revision must be a positive integer")
    return value


class ExperimentsResource:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _require_cloud(self, operation: str) -> None:
        if not self._client.is_cloud:
            raise _cloud_required(operation)

    def _create(self, draft: Mapping[str, Any]) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.create")
        response = yield Call("POST", "/v1/experiments", dict(draft))
        return _verified(response)

    def _update(
        self, experiment_id: str, draft: Mapping[str, Any], expected_revision: int
    ) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.update")
        response = yield Call(
            "PUT", _path(experiment_id), dict(draft), if_match=_revision(expected_revision)
        )
        return _verified(response)

    def _preflight(self, experiment_id: str, expected_revision: int) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.preflight")
        response = yield Call(
            "POST",
            _path(experiment_id, "preflight"),
            {},
            if_match=_revision(expected_revision),
        )
        preflight = _member(response, "preflight")
        digest = preflight.get("spec_digest")
        if digest is not None and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ApiError(
                "invalid_response", response.status, "the preflight spec_digest is malformed"
            )
        return preflight

    def _launch(
        self,
        experiment_id: str,
        expected_revision: int,
        spec_digest_value: str,
        authorization: Mapping[str, Any],
        idempotency_key: str,
    ) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.launch")
        if not isinstance(spec_digest_value, str) or not _SHA256.fullmatch(spec_digest_value):
            raise ValueError("spec_digest must be the sha256 digest returned by the preflight")
        if not isinstance(idempotency_key, str) or not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
            raise ValueError("idempotency_key must match [A-Za-z0-9._:-]{1,128}")
        body = {
            "idempotency_key": idempotency_key,
            "expected_revision": _revision(expected_revision),
            "spec_digest": spec_digest_value,
            "authorization": dict(authorization),
        }
        response = yield Call("POST", _path(experiment_id, "launch"), body, retry=True)
        _verified(response)
        return dict(response.body)

    def _cancel(self, experiment_id: str, reason: str) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.cancel")
        response = yield Call("POST", _path(experiment_id, "cancel"), {"reason": reason})
        return dict(response.body)

    def _get(self, experiment_id: str) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.get")
        response = yield Call("GET", _path(experiment_id), retry=True)
        return _verified(response)

    def _results(self, experiment_id: str) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.results")
        response = yield Call("GET", _path(experiment_id, "results"), retry=True)
        return dict(response.body)

    def _events(self, experiment_id: str, after: int, limit: Optional[int]) -> Flow[dict[str, Any]]:
        self._require_cloud("experiments.events")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a non-negative integer")
        params = {"after": str(after)}
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
                raise ValueError("limit must be between 1 and 500")
            params["limit"] = str(limit)
        response = yield Call("GET", _path(experiment_id, "events"), params=params, retry=True)
        events = response.body.get("events")
        if not isinstance(events, list):
            raise ApiError("invalid_response", response.status, "the response carries no events")
        return dict(response.body)

    def create(self, draft: Mapping[str, Any]) -> dict[str, Any]:
        return run_flow(self._client, self._create(draft))

    async def acreate(self, draft: Mapping[str, Any]) -> dict[str, Any]:
        return await arun_flow(self._client, self._create(draft))

    def update(
        self, experiment_id: str, draft: Mapping[str, Any], *, expected_revision: int
    ) -> dict[str, Any]:
        return run_flow(self._client, self._update(experiment_id, draft, expected_revision))

    async def aupdate(
        self, experiment_id: str, draft: Mapping[str, Any], *, expected_revision: int
    ) -> dict[str, Any]:
        return await arun_flow(self._client, self._update(experiment_id, draft, expected_revision))

    def preflight(self, experiment_id: str, *, expected_revision: int) -> dict[str, Any]:
        return run_flow(self._client, self._preflight(experiment_id, expected_revision))

    async def apreflight(self, experiment_id: str, *, expected_revision: int) -> dict[str, Any]:
        return await arun_flow(self._client, self._preflight(experiment_id, expected_revision))

    def launch(
        self,
        experiment_id: str,
        *,
        expected_revision: int,
        spec_digest: str,
        authorization: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        return run_flow(
            self._client,
            self._launch(
                experiment_id, expected_revision, spec_digest, authorization, idempotency_key
            ),
        )

    async def alaunch(
        self,
        experiment_id: str,
        *,
        expected_revision: int,
        spec_digest: str,
        authorization: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        return await arun_flow(
            self._client,
            self._launch(
                experiment_id, expected_revision, spec_digest, authorization, idempotency_key
            ),
        )

    def cancel(self, experiment_id: str, *, reason: str) -> dict[str, Any]:
        return run_flow(self._client, self._cancel(experiment_id, reason))

    async def acancel(self, experiment_id: str, *, reason: str) -> dict[str, Any]:
        return await arun_flow(self._client, self._cancel(experiment_id, reason))

    def get(self, experiment_id: str) -> dict[str, Any]:
        return run_flow(self._client, self._get(experiment_id))

    async def aget(self, experiment_id: str) -> dict[str, Any]:
        return await arun_flow(self._client, self._get(experiment_id))

    def results(self, experiment_id: str) -> dict[str, Any]:
        return run_flow(self._client, self._results(experiment_id))

    async def aresults(self, experiment_id: str) -> dict[str, Any]:
        return await arun_flow(self._client, self._results(experiment_id))

    def events(
        self, experiment_id: str, *, after: int = 0, limit: Optional[int] = None
    ) -> dict[str, Any]:
        return run_flow(self._client, self._events(experiment_id, after, limit))

    async def aevents(
        self, experiment_id: str, *, after: int = 0, limit: Optional[int] = None
    ) -> dict[str, Any]:
        return await arun_flow(self._client, self._events(experiment_id, after, limit))
