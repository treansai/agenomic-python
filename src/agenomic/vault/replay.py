"""Replay of vault executions from fixtures.

Replay is mock by default: with a :class:`VaultReplay` attached to the client,
``client.tools.execute`` is answered from fixtures and never reaches the
network. A call no fixture matches raises :class:`ReplayFixtureMissing`; there
is no switch that turns a miss into a live call. Fixtures hold business
results only: a vault execution never returns a credential, so there is none
to record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agenomic.canonical import content_hash
from agenomic.vault.errors import (
    ReplayFixtureMissing,
    VaultApprovalRequired,
    VaultConflict,
    VaultNotFound,
    VaultPolicyDenied,
    VaultValidationError,
)
from agenomic.vault.execution import ExecuteRequest, outcome_of
from agenomic.vault.models import ExecuteResult, ExecutionStatus

REPLAY_SCHEMA_VERSION = "agenomic.vault_replay/v1"

Key = tuple[str, str, str]


class ReplayOutcome(BaseModel):
    """What a fixture answers: a success, or any of the states a real execution can end in.

    Example:
        >>> ReplayOutcome(kind="outcome_unknown", status_code=504).kind
        'outcome_unknown'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["succeeded", "failed", "outcome_unknown", "denied", "approval_required"] = (
        "succeeded"
    )
    result: JsonValue = None
    status_code: Optional[int] = None
    receipt_id: Optional[str] = None
    error_class: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    approval_id: Optional[str] = None


class ReplayFixture(BaseModel):
    """One recorded or authored answer for ``tool`` on ``binding`` with exactly these arguments.

    Example:
        >>> fixture = ReplayFixture(fixture_id="fx-1", tool="crm.get_customer", binding="crm-read",
        ...     arguments={"id": "c_1"}, outcome=ReplayOutcome(result={"name": "Ada"}))
        >>> fixture.arguments_hash.startswith("blake3:")
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str
    tool: str
    binding: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    outcome: ReplayOutcome = Field(default_factory=ReplayOutcome)

    @property
    def arguments_hash(self) -> str:
        return content_hash(self.arguments)

    @property
    def key(self) -> Key:
        return (self.tool, self.binding, self.arguments_hash)


@dataclass(frozen=True)
class ReplayCall:
    """One call served from a fixture, kept in :attr:`VaultReplay.calls` for assertions."""

    action_id: str
    tool: str
    binding: str
    fixture_id: str
    repeated: bool = False


@dataclass(frozen=True)
class _Served:
    key: Key
    fixture: ReplayFixture
    status: ExecutionStatus


def _state_status(fixture: ReplayFixture, action_id: str) -> ExecutionStatus:
    out = fixture.outcome
    return ExecutionStatus(
        status="finished",
        action_id=action_id,
        state="refused" if out.kind == "denied" else out.kind,
        receipt_id=out.receipt_id,
        result=out.result,
        status_code=out.status_code,
        error_class=",".join(out.reason_codes) if out.kind == "denied" else out.error_class,
        replayed=True,
    )


class VaultReplay:
    """Fixture set that answers vault executions offline.

    Example:
        >>> from agenomic import Client
        >>> replay = VaultReplay([ReplayFixture(fixture_id="fx-1", tool="crm.get_customer",
        ...     binding="crm-read", arguments={"id": "c_1"},
        ...     outcome=ReplayOutcome(result={"name": "Ada"}))])
        >>> client = Client(vault_replay=replay)
        >>> out = client.tools.execute(tool="crm.get_customer", binding="crm-read",
        ...     arguments={"id": "c_1"})
        >>> (out.result, out.replayed)
        ({'name': 'Ada'}, True)
    """

    def __init__(self, fixtures: Iterable[ReplayFixture] = ()) -> None:
        self._fixtures: dict[Key, ReplayFixture] = {}
        self._served: dict[str, _Served] = {}
        #: Every call served, in order. A call no fixture matched is not in it: it raised.
        self.calls: list[ReplayCall] = []
        for fixture in fixtures:
            self.add(fixture)

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> VaultReplay:
        """Build from ``{"schema_version": ..., "fixtures": [...]}``; a malformed document is refused."""
        if document.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise VaultValidationError(
                "invalid_fixtures", f"schema_version must be {REPLAY_SCHEMA_VERSION}", 0
            )
        raw = document.get("fixtures")
        try:
            if not isinstance(raw, list):
                raise TypeError("fixtures must be a list")
            return cls(ReplayFixture.model_validate(item) for item in raw)
        except (TypeError, ValidationError) as error:
            detail = type(error).__name__
        raise VaultValidationError(
            "invalid_fixtures", f"the fixture set is malformed ({detail})", 0
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> VaultReplay:
        """Load a JSON fixture file."""
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = None
        if not isinstance(document, dict):
            raise VaultValidationError(
                "invalid_fixtures", "the fixture file is not a JSON object", 0
            )
        return cls.from_mapping(document)

    def add(self, fixture: ReplayFixture, *, replace: bool = False) -> None:
        """Register a fixture; a second one for the same call is refused unless ``replace``."""
        if fixture.key in self._fixtures and not replace:
            raise VaultValidationError(
                "duplicate_fixture",
                f"a fixture already answers {fixture.tool!r} on {fixture.binding!r}",
                0,
            )
        self._fixtures[fixture.key] = fixture

    def to_mapping(self) -> dict[str, object]:
        """The fixture set as a JSON-ready document."""
        fixtures = [f.model_dump(mode="json") for f in self._fixtures.values()]
        return {"schema_version": REPLAY_SCHEMA_VERSION, "fixtures": fixtures}

    def save(self, path: Union[str, Path]) -> None:
        """Write the fixture set as JSON."""
        Path(path).write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def execute(self, request: ExecuteRequest) -> ExecuteResult:
        """Answer an execution from its fixture, or raise :class:`ReplayFixtureMissing`."""
        key = (request.tool, request.binding, content_hash(request.arguments))
        prior = self._served.get(request.action_id)
        if prior is not None:
            return self._again(request, key, prior)
        fixture = self._fixtures.get(key)
        if fixture is None:
            raise ReplayFixtureMissing(*key)
        return self._serve(request, key, fixture)

    def status(self, action_id: str) -> ExecutionStatus:
        """The stored outcome of an action already served; an unknown action is not found."""
        served = self._served.get(action_id)
        if served is None:
            raise VaultNotFound("not_found", "no replayed execution has this action_id", 404)
        return served.status

    def _again(self, request: ExecuteRequest, key: Key, prior: _Served) -> ExecuteResult:
        if prior.key != key:
            raise VaultConflict(
                "conflict", "action_id was already used for a different request", 409
            )
        self._log(request, prior.fixture, repeated=True)
        return outcome_of(prior.status, request.action_id)

    def _serve(self, request: ExecuteRequest, key: Key, fixture: ReplayFixture) -> ExecuteResult:
        out = fixture.outcome
        if out.kind == "approval_required":
            raise VaultApprovalRequired(
                request.action_id, out.approval_id or f"replay-approval-{fixture.fixture_id}"
            )
        status = _state_status(fixture, request.action_id)
        self._served[request.action_id] = _Served(key, fixture, status)
        self._log(request, fixture, repeated=False)
        if out.kind == "denied":
            raise VaultPolicyDenied(request.action_id, out.reason_codes, out.explanation)
        return outcome_of(status, request.action_id)

    def _log(self, request: ExecuteRequest, fixture: ReplayFixture, *, repeated: bool) -> None:
        self.calls.append(
            ReplayCall(
                request.action_id, request.tool, request.binding, fixture.fixture_id, repeated
            )
        )
