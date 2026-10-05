from __future__ import annotations

import pytest

from agenomic.prompts import (
    PromptAliasRef,
    PromptRefError,
    PromptUri,
    PromptVersionRef,
    parse_execution_ref,
    parse_prompt_ref,
)

WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"
OTHER = "5a5a5a5a-1111-4222-8333-444455556666"


def test_round_trip_for_every_form() -> None:
    for ref in (
        PromptVersionRef("prm_planner", 7),
        PromptAliasRef("prm_planner", "staging"),
        PromptUri(WORKSPACE, "prm_planner", 7),
    ):
        assert parse_execution_ref(str(ref), workspace_id=WORKSPACE) == ref
    assert parse_prompt_ref("prm_planner", allow_bare_id=True) == "prm_planner"


def test_cross_workspace_uri_refused_before_any_request() -> None:
    uri = str(PromptUri(OTHER, "prm_planner", 7))
    with pytest.raises(PromptRefError) as raised:
        parse_execution_ref(uri, workspace_id=WORKSPACE)
    assert raised.value.code == "prompt_ref_cross_workspace"
    assert raised.value.status == 0
    assert OTHER not in str(raised.value)
    parsed = parse_execution_ref(uri)
    assert isinstance(parsed, PromptUri)
    with pytest.raises(PromptRefError):
        parsed.to_version_ref(WORKSPACE)
    assert parsed.to_version_ref(OTHER) == PromptVersionRef("prm_planner", 7)


def test_bare_id_is_unversioned_in_execution() -> None:
    with pytest.raises(PromptRefError) as raised:
        parse_execution_ref("prm_planner")
    assert raised.value.code == "prompt_ref_unversioned"
    with pytest.raises(PromptRefError) as management:
        parse_prompt_ref("prm_planner")
    assert management.value.code == "prompt_ref_unversioned"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("prm_x:1_0", "invalid_version"),
        ("prm_x:\u0661", "invalid_character"),
        ("prm_x:" + "9" * 11, "invalid_version"),
    ],
)
def test_python_int_quirks_are_refused(text: str, reason: str) -> None:
    with pytest.raises(PromptRefError) as raised:
        parse_execution_ref(text)
    assert raised.value.reason == reason
