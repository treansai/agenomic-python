"""The Sensitive wrapper: write-only, masked everywhere, never readable through a public name."""

from __future__ import annotations

import copy
import json
import pickle
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from agenomic.vault import MASK, IssuedToken, Sensitive, SensitiveUnavailable
from agenomic.vault.sensitive import _as_sensitive, _unseal

CANARY = "canary-secret-7f3a9c1d-do-not-leak"


@pytest.mark.parametrize("length", [1, 8, 64, 5000])
def test_every_rendering_is_the_same_constant_mask_whatever_the_length(length: int) -> None:
    value = Sensitive("x" * length)
    assert repr(value) == "Sensitive('**********')"
    assert str(value) == MASK
    assert f"{value}" == MASK
    assert f"{value:>80}" == MASK
    assert "%s" % value == MASK  # noqa: UP031
    assert f"{value}" == MASK
    assert json.dumps({"v": value}, default=str) == json.dumps({"v": MASK})
    assert json.dumps({"v": value}, default=repr).count("x") == 0


def test_json_cannot_serialize_a_sensitive_by_default() -> None:
    with pytest.raises(TypeError):
        json.dumps({"v": Sensitive(CANARY)})


def test_pickle_carries_no_value_and_unpickles_to_an_unusable_placeholder() -> None:
    blob = pickle.dumps(Sensitive(CANARY))
    assert CANARY.encode() not in blob
    restored = pickle.loads(blob)
    assert str(restored) == MASK
    with pytest.raises(SensitiveUnavailable):
        _unseal(restored)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_is_masked_on_every_protocol(protocol: int) -> None:
    assert CANARY.encode() not in pickle.dumps(Sensitive(CANARY), protocol=protocol)


def test_copy_and_deepcopy_return_the_same_masked_object() -> None:
    value = Sensitive(CANARY)
    assert copy.copy(value) is value
    assert copy.deepcopy({"k": [value]})["k"][0] is value
    assert CANARY not in repr(copy.deepcopy({"k": value}))


def test_there_is_no_instance_dict_and_no_mutation() -> None:
    value = Sensitive(CANARY)
    with pytest.raises(TypeError):
        vars(value)
    with pytest.raises(AttributeError):
        value.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del value._held


def test_only_a_str_can_be_wrapped_and_the_error_never_echoes_the_value() -> None:
    with pytest.raises(TypeError) as excinfo:
        Sensitive(CANARY.encode())  # type: ignore[arg-type]
    assert CANARY not in str(excinfo.value)


def test_equality_and_hash_do_not_compare_values() -> None:
    assert Sensitive(CANARY) != Sensitive(CANARY)
    assert len({Sensitive(CANARY), Sensitive(CANARY)}) == 2


class Holder(BaseModel):
    value: Sensitive


def test_pydantic_field_masks_in_repr_dump_json_and_schema() -> None:
    holder = Holder(value=CANARY)  # type: ignore[arg-type]
    assert CANARY not in repr(holder)
    assert CANARY not in str(holder)
    assert holder.model_dump() == {"value": MASK}
    assert holder.model_dump(mode="json") == {"value": MASK}
    assert holder.model_dump_json() == json.dumps({"value": MASK}, separators=(",", ":"))
    assert Holder.model_json_schema()["properties"]["value"]["writeOnly"] is True
    assert CANARY.encode() not in pickle.dumps(holder)
    assert CANARY not in json.dumps(holder.model_dump())


def test_pydantic_rejects_a_non_string() -> None:
    with pytest.raises(ValueError):
        Holder(value=12)  # type: ignore[arg-type]


def test_from_env_wraps_without_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGM_TEST_SECRET", CANARY)
    value = Sensitive.from_env("AGM_TEST_SECRET")
    assert CANARY not in repr(value)
    assert _unseal(value) == CANARY
    monkeypatch.delenv("AGM_TEST_SECRET")
    with pytest.raises(SensitiveUnavailable) as excinfo:
        Sensitive.from_env("AGM_TEST_SECRET")
    assert "AGM_TEST_SECRET" in str(excinfo.value)


def test_a_plain_string_is_refused_where_a_secret_value_is_expected() -> None:
    with pytest.raises(TypeError) as excinfo:
        _as_sensitive(CANARY)  # type: ignore[arg-type]
    assert CANARY not in str(excinfo.value)
    assert "Sensitive" in str(excinfo.value)


def test_a_pydantic_secret_str_is_accepted_and_rewrapped() -> None:
    wrapped = _as_sensitive(SecretStr(CANARY))
    assert isinstance(wrapped, Sensitive)
    assert CANARY not in repr(wrapped)


def test_issued_token_is_shown_once() -> None:
    token = IssuedToken("vrt_" + CANARY)
    assert CANARY not in repr(token)
    assert token.consume() == "vrt_" + CANARY
    with pytest.raises(SensitiveUnavailable):
        token.consume()
    assert CANARY not in repr(token)


def test_issued_token_pickle_keeps_the_type_and_drops_the_value() -> None:
    blob: Any = pickle.dumps(IssuedToken("vrt_" + CANARY))
    assert CANARY.encode() not in blob
    restored = pickle.loads(blob)
    assert isinstance(restored, IssuedToken)
    with pytest.raises(SensitiveUnavailable):
        restored.consume()


def test_pydantic_accepts_an_existing_sensitive_without_rewrapping() -> None:
    original = Sensitive(CANARY)
    assert Holder(value=original).value is original
