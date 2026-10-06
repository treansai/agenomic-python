from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agenomic.integrations.hermes.canonical import (
    CanonicalError,
    arguments_hash,
    canonical_json,
    schema_hash,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "hermes_canonical_vectors.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["input"][:40])
def test_matches_gateway_vectors(vector: dict[str, str]) -> None:
    value = json.loads(vector["input"])
    assert canonical_json(value) == vector["canonical"]
    assert arguments_hash(value) == vector["hash"]


def test_vector_file_is_from_the_gateway_rule() -> None:
    assert VECTORS["rule"] == "agenomic.canon/v1"
    assert len(VECTORS["vectors"]) >= 20


def test_key_order_does_not_matter_and_types_do() -> None:
    assert arguments_hash({"a": 1, "b": 2}) == arguments_hash({"b": 2, "a": 1})
    assert arguments_hash({"n": 1}) != arguments_hash({"n": "1"})
    assert arguments_hash({"n": 1}) != arguments_hash({"n": 1.0})
    assert arguments_hash({"n": None}) != arguments_hash({})
    assert canonical_json((1, 2)) == "[1,2]"
    assert schema_hash({"name": "t"}) == arguments_hash({"name": "t"})


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        {"k": float("-inf")},
        10**400,
        {"k": -(10**400)},
        "\ud800",
        {"\udc00": 1},
        {1: "x"},
        b"raw",
        object(),
    ],
)
def test_values_the_gateway_cannot_receive(value: Any) -> None:
    with pytest.raises(CanonicalError):
        canonical_json(value)
