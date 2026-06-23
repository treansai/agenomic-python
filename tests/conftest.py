"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def v03_errors() -> Callable[[dict[str, Any]], list[str]]:
    """Return a validator: ``v03_errors(trace)`` → list of schema errors (empty
    when the trace conforms to the vendored ``agenomic/v0.3`` JSON Schema)."""
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    spec = Path(__file__).parent / "schemas" / "v0.3"
    schema = json.loads((spec / "trace-event.schema.json").read_text())
    registry_schema = json.loads((spec / "event-type-registry.json").read_text())
    registry = Registry().with_resource(
        registry_schema["$id"], Resource.from_contents(registry_schema)
    )
    validator = Draft202012Validator(
        schema, registry=registry, format_checker=FormatChecker()
    )

    def _errors(trace: dict[str, Any]) -> list[str]:
        return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(trace)]

    return _errors
