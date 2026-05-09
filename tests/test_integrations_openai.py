"""Tests for the OpenAI integration. Uses a mock client so `openai` is not required."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agenomic.integrations.openai import instrument_openai
from agenomic.trace.context import set_current_recorder
from agenomic.trace.recorder import TraceRecorder


def _fake_client() -> Any:
    create = MagicMock(
        return_value=SimpleNamespace(
            id="resp_1", system_fingerprint="fp_abc", choices=[], usage=None
        )
    )
    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def test_module_imports_without_openai() -> None:
    # Importing the module is allowed even if openai is missing.
    import agenomic.integrations.openai as mod  # noqa: F401


def test_instrument_records_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the openai import inside instrument_openai to succeed even if
    # the package is not installed.
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    rec = TraceRecorder("agent://a/b", "r", "t")
    set_current_recorder(rec)
    try:
        client = instrument_openai(_fake_client())
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    finally:
        set_current_recorder(None)
    assert len(rec.model_calls) == 1
    call = rec.model_calls[0]
    assert call.provider == "openai"
    assert call.model == "gpt-4o-mini"
    assert call.fingerprint == "fp_abc"
    assert call.prompt_hash
    assert call.output_hash


def test_instrument_passes_through_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    client = instrument_openai(_fake_client())
    response = client.chat.completions.create(model="m", messages=[])
    assert response.id == "resp_1"


def test_instrument_raises_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="agenomic\\[openai\\]"):
        instrument_openai(_fake_client())
