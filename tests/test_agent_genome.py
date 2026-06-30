"""Tests for local genome loading and ``configure_model``."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenomic import Client
from agenomic.agent import GenomeError, load_agent


def test_configure_model_writes_huggingface_into_genome(tmp_path: Path) -> None:
    genome = tmp_path / "genome.yaml"
    genome.write_text("id: agent://acme/demo\nname: demo\n", encoding="utf-8")

    client = Client()
    agent = client.agent.load(str(tmp_path))
    block = agent.configure_model(
        provider="huggingface",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
    )
    assert block["provider"] == "huggingface"

    # Reload to confirm it persisted.
    reloaded = load_agent(str(tmp_path))
    model = reloaded.runtime["model"]
    assert model["provider"] == "huggingface"
    assert model["model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert model["task"] == "text-generation"
    # Original keys are preserved.
    assert reloaded.data["id"] == "agent://acme/demo"


def test_configure_model_normalizes_hf_alias(tmp_path: Path) -> None:
    agent = load_agent(str(tmp_path / "genome.yaml"))
    block = agent.configure_model(provider="Hugging-Face", model="gpt2", save=False)
    assert block["provider"] == "huggingface"


def test_configure_model_is_provider_agnostic(tmp_path: Path) -> None:
    agent = load_agent(str(tmp_path / "genome.yaml"))
    block = agent.configure_model(provider="my-custom-provider", model="x", save=False)
    assert block["provider"] == "my-custom-provider"


def test_configure_model_requires_provider_and_model(tmp_path: Path) -> None:
    agent = load_agent(str(tmp_path / "genome.yaml"))
    with pytest.raises(GenomeError):
        agent.configure_model(provider="", model="x", save=False)
    with pytest.raises(GenomeError):
        agent.configure_model(provider="huggingface", model="", save=False)


def test_load_creates_genome_on_save(tmp_path: Path) -> None:
    agent = load_agent(str(tmp_path))
    agent.configure_model(provider="huggingface", model="gpt2", task="text-generation")
    assert (tmp_path / "genome.yaml").exists()


def test_configure_model_json_genome(tmp_path: Path) -> None:
    genome = tmp_path / "genome.json"
    genome.write_text('{"id": "agent://acme/demo"}', encoding="utf-8")
    agent = load_agent(str(genome))
    agent.configure_model(provider="hf", model="gpt2", task="text-generation")
    reloaded = load_agent(str(genome))
    assert reloaded.runtime["model"]["provider"] == "huggingface"


def test_configure_model_with_parameters_and_revision(tmp_path: Path) -> None:
    agent = load_agent(str(tmp_path / "genome.yaml"))
    block = agent.configure_model(
        provider="huggingface",
        model="gpt2",
        task="text-generation",
        revision="main",
        parameters={"temperature": 0.7, "max_new_tokens": 64},
    )
    assert block["revision"] == "main"
    assert block["parameters"]["temperature"] == 0.7
    reloaded = load_agent(str(tmp_path / "genome.yaml"))
    assert reloaded.runtime["model"]["parameters"]["max_new_tokens"] == 64
