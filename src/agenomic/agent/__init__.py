"""Local agent genome loading and model configuration.

This module provides a minimal, provider-agnostic way to attach a model
configuration to an agent's ``genome.yaml`` runtime block:

    >>> client.agent.load("./my-agent")  # doctest: +SKIP
    >>> agent.configure_model(
    ...     provider="huggingface",
    ...     model="mistralai/Mistral-7B-Instruct-v0.3",
    ...     task="text-generation",
    ... )  # doctest: +SKIP

It reads/writes a local genome file (``genome.yaml`` by default, ``genome.json``
also supported). YAML is parsed with PyYAML when available and otherwise with a
small built-in parser that handles the flat/nested mapping shapes used by
genomes. Any provider string is accepted; ``huggingface`` aliases are validated
and normalized.
"""

from agenomic.agent.genome import Agent, AgentResource, GenomeError, load_agent

__all__ = ["Agent", "AgentResource", "GenomeError", "load_agent"]
