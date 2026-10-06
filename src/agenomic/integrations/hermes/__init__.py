"""Agenomic adapter for Hermes Agent (``agenomic-hermes-adapter``).

Hermes stays the runtime, Agenomic stays the control plane. The adapter is a
Hermes plugin (entry point ``hermes_agent.plugins: agenomic``) that reports
the instance, sessions, model calls and tool calls to Agenomic and, in
shadow or enforce, asks the Agenomic Execution Gateway before covered tool
actions run. It runs inside the agent's process and is cooperative: it is not
a security boundary. See ``docs/hermes.md``.

Importing this package does not import Hermes; the submodules that need
Hermes import it lazily.

Example:
    >>> from agenomic.integrations.hermes import ADAPTER_VERSION, COMPATIBLE_HERMES
    >>> ADAPTER_VERSION, sorted(COMPATIBLE_HERMES)
    ('1.0.0', ['0.21.5'])
"""

from __future__ import annotations

ADAPTER_NAME = "agenomic-hermes-adapter"
ADAPTER_VERSION = "1.0.0"
#: ``hermes_cli.__version__`` values the adapter was verified against
#: (tag v2026.9.24, commit f97608f178d1ffeca59860195ab7da295f7c8e5f).
COMPATIBLE_HERMES = frozenset({"0.21.5"})
PINNED_HERMES_COMMIT = "f97608f178d1ffeca59860195ab7da295f7c8e5f"

__all__ = ["ADAPTER_NAME", "ADAPTER_VERSION", "COMPATIBLE_HERMES", "PINNED_HERMES_COMMIT"]
