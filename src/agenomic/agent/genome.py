"""Minimal local genome loader + ``configure_model`` helper.

Provider-agnostic: ``configure_model`` accepts any provider string and writes a
``runtime`` block into the genome. When the provider is a Hugging Face alias it
is normalized to ``huggingface`` and basic validation is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agenomic.providers.huggingface import is_huggingface, normalize_provider


class GenomeError(Exception):
    """Raised when a genome file cannot be located, parsed, or updated."""


# Filenames tried (in order) when a directory is passed to ``load``.
_GENOME_FILENAMES = ("genome.yaml", "genome.yml", "genome.json")


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _minimal_yaml_load(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise GenomeError("genome root must be a mapping")
    return data


def _dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        return _minimal_yaml_dump(data)
    out: str = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return out


# --- tiny YAML fallback (sufficient for genome mappings) -------------------
#
# Supports nested mappings (2-space indent), scalar values, and inline empty
# mappings. Lists and block scalars are intentionally unsupported; genomes we
# manage here are mapping-only. PyYAML is used whenever it is installed.


def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
    if s == "" or s in ("null", "~"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    # stack of (indent, mapping)
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise GenomeError(f"genome line {lineno}: expected 'key: value'")
        key, _, value = line.strip().partition(":")
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise GenomeError(f"genome line {lineno}: bad indentation")
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _minimal_yaml_dump(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_minimal_yaml_dump(value, indent + 1).rstrip("\n"))
        else:
            lines.append(f"{pad}{key}: {_scalar_repr(value)}")
    return "\n".join(lines) + "\n"


def _scalar_repr(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    if s == "" or any(c in s for c in ":#") or s.strip() != s:
        return json.dumps(s)
    return s


@dataclass
class Agent:
    """A loaded local agent genome backed by a file on disk."""

    path: Path
    data: dict[str, Any]
    is_json: bool = False

    @property
    def runtime(self) -> dict[str, Any]:
        block = self.data.get("runtime")
        if not isinstance(block, dict):
            return {}
        return block

    def configure_model(
        self,
        *,
        provider: str,
        model: str,
        task: Optional[str] = None,
        revision: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Write a model configuration into the genome's ``runtime`` block.

        Provider-agnostic: any non-empty provider string is accepted. Hugging
        Face aliases (``hf``, ``hugging-face``, ...) are normalized to
        ``huggingface`` and validated. Returns the written model block.
        """
        if not provider or not provider.strip():
            raise GenomeError("provider is required")
        if not model or not model.strip():
            raise GenomeError("model is required")

        canonical = provider.strip()
        if is_huggingface(provider):
            normalized = normalize_provider(provider)
            assert normalized is not None  # is_huggingface guarantees this
            canonical = normalized

        model_block: dict[str, Any] = {"provider": canonical, "model": model}
        if task is not None:
            model_block["task"] = task
        if revision is not None:
            model_block["revision"] = revision
        if parameters:
            model_block["parameters"] = dict(parameters)

        runtime = self.data.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            raise GenomeError("genome 'runtime' block must be a mapping")
        runtime["model"] = model_block

        if save:
            self.save()
        return model_block

    def save(self) -> None:
        """Persist the genome back to its source file."""
        if self.is_json:
            self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        else:
            self.path.write_text(_dump_yaml(self.data), encoding="utf-8")


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in _GENOME_FILENAMES:
            candidate = p / name
            if candidate.exists():
                return candidate
        # Default to genome.yaml for a fresh directory.
        return p / "genome.yaml"
    return p


def load_agent(path: str | Path) -> Agent:
    """Load (or initialize) an agent genome at ``path``.

    ``path`` may be a directory (``genome.yaml`` / ``genome.json`` is located
    inside it) or a direct file path. A missing file yields an empty genome that
    is created on first ``save``.
    """
    resolved = _resolve_path(path)
    is_json = resolved.suffix == ".json"
    if resolved.exists():
        text = resolved.read_text(encoding="utf-8")
        if is_json:
            data = json.loads(text) if text.strip() else {}
            if not isinstance(data, dict):
                raise GenomeError("genome root must be a mapping")
        else:
            data = _load_yaml(text)
    else:
        data = {}
    return Agent(path=resolved, data=data, is_json=is_json)


@dataclass
class AgentResource:
    """``client.agent`` namespace: load local agent genomes."""

    client: Any = field(default=None)

    def load(self, path: str | Path) -> Agent:
        """Load an agent genome from a local directory or file path."""
        return load_agent(path)
