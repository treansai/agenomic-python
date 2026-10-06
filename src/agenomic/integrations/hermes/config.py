"""Adapter configuration ``agenomic.hermes.adapter_config/v1``.

The adapter reads its settings from the Hermes plugin settings
(``plugins.entries.agenomic.settings`` in ``$HERMES_HOME/config.yaml``, read
through ``ctx.get_config``) and/or from a YAML or JSON file named by
``AGENOMIC_HERMES_CONFIG``; keys from the file win. Every key is distinct from
the native Hermes keys.

The runtime credential is never written in a file: ``runtime_token`` is an
``${env:VAR}`` reference resolved against ``os.environ`` (default
``${env:AGENOMIC_HERMES_RUNTIME_TOKEN}``). Hermes expands ``${...}`` in plugin
settings before the adapter reads them, so a ``runtime_token`` that arrives
from the plugin settings is a literal and is refused; keep the default or put
the reference in the ``AGENOMIC_HERMES_CONFIG`` file.

Example:
    >>> cfg = AdapterConfig.model_validate({"endpoint": "https://agenomic.example"})
    >>> cfg.runtime_token
    '${env:AGENOMIC_HERMES_RUNTIME_TOKEN}'
    >>> cfg.timeouts.decision_s
    5.0
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    field_validator,
)

from agenomic.exceptions import AgenomicError

logger = logging.getLogger("agenomic.integrations.hermes.config")

CONFIG_SCHEMA = "agenomic.hermes.adapter_config/v1"
CONFIG_ENV = "AGENOMIC_HERMES_CONFIG"
DEFAULT_TOKEN_ENV = "AGENOMIC_HERMES_RUNTIME_TOKEN"
SUPERVISOR_TOKEN_ENV = "AGENOMIC_HERMES_SUPERVISOR_TOKEN"
#: Prefix of every Agenomic runtime token (the credential the Hermes plugin presents).
RUNTIME_TOKEN_PREFIX = "agmhr_"
#: Prefix of every Agenomic supervisor token (the credential the supervisor presents).
SUPERVISOR_TOKEN_PREFIX = "agmhs_"
_SECRET_NAME = re.compile(
    r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_ACCESS_KEY|_PRIVATE_KEY)$|^API_KEY$", re.I
)
_PROVIDER_KEYS = re.compile(
    r"_API_KEY$|^API_KEY$|^AWS_SECRET_ACCESS_KEY$|^GOOGLE_APPLICATION_CREDENTIALS$", re.I
)
PLUGIN_ID = "agenomic"
GUARD_COMMAND = "agenomic-hermes-guard"
MODEL_GATEWAY_PATH = "/v1/hermes/runtime/model/v1"

_ENV_REF = re.compile(r"^\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
_ANY_ENV_REF = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_SETTINGS_KEYS = (
    "schema_version",
    "endpoint",
    "runtime_token",
    "mode_hint",
    "timeouts",
    "buffer",
    "capture",
    "fail_mode",
)


class ConfigError(AgenomicError):
    """The adapter configuration is missing, malformed or references an unset variable.

    Messages name keys and variables, never values.
    """


class Timeouts(BaseModel):
    """HTTP budgets in seconds."""

    model_config = ConfigDict(extra="forbid")

    connect_s: float = Field(default=3.0, gt=0, le=60)
    decision_s: float = Field(default=5.0, gt=0, le=60)
    report_s: float = Field(default=10.0, gt=0, le=120)


class BufferConfig(BaseModel):
    """Event exporter bounds. ``spool_path`` enables a bounded JSONL spool on disk."""

    model_config = ConfigDict(extra="forbid")

    max_events: int = Field(default=10_000, ge=1, le=1_000_000)
    max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    flush_interval_s: float = Field(default=1.0, gt=0, le=300)
    batch_size: int = Field(default=500, ge=1, le=500)
    spool_path: Optional[str] = None
    spool_max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)


class CaptureConfig(BaseModel):
    """What content leaves the process: hashes only, or redacted, truncated previews."""

    model_config = ConfigDict(extra="forbid")

    content: Literal["metadata", "redacted_preview"] = "metadata"
    preview_chars: int = Field(default=200, ge=1, le=4000)


class AdapterConfig(BaseModel):
    """``agenomic.hermes.adapter_config/v1``.

    ``mode_hint`` is informational: the server decides the mode. ``fail_mode``
    has one value, ``closed``: in enforce, no valid decision means the action
    is blocked.

    Example:
        >>> AdapterConfig.model_validate({"endpoint": "https://a.example", "mode_hint": "shadow"}).mode_hint
        'shadow'
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agenomic.hermes.adapter_config/v1"] = (
        "agenomic.hermes.adapter_config/v1"
    )
    endpoint: str
    runtime_token: str = f"${{env:{DEFAULT_TOKEN_ENV}}}"
    mode_hint: Optional[Literal["observe", "shadow", "enforce"]] = None
    timeouts: Timeouts = Field(default_factory=Timeouts)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    fail_mode: Literal["closed"] = "closed"

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("https", "http") or not parts.hostname:
            raise ValueError("endpoint must be an absolute http(s) URL")
        if parts.query or parts.fragment:
            raise ValueError("endpoint must not carry a query or fragment")
        if parts.scheme == "http" and parts.hostname not in _LOOPBACK:
            logger.warning(
                "Agenomic endpoint uses plain http outside loopback (%s)", parts.hostname
            )
        return value.rstrip("/")

    @field_validator("runtime_token")
    @classmethod
    def _check_token_ref(cls, value: str) -> str:
        if not _ENV_REF.match(value):
            raise ValueError(
                "runtime_token must be an ${env:VAR} reference; literal tokens are refused "
                "(Hermes expands ${...} in plugin settings, so set the reference in the "
                f"{CONFIG_ENV} file or keep the default)"
            )
        name = _ENV_REF.match(value)
        assert name is not None
        problem = runtime_token_env_problem(name.group(1))
        if problem is not None:
            # RuntimeClient would send this variable's value as a Bearer token to Agenomic.
            raise ValueError(f"runtime_token: {problem}")
        return value

    @property
    def token_env(self) -> str:
        """Name of the environment variable holding the runtime token.

        Example:
            >>> AdapterConfig.model_validate({"endpoint": "https://a.example"}).token_env
            'AGENOMIC_HERMES_RUNTIME_TOKEN'
        """
        match = _ENV_REF.match(self.runtime_token)
        assert match is not None
        return match.group(1)

    def resolve_token(self, environ: Optional[Mapping[str, str]] = None) -> SecretStr:
        """Read the runtime token from the environment.

        Raises :class:`ConfigError` naming the variable (never its value) when
        it is unset or empty, or when its value is not an Agenomic runtime
        token (``agmhr_...``): another credential is never sent to Agenomic.

        Example:
            >>> cfg = AdapterConfig.model_validate({"endpoint": "https://a.example"})
            >>> cfg.resolve_token({"AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_x"}).get_secret_value()
            'agmhr_x'
            >>> cfg.resolve_token({"AGENOMIC_HERMES_RUNTIME_TOKEN": "sk-x"})
            Traceback (most recent call last):
            ...
            agenomic.integrations.hermes.config.ConfigError: AGENOMIC_HERMES_RUNTIME_TOKEN does not hold an Agenomic runtime token (agmhr_...)
        """
        env = os.environ if environ is None else environ
        value = env.get(self.token_env)
        if not value:
            raise ConfigError(
                f"runtime_token references environment variable {self.token_env}, which is not set"
            )
        if not value.startswith(RUNTIME_TOKEN_PREFIX):
            raise ConfigError(
                f"{self.token_env} does not hold an Agenomic runtime token "
                f"({RUNTIME_TOKEN_PREFIX}...)"
            )
        return SecretStr(value)


def _resolve_env_refs(value: Any, environ: Mapping[str, str], path: str) -> Any:
    """Resolve ``${env:VAR}`` in string values except ``runtime_token`` (kept as a reference)."""
    if isinstance(value, dict):
        return {
            k: (
                v
                if path == "" and k == "runtime_token"
                else _resolve_env_refs(v, environ, f"{path}.{k}")
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_resolve_env_refs(v, environ, path) for v in value]
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in environ:
                raise ConfigError(
                    f"{path.lstrip('.')} references environment variable {name}, which is not set"
                )
            return environ[name]

        return _ANY_ENV_REF.sub(_sub, value)
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"{CONFIG_ENV} file {path} cannot be read: {e.strerror}") from e
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"{CONFIG_ENV} file {path} is not valid JSON (line {e.lineno})"
            ) from e
    else:
        try:
            import yaml
        except ImportError as e:
            raise ConfigError(
                "YAML adapter config needs PyYAML: pip install 'agenomic[hermes]' (or use a .json file)"
            ) from e
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"{CONFIG_ENV} file {path} is not valid YAML") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{CONFIG_ENV} file {path} must contain a mapping")
    return data


def _format_errors(err: ValidationError) -> str:
    # ValidationError's own str() echoes input values; only locations and messages are kept.
    parts = []
    for item in err.errors(include_input=False, include_url=False):
        loc = ".".join(str(p) for p in item.get("loc", ())) or "config"
        parts.append(f"{loc}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def build_config(
    settings: Optional[Mapping[str, object]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    resolve_token: bool = True,
) -> tuple[AdapterConfig, Optional[SecretStr]]:
    """Merge plugin settings with the ``AGENOMIC_HERMES_CONFIG`` file and validate.

    Returns the config and, when ``resolve_token`` is true, the resolved
    runtime token. Raises :class:`ConfigError` with a message that never
    contains a secret.

    Example:
        >>> cfg, token = build_config(
        ...     {"endpoint": "https://a.example"},
        ...     environ={"AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_x"},
        ... )
        >>> cfg.endpoint, token.get_secret_value()
        ('https://a.example', 'agmhr_x')
    """
    env = os.environ if environ is None else environ
    merged: dict[str, Any] = {}
    if settings:
        if "runtime_token" in settings:
            raise ConfigError(
                "runtime_token must not be set in the Hermes plugin settings: Hermes expands "
                f"${{...}} there, so the adapter would receive a literal token; set it in the "
                f"{CONFIG_ENV} file or keep the default {DEFAULT_TOKEN_ENV}"
            )
        merged = _deep_merge(merged, settings)
    file_path = env.get(CONFIG_ENV)
    if file_path:
        merged = _deep_merge(merged, _load_file(Path(file_path).expanduser()))
    if not merged.get("endpoint"):
        raise ConfigError(
            "endpoint is not configured (plugins.entries.agenomic.settings.endpoint "
            f"or the {CONFIG_ENV} file)"
        )
    resolved = _resolve_env_refs(merged, env, "")
    try:
        config = AdapterConfig.model_validate(resolved)
    except ValidationError as e:
        raise ConfigError(f"invalid adapter config: {_format_errors(e)}") from None
    token = config.resolve_token(env) if resolve_token else None
    return config, token


def settings_from_context(ctx: object) -> dict[str, object]:
    """Read the adapter keys from a Hermes ``PluginContext`` (``ctx.get_config``).

    Example:
        >>> class Ctx:
        ...     def get_config(self, key, default=None):
        ...         return {"endpoint": "https://a.example"}.get(key, default)
        >>> settings_from_context(Ctx())
        {'endpoint': 'https://a.example'}
    """
    out: dict[str, object] = {}
    getter = getattr(ctx, "get_config", None)
    if not callable(getter):
        return out
    for key in _SETTINGS_KEYS:
        try:
            value = getter(key, None)
        except (ValueError, KeyError, TypeError, OSError) as e:
            logger.debug("plugin setting %s unreadable: %s", key, type(e).__name__)
            continue
        if value is not None:
            out[key] = value
    return out


def _to_hermes_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_hermes_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_hermes_refs(v) for v in value]
    if isinstance(value, str):
        return _ANY_ENV_REF.sub(lambda m: "${" + m.group(1) + "}", value)
    return value


def runtime_token_env_problem(name: str) -> Optional[str]:
    """Why ``name`` cannot carry the runtime token into the child, or ``None``.

    The runtime token variable is the only credential the child receives, so it must
    not be the supervisor token, a provider key, or any other credential shaped name
    (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET``, ``*_PASSWORD``...) unless it is an
    ``AGENOMIC_`` name.

    Example:
        >>> runtime_token_env_problem("AGENOMIC_HERMES_RUNTIME_TOKEN") is None
        True
        >>> runtime_token_env_problem("OPENAI_API_KEY")
        'OPENAI_API_KEY is a provider credential name, not the Agenomic runtime token'
    """
    if name == SUPERVISOR_TOKEN_ENV:
        return f"{name} is the supervisor credential and is never passed to the child"
    if _PROVIDER_KEYS.search(name):
        return f"{name} is a provider credential name, not the Agenomic runtime token"
    if _SECRET_NAME.search(name) and not name.upper().startswith("AGENOMIC_"):
        return f"{name} names another credential; use an AGENOMIC_ variable for the runtime token"
    return None


def _check_runtime_token_env(name: str) -> None:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError("runtime_token_env must be an environment variable name")
    problem = runtime_token_env_problem(name)
    if problem is not None:
        # Hermes would send this variable's value as the model API key to Agenomic.
        raise ValueError(problem)


def render_hermes_config(
    endpoint: str,
    *,
    model: Optional[str] = None,
    runtime_token_env: str = DEFAULT_TOKEN_ENV,
    guard_timeout_s: int = 10,
    settings: Optional[Mapping[str, object]] = None,
) -> dict[str, JsonValue]:
    """Hermes ``config.yaml`` fragment routing Hermes through Agenomic.

    Agenomic ``${env:VAR}`` references in ``settings`` are rewritten to the
    ``${VAR}`` form Hermes expands in string values. ``runtime_token`` is never
    rendered into the plugin settings (Hermes would expand it into a literal);
    the adapter reads the default variable, or the reference from the
    ``AGENOMIC_HERMES_CONFIG`` file.

    Example:
        >>> cfg = render_hermes_config("https://agenomic.example", model="demo-model")
        >>> cfg["model"]["base_url"]
        'https://agenomic.example/v1/hermes/runtime/model/v1'
        >>> cfg["model"]["api_key"]
        '${AGENOMIC_HERMES_RUNTIME_TOKEN}'
        >>> cfg["hooks"]["pre_tool_call"][0]["fail_closed"]
        True
    """
    base = endpoint.rstrip("/")
    _check_runtime_token_env(runtime_token_env)
    if guard_timeout_s < 1 or guard_timeout_s > 300:
        raise ValueError("guard_timeout_s must be within [1, 300] (the Hermes clamp)")
    plugin_settings: dict[str, Any] = {"endpoint": base}
    for key, value in (settings or {}).items():
        if key == "runtime_token":
            raise ValueError(
                f"runtime_token cannot be rendered into Hermes settings; use {CONFIG_ENV}"
            )
        if key == "endpoint":
            continue
        plugin_settings[key] = _to_hermes_refs(value)
    model_block: dict[str, Any] = {
        "provider": "custom",
        "base_url": base + MODEL_GATEWAY_PATH,
        "api_key": "${" + runtime_token_env + "}",
        "api_mode": "chat_completions",
    }
    if model:
        model_block["default"] = model
    return {
        "model": model_block,
        "plugins": {
            "enabled": [PLUGIN_ID],
            # Shell hooks also run under this Python budget; keep it >= the guard timeout.
            "hook_callback_timeout": max(30, guard_timeout_s),
            "entries": {PLUGIN_ID: {"settings": plugin_settings}},
        },
        "skills": {"write_approval": True},
        "hooks": {
            "pre_tool_call": [
                {"command": GUARD_COMMAND, "fail_closed": True, "timeout": guard_timeout_s}
            ]
        },
        "hooks_auto_accept": True,
    }


def render_adapter_config(
    endpoint: str,
    *,
    runtime_token_env: str = DEFAULT_TOKEN_ENV,
    capture: Literal["metadata", "redacted_preview"] = "metadata",
    spool_path: Optional[str] = None,
) -> dict[str, JsonValue]:
    """Adapter config document for the ``AGENOMIC_HERMES_CONFIG`` file.

    Example:
        >>> doc = render_adapter_config("https://agenomic.example", runtime_token_env="AGENOMIC_RT")
        >>> doc["runtime_token"], doc["schema_version"]
        ('${env:AGENOMIC_RT}', 'agenomic.hermes.adapter_config/v1')
        >>> render_adapter_config("https://agenomic.example", runtime_token_env="OPENAI_API_KEY")
        Traceback (most recent call last):
        ...
        ValueError: OPENAI_API_KEY is a provider credential name, not the Agenomic runtime token
    """
    _check_runtime_token_env(runtime_token_env)
    buffer: dict[str, JsonValue] = {}
    if spool_path:
        buffer["spool_path"] = spool_path
    doc: dict[str, JsonValue] = {
        "schema_version": CONFIG_SCHEMA,
        "endpoint": endpoint.rstrip("/"),
        "runtime_token": "${env:" + runtime_token_env + "}",
        "capture": {"content": capture},
    }
    if buffer:
        doc["buffer"] = buffer
    AdapterConfig.model_validate(doc)
    return doc
