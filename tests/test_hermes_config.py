from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from agenomic.integrations.hermes.config import (
    AdapterConfig,
    ConfigError,
    build_config,
    render_adapter_config,
    render_hermes_config,
    settings_from_context,
)

SECRET = "agmhr_SUPERSECRETVALUE123"


def test_defaults_and_token_resolution() -> None:
    cfg, token = build_config(
        {"endpoint": "https://agenomic.example/"}, environ={"AGENOMIC_HERMES_RUNTIME_TOKEN": SECRET}
    )
    assert cfg.endpoint == "https://agenomic.example"
    assert cfg.timeouts.decision_s == 5.0
    assert cfg.buffer.max_events == 10_000
    assert cfg.buffer.batch_size == 500
    assert cfg.capture.content == "metadata"
    assert cfg.fail_mode == "closed"
    assert token is not None
    assert token.get_secret_value() == SECRET
    assert SECRET not in repr(cfg)
    assert SECRET not in repr(token)


def test_missing_token_variable_names_var_never_value(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.DEBUG)
    path = tmp_path / "adapter.json"
    path.write_text(
        json.dumps({"endpoint": "https://a.example", "runtime_token": "${env:AGENOMIC_MY_RT}"})
    )
    with pytest.raises(ConfigError) as err:
        build_config({}, environ={"AGENOMIC_HERMES_CONFIG": str(path), "OTHER": SECRET})
    assert "AGENOMIC_MY_RT" in str(err.value)
    assert SECRET not in str(err.value)
    assert SECRET not in caplog.text


def test_literal_token_is_refused_without_echo(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.DEBUG)
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps({"endpoint": "https://a.example", "runtime_token": SECRET}))
    with pytest.raises(ConfigError) as err:
        build_config({}, environ={"AGENOMIC_HERMES_CONFIG": str(path)}, resolve_token=False)
    assert "runtime_token must be an ${env:VAR} reference" in str(err.value)
    assert SECRET not in str(err.value)
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "name",
    ["OPENAI_API_KEY", "AGENOMIC_HERMES_SUPERVISOR_TOKEN", "GITHUB_TOKEN", "DB_PASSWORD"],
)
def test_file_runtime_token_cannot_reference_another_credential(
    name: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # RuntimeClient would otherwise send the provider/supervisor credential to Agenomic.
    caplog.set_level(logging.DEBUG)
    path = tmp_path / "adapter.yaml"
    path.write_text(f"endpoint: https://a.example\nruntime_token: ${{env:{name}}}\n")
    env = {"AGENOMIC_HERMES_CONFIG": str(path), name: SECRET}
    with pytest.raises(ConfigError, match=f"runtime_token.*{name}") as err:
        build_config({}, environ=env)
    assert SECRET not in str(err.value)
    assert SECRET not in caplog.text
    with pytest.raises(ValidationError):
        AdapterConfig.model_validate(
            {"endpoint": "https://a.example", "runtime_token": f"${{env:{name}}}"}
        )


@pytest.mark.parametrize("value", ["sk-openai-key", "agmhs_supervisor", "ghp_x", " agmhr_x"])
def test_resolved_runtime_token_must_be_an_agenomic_runtime_token(value: str) -> None:
    with pytest.raises(ConfigError, match="AGENOMIC_HERMES_RUNTIME_TOKEN does not hold") as err:
        build_config(
            {"endpoint": "https://a.example"}, environ={"AGENOMIC_HERMES_RUNTIME_TOKEN": value}
        )
    assert value not in str(err.value)
    _, token = build_config(
        {"endpoint": "https://a.example"}, environ={"AGENOMIC_HERMES_RUNTIME_TOKEN": "agmhr_ok"}
    )
    assert token is not None
    assert token.get_secret_value() == "agmhr_ok"


def test_token_in_plugin_settings_is_refused() -> None:
    # Hermes expands ${...} in plugin settings, so the adapter would receive the literal.
    with pytest.raises(ConfigError, match="must not be set in the Hermes plugin settings") as err:
        build_config({"endpoint": "https://a.example", "runtime_token": SECRET}, environ={})
    assert SECRET not in str(err.value)


def test_file_overrides_settings_and_resolves_env_refs(tmp_path: Path) -> None:
    path = tmp_path / "adapter.yaml"
    path.write_text(
        "endpoint: ${env:AGENOMIC_URL}\n"
        "runtime_token: ${env:TOK}\n"
        "capture: {content: redacted_preview}\n"
        'buffer:\n  spool_path: "${env:SPOOL}"\n'
    )
    env = {
        "AGENOMIC_HERMES_CONFIG": str(path),
        "AGENOMIC_URL": "https://b.example",
        "TOK": SECRET,
        "SPOOL": str(tmp_path / "spool.jsonl"),
    }
    cfg, token = build_config(
        {"endpoint": "https://a.example", "timeouts": {"decision_s": 2}}, environ=env
    )
    assert cfg.endpoint == "https://b.example"
    assert cfg.timeouts.decision_s == 2
    assert cfg.capture.content == "redacted_preview"
    assert cfg.buffer.spool_path == str(tmp_path / "spool.jsonl")
    assert cfg.runtime_token == "${env:TOK}"
    assert token is not None
    assert token.get_secret_value() == SECRET


def test_missing_reference_in_other_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="endpoint references environment variable NOPE"):
        build_config({"endpoint": "${env:NOPE}"}, environ={}, resolve_token=False)


@pytest.mark.parametrize(
    ("settings", "fragment"),
    [
        ({}, "endpoint is not configured"),
        ({"endpoint": "ftp://x"}, "endpoint must be an absolute http(s) URL"),
        (
            {"endpoint": "https://a.example", "unknown": 1},
            "unknown: Extra inputs are not permitted",
        ),
        ({"endpoint": "https://a.example", "buffer": {"batch_size": 501}}, "buffer.batch_size"),
        ({"endpoint": "https://a.example", "fail_mode": "open"}, "fail_mode"),
    ],
)
def test_invalid_configs(settings: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError) as err:
        build_config(settings, environ={}, resolve_token=False)
    assert fragment in str(err.value)


def test_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ConfigError, match="not valid JSON"):
        build_config({}, environ={"AGENOMIC_HERMES_CONFIG": str(bad)})
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n")
    with pytest.raises(ConfigError, match="must contain a mapping"):
        build_config({}, environ={"AGENOMIC_HERMES_CONFIG": str(lst)})
    with pytest.raises(ConfigError, match="cannot be read"):
        build_config({}, environ={"AGENOMIC_HERMES_CONFIG": str(tmp_path / "missing.yaml")})


def test_settings_from_context_reads_only_adapter_keys() -> None:
    class Ctx:
        def get_config(self, key: str, default: object = None) -> object:
            if key == "timeouts":
                raise ValueError("rejected path")
            return {"endpoint": "https://a.example", "capture": {"content": "metadata"}}.get(
                key, default
            )

    assert settings_from_context(Ctx()) == {
        "endpoint": "https://a.example",
        "capture": {"content": "metadata"},
    }
    assert settings_from_context(object()) == {}


def test_render_hermes_config() -> None:
    cfg = render_hermes_config(
        "https://agenomic.example/",
        model="demo-model",
        guard_timeout_s=40,
        settings={
            "capture": {"content": "metadata"},
            "buffer": {"spool_path": "${env:SPOOL_DIR}/s.jsonl"},
        },
    )
    assert cfg["model"] == {
        "provider": "custom",
        "base_url": "https://agenomic.example/v1/hermes/runtime/model/v1",
        "api_key": "${AGENOMIC_HERMES_RUNTIME_TOKEN}",
        "api_mode": "chat_completions",
        "default": "demo-model",
    }
    assert cfg["plugins"]["enabled"] == ["agenomic"]
    assert cfg["plugins"]["hook_callback_timeout"] >= 40
    assert cfg["skills"] == {"write_approval": True}
    assert cfg["hooks"]["pre_tool_call"] == [
        {"command": "agenomic-hermes-guard", "fail_closed": True, "timeout": 40}
    ]
    settings = cfg["plugins"]["entries"]["agenomic"]["settings"]
    assert settings["endpoint"] == "https://agenomic.example"
    assert settings["buffer"]["spool_path"] == "${SPOOL_DIR}/s.jsonl"
    assert "runtime_token" not in settings
    # The rendered settings are valid adapter settings once Hermes expanded them.
    AdapterConfig.model_validate({**settings, "buffer": {"spool_path": "/tmp/s.jsonl"}})
    with pytest.raises(ValueError, match="runtime_token cannot be rendered"):
        render_hermes_config("https://a.example", settings={"runtime_token": "${env:X}"})
    with pytest.raises(ValueError):
        render_hermes_config("https://a.example", runtime_token_env="bad name")
    with pytest.raises(ValueError):
        render_hermes_config("https://a.example", guard_timeout_s=0)


def test_render_adapter_config() -> None:
    doc = render_adapter_config(
        "https://a.example/", runtime_token_env="TOK", spool_path="/var/spool/a.jsonl"
    )
    assert doc == {
        "schema_version": "agenomic.hermes.adapter_config/v1",
        "endpoint": "https://a.example",
        "runtime_token": "${env:TOK}",
        "capture": {"content": "metadata"},
        "buffer": {"spool_path": "/var/spool/a.jsonl"},
    }


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "API_KEY",
        "AGENOMIC_HERMES_SUPERVISOR_TOKEN",
        "GITHUB_TOKEN",
    ],
)
def test_renderers_refuse_credential_names_for_the_runtime_token(name: str) -> None:
    # Hermes sends the variable named here as the model API key to Agenomic: a provider
    # or other credential name would hand that secret to the gateway.
    with pytest.raises(ValueError, match=name):
        render_hermes_config("https://a.example", runtime_token_env=name)
    with pytest.raises(ValueError, match=name):
        render_adapter_config("https://a.example", runtime_token_env=name)
    assert (
        render_hermes_config("https://a.example", runtime_token_env="AGENOMIC_RT")["model"][
            "api_key"
        ]
        == "${AGENOMIC_RT}"
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://gateway",
        "gateway.example",
        "http://[::1",
        "https://a.example:0",
        "https://a.example/?q=1",
        "https://user:pw@a.example",
        "https://user@a.example",
    ],
)
def test_render_hermes_config_refuses_an_endpoint_the_adapter_would_refuse(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        render_hermes_config(endpoint)
    with pytest.raises(ValueError, match="endpoint"):
        render_adapter_config(endpoint)


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": True},
        {"mode_hint": "invalid"},
        {"timeouts": {"decision_s": 0}},
        {"capture": {"content": "full"}},
        {"fail_mode": "open"},
    ],
)
def test_render_hermes_config_refuses_settings_the_adapter_would_refuse(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="setting"):
        render_hermes_config("https://a.example", settings=settings)


def test_render_hermes_config_keeps_env_references_for_load_time() -> None:
    cfg = render_hermes_config(
        "https://a.example",
        settings={"mode_hint": "shadow", "buffer": {"spool_path": "${env:SPOOL}"}},
    )
    entry = cfg["plugins"]["entries"]["agenomic"]["settings"]  # type: ignore[index,call-overload]
    assert entry["buffer"] == {"spool_path": "${SPOOL}"}
    assert entry["mode_hint"] == "shadow"


@pytest.mark.parametrize("key", ["timeouts", "buffer", "capture"])
def test_render_hermes_config_refuses_an_env_reference_for_a_mapping_setting(key: str) -> None:
    with pytest.raises(ValueError, match="mapping"):
        render_hermes_config("https://a.example", settings={key: "${env:X}"})


def test_render_hermes_config_accepts_an_env_reference_for_a_string_setting() -> None:
    cfg = render_hermes_config("https://a.example", settings={"mode_hint": "${env:MODE}"})
    entry = cfg["plugins"]["entries"]["agenomic"]["settings"]  # type: ignore[index,call-overload]
    assert entry["mode_hint"] == "${MODE}"
