"""Tests for the Hugging Face provider connector. No real network is used."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from agenomic.providers.huggingface import (
    CANONICAL_PROVIDER,
    HuggingFaceAuthError,
    HuggingFaceClient,
    HuggingFaceConfig,
    HuggingFaceError,
    build_lockfile_model,
    is_huggingface,
    normalize_provider,
)

# -- normalization / aliases ------------------------------------------------


@pytest.mark.parametrize(
    "alias",
    ["huggingface", "HuggingFace", "hf", "HF", "hugging_face", "Hugging-Face", "HUGGING-FACE"],
)
def test_normalize_provider_accepts_aliases(alias: str) -> None:
    assert normalize_provider(alias) == CANONICAL_PROVIDER
    assert is_huggingface(alias) is True


@pytest.mark.parametrize("name", ["openai", "anthropic", "", None, "huggingfaces"])
def test_normalize_provider_rejects_others(name: str | None) -> None:
    assert normalize_provider(name) is None
    assert is_huggingface(name) is False


# -- env fallback -----------------------------------------------------------


def test_from_env_prefers_huggingface_api_token() -> None:
    cfg = HuggingFaceConfig.from_env(
        {"HUGGINGFACE_API_TOKEN": "hf_primary", "HF_TOKEN": "hf_fallback"}
    )
    assert cfg.has_token
    assert cfg.auth_header() == {"Authorization": "Bearer hf_primary"}


def test_from_env_falls_back_to_hf_token() -> None:
    cfg = HuggingFaceConfig.from_env({"HF_TOKEN": "hf_fallback"})
    assert cfg.auth_header() == {"Authorization": "Bearer hf_fallback"}


def test_from_env_optional_vars_and_timeout_default() -> None:
    cfg = HuggingFaceConfig.from_env({"HF_TOKEN": "hf_x"})
    assert cfg.timeout_seconds == 30.0
    cfg2 = HuggingFaceConfig.from_env(
        {
            "HF_TOKEN": "hf_x",
            "HUGGINGFACE_ENDPOINT_URL": "https://example.endpoints.huggingface.cloud",
            "HUGGINGFACE_ORG": "acme",
            "HUGGINGFACE_DEFAULT_MODEL": "gpt2",
            "HUGGINGFACE_TIMEOUT_SECONDS": "12.5",
        }
    )
    assert cfg2.org == "acme"
    assert cfg2.default_model == "gpt2"
    assert cfg2.timeout_seconds == 12.5
    assert cfg2.inference_base == "https://example.endpoints.huggingface.cloud"


def test_token_not_in_repr() -> None:
    cfg = HuggingFaceConfig(_token="hf_supersecret")
    assert "hf_supersecret" not in repr(cfg)


def test_inline_credentials_rejected() -> None:
    with pytest.raises(HuggingFaceError, match="inline credentials"):
        HuggingFaceConfig(endpoint_url="https://user:pass@host.example/x")


# -- validate_credentials ---------------------------------------------------


def test_validate_credentials_success(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://huggingface.co/api/whoami-v2",
        json={"name": "octocat", "type": "user", "auth": {"accessToken": "secret"}},
    )
    client = HuggingFaceClient(HuggingFaceConfig(_token="hf_abc"))
    identity = client.validate_credentials()
    assert identity["name"] == "octocat"
    # Auth material is stripped from the returned identity.
    assert "auth" not in identity


def test_validate_credentials_401(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://huggingface.co/api/whoami-v2", status_code=401)
    client = HuggingFaceClient(HuggingFaceConfig(_token="hf_abc"))
    with pytest.raises(HuggingFaceAuthError):
        client.validate_credentials()


def test_validate_credentials_no_token() -> None:
    client = HuggingFaceClient(HuggingFaceConfig())
    with pytest.raises(HuggingFaceAuthError, match="no Hugging Face token"):
        client.validate_credentials()


# -- resolve_model_metadata -------------------------------------------------


def test_resolve_model_metadata(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.3/revision/main",
        json={
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "abc123def456",
            "pipeline_tag": "text-generation",
            "private": False,
        },
    )
    client = HuggingFaceClient(HuggingFaceConfig(_token="hf_abc"))
    meta = client.resolve_model_metadata("mistralai/Mistral-7B-Instruct-v0.3")
    assert meta.resolved_commit == "abc123def456"
    assert meta.task == "text-generation"
    assert meta.private is False
    assert meta.revision == "main"


# -- generate_text / embeddings ---------------------------------------------


def test_generate_text_happy_path(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api-inference.huggingface.co/models/gpt2",
        json=[{"generated_text": "hello world"}],
    )
    client = HuggingFaceClient(HuggingFaceConfig(_token="hf_abc"))
    out = client.generate_text("gpt2", "hello", parameters={"max_new_tokens": 5})
    assert out == [{"generated_text": "hello world"}]
    request = httpx_mock.get_request()
    assert request is not None
    body = request.read().decode()
    assert '"inputs":"hello"' in body.replace(" ", "")


def test_embeddings_uses_endpoint_url(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://my.endpoint.example/models/sentence-transformers/all-MiniLM-L6-v2",
        json=[[0.1, 0.2, 0.3]],
    )
    cfg = HuggingFaceConfig(_token="hf_abc", endpoint_url="https://my.endpoint.example")
    client = HuggingFaceClient(cfg)
    out = client.embeddings("sentence-transformers/all-MiniLM-L6-v2", "hi")
    assert out == [[0.1, 0.2, 0.3]]


# -- token redaction in errors ----------------------------------------------


def test_token_redacted_in_error_messages(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api-inference.huggingface.co/models/gpt2",
        status_code=500,
        text="boom leaking hf_secrettoken in body",
    )
    cfg = HuggingFaceConfig(_token="hf_configuredtoken")
    client = HuggingFaceClient(cfg)
    with pytest.raises(HuggingFaceError) as exc:
        client.generate_text("gpt2", "x")
    msg = str(exc.value)
    assert "hf_secrettoken" not in msg
    assert "hf_configuredtoken" not in msg
    assert "***" in msg


def test_config_redact_scrubs_tokens() -> None:
    cfg = HuggingFaceConfig(_token="hf_configured")
    redacted = cfg.redact("value hf_configured and another hf_abcdef plus text")
    assert "hf_configured" not in redacted
    assert "hf_abcdef" not in redacted


# -- lockfile builder -------------------------------------------------------


def test_build_lockfile_model() -> None:
    from agenomic.providers.huggingface import ModelMetadata

    cfg = HuggingFaceConfig(_token="hf_secret")
    meta = ModelMetadata(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="main",
        resolved_commit="deadbeef",
        task="text-generation",
        private=False,
    )
    entry = build_lockfile_model(config=cfg, metadata=meta, parameters={"temperature": 0.7})
    assert entry["provider"] == "huggingface"
    assert entry["model_id"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert entry["resolved_commit"] == "deadbeef"
    assert entry["task"] == "text-generation"
    assert entry["endpoint_ref"] == "https://api-inference.huggingface.co"
    # Hashes are deterministic sha256 hex (64 chars).
    for key in ("endpoint_hash", "metadata_hash", "parameter_hash"):
        assert len(entry[key]) == 64
    # No token anywhere in the serialized entry.
    assert "hf_secret" not in str(entry)


def test_lockfile_endpoint_ref_redacts_credentials() -> None:
    from agenomic.providers.huggingface import ModelMetadata, _redact_endpoint

    assert (
        _redact_endpoint("https://user:pass@host.example/path?token=x")
        == "https://host.example/path"
    )
    meta = ModelMetadata("m", "main", "sha", "text-generation", False)
    # endpoint_url with creds is rejected at construction, so feed via redact path
    cfg = HuggingFaceConfig(_token="hf_x")
    entry = build_lockfile_model(config=cfg, metadata=meta)
    assert "@" not in entry["endpoint_ref"]


def test_lockfile_parameter_hash_stable() -> None:
    from agenomic.providers.huggingface import ModelMetadata

    cfg = HuggingFaceConfig()
    meta = ModelMetadata("m", "main", "sha", "text-generation", False)
    a = build_lockfile_model(config=cfg, metadata=meta, parameters={"a": 1, "b": 2})
    b = build_lockfile_model(config=cfg, metadata=meta, parameters={"b": 2, "a": 1})
    assert a["parameter_hash"] == b["parameter_hash"]
