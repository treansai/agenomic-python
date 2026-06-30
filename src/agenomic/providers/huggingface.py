"""Hugging Face provider connector.

Pure-``httpx`` connector for the Hugging Face Hub and Inference API. The SDK
core never hard-requires ``huggingface-hub``; everything here uses ``httpx``,
which is already a hard dependency. The optional ``agenomic[huggingface]``
extra only matters for downstream tooling that wants the official hub client.

Security
--------
The API token is held privately on :class:`HuggingFaceConfig` (``_token``) and
is **never** placed in a ``repr``, returned object, trace, or error string.
Every error message produced here is passed through token redaction, which
scrubs both the configured token value and any ``hf_...``-shaped token.

Hashing
-------
Lockfile content hashes use **SHA-256 over canonical JSON** (sorted keys, no
insignificant whitespace, UTF-8). This is intentionally portable and matches
the cross-platform lockfile convention documented in
``docs/providers/huggingface.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

# Canonical provider name (accepted aliases are defined below).
CANONICAL_PROVIDER = "huggingface"

#: Default public Hugging Face Inference API base URL.
DEFAULT_INFERENCE_BASE = "https://api-inference.huggingface.co"
#: Hub API base URL (model metadata, whoami).
HUB_API_BASE = "https://huggingface.co"
#: Default request timeout in seconds.
DEFAULT_TIMEOUT_SECONDS = 30.0

# Matches ``hf_...`` tokens (user access tokens and fine-grained tokens).
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{4,}")


class HuggingFaceError(Exception):
    """Base error for the Hugging Face connector. Token-free by construction."""


class HuggingFaceAuthError(HuggingFaceError):
    """Authentication or authorization with Hugging Face failed (401/403)."""


def _normalized_alias(name: str) -> str:
    """Lower-case and replace ``-`` with ``_`` for alias comparison."""
    return name.strip().lower().replace("-", "_")


# Accepted aliases after normalization (``-``/``_`` equivalent, case-insensitive).
_ACCEPTED_ALIASES = {"huggingface", "hf", "hugging_face"}


def normalize_provider(name: Optional[str]) -> Optional[str]:
    """Return the canonical provider name for a Hugging Face alias, else ``None``.

    Accepts (case-insensitive, ``-``/``_`` equivalent): ``huggingface``, ``hf``,
    ``hugging_face`` (and e.g. ``hugging-face``, ``HuggingFace``).

    Example:
        >>> normalize_provider("Hugging-Face")
        'huggingface'
        >>> normalize_provider("openai") is None
        True
    """
    if not name:
        return None
    if _normalized_alias(name) in _ACCEPTED_ALIASES:
        return CANONICAL_PROVIDER
    return None


def is_huggingface(name: Optional[str]) -> bool:
    """True when ``name`` is any accepted Hugging Face alias.

    Example:
        >>> is_huggingface("HF")
        True
        >>> is_huggingface("anthropic")
        False
    """
    return normalize_provider(name) is not None


def _redact_tokens(text: str, token: Optional[str]) -> str:
    """Scrub the configured token value and any ``hf_...`` token from ``text``."""
    if token:
        text = text.replace(token, "***")
    return _HF_TOKEN_RE.sub("***", text)


def _reject_inline_credentials(url: str) -> None:
    """Raise if ``url`` embeds inline ``user:pass@host`` credentials."""
    parts = urlsplit(url)
    if parts.username or parts.password:
        raise HuggingFaceError("endpoint URL must not contain inline credentials (user:pass@host)")


def _redact_endpoint(url: str) -> str:
    """Return ``scheme://host[/path]`` with any credentials and query stripped."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


@dataclass
class ModelMetadata:
    """Resolved Hugging Face Hub metadata for a model at a revision."""

    model_id: str
    revision: str
    resolved_commit: Optional[str]
    task: Optional[str]
    private: bool


@dataclass
class HuggingFaceConfig:
    """Connection settings for Hugging Face.

    The token is stored in the private ``_token`` field and is excluded from
    ``repr`` so it cannot leak into logs. Use :meth:`from_env` to load from the
    environment with the documented precedence.
    """

    _token: Optional[str] = field(default=None, repr=False)
    endpoint_url: Optional[str] = None
    org: Optional[str] = None
    default_model: Optional[str] = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.endpoint_url:
            _reject_inline_credentials(self.endpoint_url)

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> HuggingFaceConfig:
        """Build a config from environment variables.

        Token precedence: ``HUGGINGFACE_API_TOKEN`` then ``HF_TOKEN``. Optional:
        ``HUGGINGFACE_ENDPOINT_URL``, ``HUGGINGFACE_ORG``,
        ``HUGGINGFACE_DEFAULT_MODEL``, ``HUGGINGFACE_TIMEOUT_SECONDS`` (default 30).
        """
        source = os.environ if env is None else env
        token = source.get("HUGGINGFACE_API_TOKEN") or source.get("HF_TOKEN")
        timeout_raw = source.get("HUGGINGFACE_TIMEOUT_SECONDS")
        try:
            timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_SECONDS
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_SECONDS
        return cls(
            _token=token,
            endpoint_url=source.get("HUGGINGFACE_ENDPOINT_URL"),
            org=source.get("HUGGINGFACE_ORG"),
            default_model=source.get("HUGGINGFACE_DEFAULT_MODEL"),
            timeout_seconds=timeout,
        )

    @property
    def has_token(self) -> bool:
        """True when a token is configured (the value itself is never exposed)."""
        return bool(self._token)

    @property
    def inference_base(self) -> str:
        """Inference base URL: the configured endpoint or the public API."""
        return (self.endpoint_url or DEFAULT_INFERENCE_BASE).rstrip("/")

    def redact(self, text: str) -> str:
        """Scrub the configured token and any ``hf_...`` token from ``text``.

        Example:
            >>> HuggingFaceConfig(_token="secret123").redact("got secret123 and hf_aaaa")
            'got *** and ***'
        """
        return _redact_tokens(text, self._token)

    def auth_header(self) -> dict[str, str]:
        """Return the ``Authorization`` header, or ``{}`` when no token is set."""
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}


class HuggingFaceClient:
    """Thin ``httpx`` adapter over the Hugging Face Hub and Inference APIs.

    All network errors are surfaced as :class:`HuggingFaceError` (or
    :class:`HuggingFaceAuthError` for 401/403) with tokens redacted.
    """

    def __init__(
        self,
        config: Optional[HuggingFaceConfig] = None,
        *,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.config = config or HuggingFaceConfig()
        self._transport = transport

    # -- internals ---------------------------------------------------------

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self.config.timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _safe(self, text: str) -> str:
        return self.config.redact(text)

    def _raise_for_status(self, response: httpx.Response, *, context: str) -> None:
        if response.status_code in (401, 403):
            raise HuggingFaceAuthError(
                self._safe(
                    f"Hugging Face authentication failed ({response.status_code}) "
                    f"for {context}. Check HUGGINGFACE_API_TOKEN / HF_TOKEN."
                )
            )
        if response.status_code >= 400:
            raise HuggingFaceError(
                self._safe(
                    f"Hugging Face request failed ({response.status_code}) for {context}: "
                    f"{response.text}"
                )
            )

    # -- public API --------------------------------------------------------

    def validate_credentials(self) -> dict[str, Any]:
        """Verify the token via ``whoami-v2``. Returns the (token-free) identity.

        Raises :class:`HuggingFaceAuthError` on 401/403, :class:`HuggingFaceError`
        otherwise.
        """
        if not self.config.has_token:
            raise HuggingFaceAuthError(
                "no Hugging Face token configured (set HUGGINGFACE_API_TOKEN or HF_TOKEN)"
            )
        url = f"{HUB_API_BASE}/api/whoami-v2"
        try:
            with self._client() as http:
                response = http.get(url, headers=self.config.auth_header())
        except httpx.HTTPError as exc:
            raise HuggingFaceError(self._safe(f"whoami request failed: {exc}")) from None
        self._raise_for_status(response, context="whoami-v2")
        data: dict[str, Any] = response.json()
        # Never echo any auth material back to the caller.
        data.pop("auth", None)
        return data

    def resolve_model_metadata(self, model_id: str, revision: str = "main") -> ModelMetadata:
        """Resolve Hub metadata for ``model_id`` at ``revision``.

        Reads ``id``, ``sha`` (resolved commit), ``pipeline_tag`` (task) and
        ``private`` from ``/api/models/{id}/revision/{rev}``.
        """
        url = f"{HUB_API_BASE}/api/models/{model_id}/revision/{revision}"
        try:
            with self._client() as http:
                response = http.get(url, headers=self.config.auth_header())
        except httpx.HTTPError as exc:
            raise HuggingFaceError(self._safe(f"model metadata request failed: {exc}")) from None
        self._raise_for_status(response, context=f"model {model_id}@{revision}")
        data: dict[str, Any] = response.json()
        return ModelMetadata(
            model_id=data.get("id", model_id),
            revision=revision,
            resolved_commit=data.get("sha"),
            task=data.get("pipeline_tag"),
            private=bool(data.get("private", False)),
        )

    def _inference(self, model: str, payload: dict[str, Any], *, context: str) -> Any:
        url = f"{self.config.inference_base}/models/{model}"
        try:
            with self._client() as http:
                response = http.post(url, headers=self.config.auth_header(), json=payload)
        except httpx.HTTPError as exc:
            raise HuggingFaceError(self._safe(f"{context} request failed: {exc}")) from None
        self._raise_for_status(response, context=context)
        return response.json()

    def generate_text(
        self,
        model: str,
        prompt: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Run text generation against the Inference API for ``model``.

        Returns the parsed JSON response. The body is
        ``{"inputs": prompt, "parameters": parameters}``.
        """
        payload: dict[str, Any] = {"inputs": prompt}
        if parameters:
            payload["parameters"] = parameters
        return self._inference(model, payload, context=f"generate_text({model})")

    def embeddings(self, model: str, inputs: Any) -> Any:
        """Compute embeddings / feature-extraction for ``inputs`` on ``model``."""
        payload: dict[str, Any] = {"inputs": inputs}
        return self._inference(model, payload, context=f"embeddings({model})")


def _canonical_json(value: Any) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def build_lockfile_model(
    *,
    config: HuggingFaceConfig,
    metadata: ModelMetadata,
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a lockfile entry for a Hugging Face model.

    Hashes are **SHA-256 over canonical JSON** (sorted keys, compact). The
    returned dict contains no token and a redacted ``endpoint_ref`` (no
    credentials, no query string).

    Keys: ``provider``, ``model_id``, ``revision``, ``resolved_commit``,
    ``task``, ``endpoint_ref``, ``endpoint_hash``, ``metadata_hash``,
    ``parameter_hash``.
    """
    endpoint_ref = _redact_endpoint(config.inference_base)
    params = parameters or {}
    metadata_payload = {
        "model_id": metadata.model_id,
        "revision": metadata.revision,
        "resolved_commit": metadata.resolved_commit,
        "task": metadata.task,
        "private": metadata.private,
    }
    return {
        "provider": CANONICAL_PROVIDER,
        "model_id": metadata.model_id,
        "revision": metadata.revision,
        "resolved_commit": metadata.resolved_commit,
        "task": metadata.task,
        "endpoint_ref": endpoint_ref,
        "endpoint_hash": _sha256_hex(endpoint_ref),
        "metadata_hash": _sha256_hex(metadata_payload),
        "parameter_hash": _sha256_hex(params),
    }


__all__ = [
    "CANONICAL_PROVIDER",
    "DEFAULT_TIMEOUT_SECONDS",
    "HuggingFaceAuthError",
    "HuggingFaceClient",
    "HuggingFaceConfig",
    "HuggingFaceError",
    "ModelMetadata",
    "build_lockfile_model",
    "is_huggingface",
    "normalize_provider",
]
