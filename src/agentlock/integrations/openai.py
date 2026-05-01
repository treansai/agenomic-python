"""OpenAI integration — lazy-imported wrapper for chat completions.

Lazy: ``openai`` is imported inside :func:`instrument_openai`. The module
itself is safe to import without ``openai`` installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentlock.crypto.canonical import canonical_cbor
from agentlock.crypto.hashing import blake3_hex
from agentlock.trace.context import current_recorder
from agentlock.types.trace import CallStatus, ModelCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI, OpenAI


def _hash_request_body(payload: dict[str, Any]) -> str:
    return blake3_hex(canonical_cbor(payload))


def _hash_response_body(response: Any) -> str:
    try:
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception:
        data = {"repr": repr(response)}
    return blake3_hex(canonical_cbor(data))


def instrument_openai(client: OpenAI) -> OpenAI:
    """Wrap an OpenAI client so ``chat.completions.create`` records ModelCalls.

    Lazy-imports ``openai``. Raises ImportError with a helpful message if the
    package is not installed.

    Example:
        >>> # client = instrument_openai(OpenAI())  # doctest: +SKIP
    """
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "openai not installed. Install with: pip install agentlock[openai]"
        ) from e

    original = client.chat.completions.create

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        model = kwargs.get("model", "")
        prompt_hash = _hash_request_body({"model": model, **kwargs})
        started = time.perf_counter()
        status = CallStatus.SUCCESS
        try:
            response = original(*args, **kwargs)
        except Exception:
            status = CallStatus.ERROR
            if recorder is not None:
                recorder.record_model_call(
                    ModelCall(
                        provider="openai",
                        model=model,
                        prompt_hash=prompt_hash,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status=status,
                    )
                )
            raise
        latency = int((time.perf_counter() - started) * 1000)
        if recorder is not None:
            recorder.record_model_call(
                ModelCall(
                    provider="openai",
                    model=model,
                    fingerprint=getattr(response, "system_fingerprint", None),
                    temperature=kwargs.get("temperature"),
                    prompt_hash=prompt_hash,
                    output_hash=_hash_response_body(response),
                    latency_ms=latency,
                    status=status,
                )
            )
        return response

    client.chat.completions.create = wrapped
    return client


def instrument_openai_async(client: AsyncOpenAI) -> AsyncOpenAI:
    """Wrap an AsyncOpenAI client. See :func:`instrument_openai`."""
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "openai not installed. Install with: pip install agentlock[openai]"
        ) from e

    original = client.chat.completions.create

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        model = kwargs.get("model", "")
        prompt_hash = _hash_request_body({"model": model, **kwargs})
        started = time.perf_counter()
        status = CallStatus.SUCCESS
        try:
            response = await original(*args, **kwargs)
        except Exception:
            status = CallStatus.ERROR
            if recorder is not None:
                recorder.record_model_call(
                    ModelCall(
                        provider="openai",
                        model=model,
                        prompt_hash=prompt_hash,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status=status,
                    )
                )
            raise
        latency = int((time.perf_counter() - started) * 1000)
        if recorder is not None:
            recorder.record_model_call(
                ModelCall(
                    provider="openai",
                    model=model,
                    fingerprint=getattr(response, "system_fingerprint", None),
                    temperature=kwargs.get("temperature"),
                    prompt_hash=prompt_hash,
                    output_hash=_hash_response_body(response),
                    latency_ms=latency,
                    status=status,
                )
            )
        return response

    client.chat.completions.create = wrapped
    return client


__all__ = ["instrument_openai", "instrument_openai_async"]
