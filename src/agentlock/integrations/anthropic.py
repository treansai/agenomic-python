"""Anthropic integration — lazy-imported wrapper for messages.create."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentlock.crypto.canonical import canonical_cbor
from agentlock.crypto.hashing import blake3_hex
from agentlock.trace.context import current_recorder
from agentlock.types.trace import CallStatus, ModelCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic import Anthropic, AsyncAnthropic


def _hash_request(payload: dict[str, Any]) -> str:
    return blake3_hex(canonical_cbor(payload))


def _hash_response(response: Any) -> str:
    try:
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception:
        data = {"repr": repr(response)}
    return blake3_hex(canonical_cbor(data))


def instrument_anthropic(client: Anthropic) -> Anthropic:
    """Wrap an Anthropic client so ``messages.create`` records ModelCalls.

    Lazy-imports ``anthropic``. Raises ImportError with a helpful message
    if not installed.

    Example:
        >>> # client = instrument_anthropic(Anthropic())  # doctest: +SKIP
    """
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "anthropic not installed. Install with: pip install agentlock[anthropic]"
        ) from e

    original = client.messages.create

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        model = kwargs.get("model", "")
        prompt_hash = _hash_request({"model": model, **kwargs})
        started = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        except Exception:
            if recorder is not None:
                recorder.record_model_call(
                    ModelCall(
                        provider="anthropic",
                        model=model,
                        prompt_hash=prompt_hash,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status=CallStatus.ERROR,
                    )
                )
            raise
        if recorder is not None:
            recorder.record_model_call(
                ModelCall(
                    provider="anthropic",
                    model=model,
                    temperature=kwargs.get("temperature"),
                    prompt_hash=prompt_hash,
                    output_hash=_hash_response(response),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return response

    setattr(client.messages, "create", wrapped)  # noqa: B010
    return client


def instrument_anthropic_async(client: AsyncAnthropic) -> AsyncAnthropic:
    """Async variant. See :func:`instrument_anthropic`."""
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "anthropic not installed. Install with: pip install agentlock[anthropic]"
        ) from e

    original = client.messages.create

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        model = kwargs.get("model", "")
        prompt_hash = _hash_request({"model": model, **kwargs})
        started = time.perf_counter()
        try:
            response = await original(*args, **kwargs)
        except Exception:
            if recorder is not None:
                recorder.record_model_call(
                    ModelCall(
                        provider="anthropic",
                        model=model,
                        prompt_hash=prompt_hash,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status=CallStatus.ERROR,
                    )
                )
            raise
        if recorder is not None:
            recorder.record_model_call(
                ModelCall(
                    provider="anthropic",
                    model=model,
                    temperature=kwargs.get("temperature"),
                    prompt_hash=prompt_hash,
                    output_hash=_hash_response(response),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return response

    setattr(client.messages, "create", wrapped)  # noqa: B010
    return client


__all__ = ["instrument_anthropic", "instrument_anthropic_async"]
