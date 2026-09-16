"""OpenAI integration — lazy-imported wrapper for chat completions.

Lazy: ``openai`` is imported inside :func:`instrument_openai`. The module
itself is safe to import without ``openai`` installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.protect.models import Overlay, overlay_text
from agenomic.trace.context import current_recorder
from agenomic.types.trace import CallStatus, ModelCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI, OpenAI


def inject_openai_overlay(kwargs: dict[str, Any], text: Optional[str]) -> dict[str, Any]:
    """Prepend the overlay as the first system message, once.

    Example:
        >>> inject_openai_overlay({"messages": [{"role": "user", "content": "hi"}]}, "Rule.")["messages"][0]
        {'role': 'system', 'content': 'Rule.'}
    """
    if text is None:
        return kwargs
    messages = list(kwargs.get("messages") or [])
    first = messages[0] if messages else None
    if (
        isinstance(first, Mapping)
        and first.get("role") == "system"
        and first.get("content") == text
    ):
        return kwargs
    return {**kwargs, "messages": [{"role": "system", "content": text}, *messages]}


def _hash_request_body(payload: dict[str, Any]) -> str:
    return blake3_hex(canonical_cbor(payload))


def _hash_response_body(response: Any) -> str:
    try:
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception:
        data = {"repr": repr(response)}
    return blake3_hex(canonical_cbor(data))


def instrument_openai(client: OpenAI, *, overlay: Overlay = None) -> OpenAI:
    """Wrap an OpenAI client so ``chat.completions.create`` records ModelCalls.

    Lazy-imports ``openai``. Raises ImportError with a helpful message if the
    package is not installed. ``overlay`` (text or :class:`ProtectOverlay`)
    is prepended as the first system message of every request, once; the
    prompt hash covers the injected message.

    Example:
        >>> # client = instrument_openai(OpenAI(), overlay=protect.overlay(run_id))  # doctest: +SKIP
    """
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise ImportError("openai not installed. Install with: pip install agenomic[openai]") from e

    original = client.chat.completions.create
    text = overlay_text(overlay)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        kwargs = inject_openai_overlay(kwargs, text)
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

    setattr(client.chat.completions, "create", wrapped)  # noqa: B010
    return client


def instrument_openai_async(client: AsyncOpenAI, *, overlay: Overlay = None) -> AsyncOpenAI:
    """Wrap an AsyncOpenAI client. See :func:`instrument_openai`."""
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise ImportError("openai not installed. Install with: pip install agenomic[openai]") from e

    original = client.chat.completions.create
    text = overlay_text(overlay)

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        kwargs = inject_openai_overlay(kwargs, text)
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

    setattr(client.chat.completions, "create", wrapped)  # noqa: B010
    return client


__all__ = ["instrument_openai", "instrument_openai_async"]
