"""Anthropic integration — lazy-imported wrapper for messages.create."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.protect.models import Overlay, overlay_text
from agenomic.trace.context import current_recorder
from agenomic.types.trace import CallStatus, ModelCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic import Anthropic, AsyncAnthropic

_SYSTEM_SEPARATOR = "\n\n"


def inject_anthropic_overlay(kwargs: dict[str, Any], text: Optional[str]) -> dict[str, Any]:
    """Set ``system`` to the overlay, or prefix the existing system prompt with it, once.

    Example:
        >>> inject_anthropic_overlay({"system": "You help."}, "Rule.")["system"]
        'Rule.\\n\\nYou help.'
        >>> inject_anthropic_overlay({"system": "Rule.\\n\\nYou help."}, "Rule.")["system"]
        'Rule.\\n\\nYou help.'
    """
    if text is None:
        return kwargs
    system = kwargs.get("system")
    if system is None or system == "":
        return {**kwargs, "system": text}
    if isinstance(system, str):
        if system == text or system.startswith(text + _SYSTEM_SEPARATOR):
            return kwargs
        return {**kwargs, "system": text + _SYSTEM_SEPARATOR + system}
    blocks = list(system)
    first = blocks[0] if blocks else None
    if isinstance(first, Mapping) and first.get("type") == "text" and first.get("text") == text:
        return kwargs
    return {**kwargs, "system": [{"type": "text", "text": text}, *blocks]}


def _hash_request(payload: dict[str, Any]) -> str:
    return blake3_hex(canonical_cbor(payload))


def _hash_response(response: Any) -> str:
    try:
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception:
        data = {"repr": repr(response)}
    return blake3_hex(canonical_cbor(data))


def instrument_anthropic(client: Anthropic, *, overlay: Overlay = None) -> Anthropic:
    """Wrap an Anthropic client so ``messages.create`` records ModelCalls.

    Lazy-imports ``anthropic``. Raises ImportError with a helpful message
    if not installed. ``overlay`` (text or :class:`ProtectOverlay`) becomes
    the ``system`` prompt, or prefixes the caller's one, once per request;
    the prompt hash covers the injected text.

    Example:
        >>> # client = instrument_anthropic(Anthropic(), overlay=protect.overlay(run_id))  # doctest: +SKIP
    """
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "anthropic not installed. Install with: pip install agenomic[anthropic]"
        ) from e

    original = client.messages.create
    text = overlay_text(overlay)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        kwargs = inject_anthropic_overlay(kwargs, text)
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


def instrument_anthropic_async(
    client: AsyncAnthropic, *, overlay: Overlay = None
) -> AsyncAnthropic:
    """Async variant. See :func:`instrument_anthropic`."""
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "anthropic not installed. Install with: pip install agenomic[anthropic]"
        ) from e

    original = client.messages.create
    text = overlay_text(overlay)

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        kwargs = inject_anthropic_overlay(kwargs, text)
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
