"""Hugging Face integration — record ModelCalls for inference calls.

Mirrors the OpenAI/Anthropic integrations. Two entry points:

* :func:`instrument_huggingface` wraps an
  :class:`~agenomic.providers.huggingface.HuggingFaceClient` so its
  ``generate_text`` / ``embeddings`` calls record a ``ModelCall`` on the
  active recorder.
* :func:`trace_huggingface_call` wraps an arbitrary inference callable (e.g.
  ``huggingface_hub.InferenceClient.text_generation``) when you do not use the
  bundled client.

The token is never recorded: only ``provider``, ``model``, hashes, latency and
status reach the trace.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypeVar

from agenomic.crypto.canonical import canonical_cbor
from agenomic.crypto.hashing import blake3_hex
from agenomic.providers.huggingface import CANONICAL_PROVIDER, HuggingFaceClient
from agenomic.trace.context import current_recorder
from agenomic.types.trace import CallStatus, ModelCall

T = TypeVar("T")


def _hash_request(payload: dict[str, Any]) -> str:
    return blake3_hex(canonical_cbor(payload))


def _hash_response(response: Any) -> str:
    try:
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, (dict, list, str, int, float, bool)) or response is None:
            data = response
        else:
            data = dict(response)
    except Exception:
        data = {"repr": repr(response)}
    return blake3_hex(canonical_cbor(data))


def _record(
    *,
    model: str,
    prompt_hash: str,
    parameters: Optional[dict[str, Any]],
    response: Any,
    started: float,
    status: CallStatus,
) -> None:
    recorder = current_recorder()
    if recorder is None:
        return
    temperature = None
    if isinstance(parameters, dict):
        temp = parameters.get("temperature")
        if isinstance(temp, (int, float)):
            temperature = float(temp)
    recorder.record_model_call(
        ModelCall(
            provider=CANONICAL_PROVIDER,
            model=model,
            temperature=temperature,
            prompt_hash=prompt_hash,
            output_hash=_hash_response(response) if status is CallStatus.SUCCESS else None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=status,
        )
    )


def trace_huggingface_call(
    fn: Callable[..., T],
    *,
    model: str,
    prompt: Any = None,
    parameters: Optional[dict[str, Any]] = None,
    **call_kwargs: Any,
) -> T:
    """Invoke ``fn(**call_kwargs)`` and record a ``ModelCall`` for it.

    Use when you call a Hugging Face inference function directly. ``model`` and
    optional ``prompt``/``parameters`` are used only to build the prompt hash
    and trace metadata; they are not forwarded unless present in ``call_kwargs``.
    The token never enters the recorded call.
    """
    prompt_hash = _hash_request({"model": model, "inputs": prompt, "parameters": parameters or {}})
    started = time.perf_counter()
    try:
        response = fn(**call_kwargs)
    except Exception:
        _record(
            model=model,
            prompt_hash=prompt_hash,
            parameters=parameters,
            response=None,
            started=started,
            status=CallStatus.ERROR,
        )
        raise
    _record(
        model=model,
        prompt_hash=prompt_hash,
        parameters=parameters,
        response=response,
        started=started,
        status=CallStatus.SUCCESS,
    )
    return response


def instrument_huggingface(client: HuggingFaceClient) -> HuggingFaceClient:
    """Wrap a :class:`HuggingFaceClient` so inference records ``ModelCall``s.

    Wraps ``generate_text`` and ``embeddings`` (in place) so each call records a
    ``ModelCall(provider="huggingface", model=...)`` on success and on error.

    Example:
        >>> from agenomic.providers.huggingface import HuggingFaceClient
        >>> client = instrument_huggingface(HuggingFaceClient())  # doctest: +SKIP
    """
    original_generate = client.generate_text
    original_embeddings = client.embeddings

    def generate_text(model: str, prompt: str, parameters: Optional[dict[str, Any]] = None) -> Any:
        prompt_hash = _hash_request(
            {"model": model, "inputs": prompt, "parameters": parameters or {}}
        )
        started = time.perf_counter()
        try:
            response = original_generate(model, prompt, parameters)
        except Exception:
            _record(
                model=model,
                prompt_hash=prompt_hash,
                parameters=parameters,
                response=None,
                started=started,
                status=CallStatus.ERROR,
            )
            raise
        _record(
            model=model,
            prompt_hash=prompt_hash,
            parameters=parameters,
            response=response,
            started=started,
            status=CallStatus.SUCCESS,
        )
        return response

    def embeddings(model: str, inputs: Any) -> Any:
        prompt_hash = _hash_request({"model": model, "inputs": inputs})
        started = time.perf_counter()
        try:
            response = original_embeddings(model, inputs)
        except Exception:
            _record(
                model=model,
                prompt_hash=prompt_hash,
                parameters=None,
                response=None,
                started=started,
                status=CallStatus.ERROR,
            )
            raise
        _record(
            model=model,
            prompt_hash=prompt_hash,
            parameters=None,
            response=response,
            started=started,
            status=CallStatus.SUCCESS,
        )
        return response

    client.generate_text = generate_text  # type: ignore[method-assign]
    client.embeddings = embeddings  # type: ignore[method-assign]
    return client


__all__ = ["instrument_huggingface", "trace_huggingface_call"]
