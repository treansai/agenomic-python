from __future__ import annotations

from time import perf_counter
from typing import Any

from agentlock.tracing import TraceRecorder
from agentlock.utils import to_jsonable


class AnthropicClientWrapper:
    def __init__(
        self,
        client: Any,
        trace: TraceRecorder,
        default_model: str | None = None,
    ) -> None:
        self.client = client
        self.trace = trace
        self.default_model = default_model

    def messages_create(self, *args: Any, **kwargs: Any) -> Any:
        messages = getattr(self.client, "messages", None)
        target = getattr(messages, "create", None)
        if target is None:
            msg = "Wrapped client does not expose messages.create(...)"
            raise AttributeError(msg)

        started = perf_counter()
        request_payload = {"args": args, "kwargs": kwargs}
        model_name = kwargs.get("model") or self.default_model or "anthropic.messages"

        try:
            response = target(*args, **kwargs)
        except Exception as exc:
            self.trace.add_model_call(
                name=model_name,
                provider="anthropic",
                request=request_payload,
                response={"error": type(exc).__name__},
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
            )
            raise

        self.trace.add_model_call(
            name=model_name,
            provider="anthropic",
            request=request_payload,
            response=to_jsonable(response),
            latency_ms=(perf_counter() - started) * 1000,
            success=True,
        )
        return response


def instrument_anthropic_client(
    client: Any,
    trace: TraceRecorder,
    default_model: str | None = None,
) -> AnthropicClientWrapper:
    return AnthropicClientWrapper(client=client, trace=trace, default_model=default_model)
