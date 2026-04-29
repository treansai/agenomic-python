from __future__ import annotations

from time import perf_counter
from typing import Any

from agentlock.tracing import TraceRecorder
from agentlock.utils import to_jsonable


class OpenAIClientWrapper:
    def __init__(
        self,
        client: Any,
        trace: TraceRecorder,
        default_model: str | None = None,
    ) -> None:
        self.client = client
        self.trace = trace
        self.default_model = default_model

    def responses_create(self, *args: Any, **kwargs: Any) -> Any:
        target = getattr(getattr(self.client, "responses", None), "create", None)
        if target is None:
            msg = "Wrapped client does not expose responses.create(...)"
            raise AttributeError(msg)
        return self._invoke(
            target=target,
            request_args=args,
            request_kwargs=kwargs,
            model_name=kwargs.get("model") or self.default_model or "openai.responses",
        )

    def chat_completions_create(self, *args: Any, **kwargs: Any) -> Any:
        chat = getattr(self.client, "chat", None)
        completions = getattr(chat, "completions", None)
        target = getattr(completions, "create", None)
        if target is None:
            msg = "Wrapped client does not expose chat.completions.create(...)"
            raise AttributeError(msg)
        return self._invoke(
            target=target,
            request_args=args,
            request_kwargs=kwargs,
            model_name=kwargs.get("model") or self.default_model or "openai.chat",
        )

    def _invoke(
        self,
        target: Any,
        request_args: tuple[Any, ...],
        request_kwargs: dict[str, Any],
        model_name: str,
    ) -> Any:
        started = perf_counter()
        request_payload = {"args": request_args, "kwargs": request_kwargs}
        try:
            response = target(*request_args, **request_kwargs)
        except Exception as exc:
            self.trace.add_model_call(
                name=model_name,
                provider="openai",
                request=request_payload,
                response={"error": type(exc).__name__},
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
            )
            raise

        self.trace.add_model_call(
            name=model_name,
            provider="openai",
            request=request_payload,
            response=to_jsonable(response),
            latency_ms=(perf_counter() - started) * 1000,
            success=True,
        )
        return response


def instrument_openai_client(
    client: Any,
    trace: TraceRecorder,
    default_model: str | None = None,
) -> OpenAIClientWrapper:
    return OpenAIClientWrapper(client=client, trace=trace, default_model=default_model)
