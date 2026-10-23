from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.llm import errors
from app.llm.types import (
    Completion,
    CompletionRequest,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)


class AnthropicProvider:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        name: str = "anthropic",
        **client_kwargs: Any,
    ) -> None:
        # max_retries=0: the resilient client owns the retry policy, and two
        # stacked policies multiply attempts (3 SDK retries x 3 ours = 9 calls).
        self.client = client or anthropic.AsyncAnthropic(max_retries=0, **client_kwargs)
        self.name = name

    def _params(self, model: str, request: CompletionRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            params["system"] = request.system
        if request.json_schema is not None:
            # transform_schema folds keywords the API does not enforce (minimum,
            # maxLength...) into descriptions instead of letting them 400.
            schema = anthropic.transform_schema(request.json_schema)
            params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return params

    async def complete(self, model: str, request: CompletionRequest) -> Completion:
        try:
            message = await self.client.messages.create(**self._params(model, request))
        except anthropic.AnthropicError as exc:
            raise map_error(exc) from exc
        except TypeError as exc:
            raise _missing_auth(exc) from exc
        return self._completion(message)

    async def stream(self, model: str, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        try:
            async with self.client.messages.stream(**self._params(model, request)) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                message = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise map_error(exc) from exc
        except TypeError as exc:
            raise _missing_auth(exc) from exc
        yield StreamDone(self._completion(message))

    def _completion(self, message: Any) -> Completion:
        if message.stop_reason == "refusal":
            raise errors.Refused("model refused")
        text = "".join(block.text for block in message.content if block.type == "text")
        u = message.usage
        return Completion(
            text=text,
            provider=self.name,
            model=message.model,
            stop_reason=message.stop_reason,
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=u.cache_read_input_tokens or 0,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
            ),
        )


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _missing_auth(exc: TypeError) -> Exception:
    # With no key, token or profile the SDK raises a bare TypeError at request
    # time instead of an AnthropicError. Without this, a machine with no Anthropic
    # credentials would 500 instead of falling back to the next target.
    if "authentication" in str(exc):
        return errors.TargetUnavailable(f"no Anthropic credentials: {exc}")
    return exc


def _error_type(exc: anthropic.APIStatusError) -> str | None:
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get("type")
    return None


def map_error(exc: anthropic.AnthropicError) -> errors.LLMError:
    if isinstance(exc, anthropic.CredentialsError):
        return errors.TargetUnavailable(f"no Anthropic credentials: {exc}")
    # Most specific first: APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, anthropic.APITimeoutError):
        return errors.ProviderTimeout(str(exc))
    if isinstance(exc, anthropic.APIConnectionError):
        return errors.ConnectionFailed(str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        if status < 400:
            # An SSE `error` event after a 200: the failure is server-side by
            # definition, its type lives only in the event body.
            if _error_type(exc) == "rate_limit_error":
                return errors.RateLimited(exc.message)
            return errors.ProviderOverloaded(exc.message)
        retry_after = _retry_after(exc)
        if status == 429:
            return errors.RateLimited(exc.message, retry_after=retry_after)
        if status in (408, 409) or status >= 500:
            return errors.ProviderOverloaded(exc.message, retry_after=retry_after)
        if status in (401, 403, 404):
            return errors.TargetUnavailable(exc.message)
        return errors.BadRequest(exc.message)
    return errors.LLMError(str(exc))
