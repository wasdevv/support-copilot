from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm import errors
from app.llm.types import (
    Completion,
    CompletionRequest,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)


class OllamaProvider:
    """Local model over Ollama's /api/chat. The fallback that survives any cloud outage."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        name: str = "ollama",
        http: httpx.AsyncClient | None = None,
    ) -> None:
        # No httpx timeout here: the resilient client bounds every attempt.
        self.http = http or httpx.AsyncClient(base_url=base_url, timeout=None)
        self.name = name

    def _body(self, model: str, request: CompletionRequest, stream: bool) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": request.max_tokens},
        }
        if request.json_schema is not None:
            body["format"] = request.json_schema
        return body

    async def complete(self, model: str, request: CompletionRequest) -> Completion:
        try:
            response = await self.http.post("/api/chat", json=self._body(model, request, False))
        except httpx.TimeoutException as exc:
            raise errors.ProviderTimeout(str(exc)) from exc
        except httpx.TransportError as exc:
            raise errors.ConnectionFailed(str(exc)) from exc
        _raise_for_status(response)
        return self._completion(model, _json_object(response.text))

    async def stream(self, model: str, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        parts: list[str] = []
        try:
            async with self.http.stream(
                "POST", "/api/chat", json=self._body(model, request, True)
            ) as response:
                if response.is_error:
                    await response.aread()
                    _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = _json_object(line)
                    if "error" in chunk:
                        raise errors.ProviderOverloaded(str(chunk["error"]))
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        parts.append(text)
                        yield TextDelta(text)
                    if chunk.get("done"):
                        chunk["message"] = {"content": "".join(parts)}
                        yield StreamDone(self._completion(model, chunk))
                        return
        except httpx.TimeoutException as exc:
            raise errors.ProviderTimeout(str(exc)) from exc
        except httpx.TransportError as exc:
            raise errors.ConnectionFailed(str(exc)) from exc
        raise errors.ConnectionFailed("stream ended without a done chunk")

    def _completion(self, model: str, data: dict[str, Any]) -> Completion:
        return Completion(
            text=data.get("message", {}).get("content", ""),
            provider=self.name,
            model=data.get("model", model),
            stop_reason=data.get("done_reason"),
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            ),
        )


def _json_object(raw: str) -> dict[str, Any]:
    # json.loads succeeds on "[]" or "7"; only an object is a valid Ollama payload.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise errors.ProviderOverloaded(f"unparseable response: {raw[:200]}") from exc
    if not isinstance(data, dict):
        raise errors.ProviderOverloaded(f"unexpected response: {raw[:200]}")
    return data


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    detail = response.text[:200]
    if status == 404:
        raise errors.TargetUnavailable(f"model not found: {detail}")
    if status == 429:
        raise errors.RateLimited(detail)
    if status >= 500:
        raise errors.ProviderOverloaded(detail)
    raise errors.BadRequest(detail)
