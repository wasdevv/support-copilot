"""Providers exercised through their real client stacks against mock transports that
speak each API's wire format. Fixtures follow the providers' documented payloads,
not what our mapping code happens to expect."""

from __future__ import annotations

import json

import anthropic
import httpx
import httpx2
import pytest

from app.llm import errors
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.types import CompletionRequest, Message, StreamDone, TextDelta

REQUEST = CompletionRequest(messages=[Message("user", "hi")], system="be brief", max_tokens=50)

MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 0,
    },
}


def anthropic_provider(handler) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="test",
        base_url="https://api.anthropic.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(client)


def api_error(status: int, kind: str, headers: dict[str, str] | None = None) -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": kind}}
    return httpx2.Response(status, json=body, headers=headers or {})


async def test_anthropic_complete_maps_request_and_usage():
    sent = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        return httpx2.Response(200, json=MESSAGE)

    completion = await anthropic_provider(handler).complete("claude-haiku-5-5", REQUEST)

    assert completion.text == "Hello!"
    assert completion.usage.input_tokens == 12 and completion.usage.cache_read_tokens == 100
    assert sent["system"] == "be brief" and sent["max_tokens"] == 50
    # Sampling parameters are rejected by the Messages API since 2026-10-21.
    assert "temperature" not in sent


async def test_anthropic_sends_json_schema_as_output_config():
    sent = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        return httpx2.Response(200, json=MESSAGE)

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    await anthropic_provider(handler).complete(
        "claude-haiku-5-5", CompletionRequest(messages=[Message("user", "x")], json_schema=schema)
    )

    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"] == anthropic.transform_schema(schema)


@pytest.mark.parametrize(
    ("status", "kind", "expected", "retry_after"),
    [
        (429, "rate_limit_error", errors.RateLimited, 7.0),
        (529, "overloaded_error", errors.ProviderOverloaded, None),
        (500, "api_error", errors.ProviderOverloaded, None),
        (401, "authentication_error", errors.TargetUnavailable, None),
        (404, "not_found_error", errors.TargetUnavailable, None),
        (400, "invalid_request_error", errors.BadRequest, None),
    ],
)
async def test_anthropic_status_codes_map_to_policy_errors(status, kind, expected, retry_after):
    headers = {"retry-after": "7"} if retry_after else None
    provider = anthropic_provider(lambda _r: api_error(status, kind, headers))

    with pytest.raises(expected) as exc:
        await provider.complete("claude-haiku-5-5", REQUEST)

    assert exc.value.retry_after == retry_after


async def test_anthropic_refusal_is_not_an_answer():
    refusal = {**MESSAGE, "stop_reason": "refusal", "content": []}
    provider = anthropic_provider(lambda _r: httpx2.Response(200, json=refusal))

    with pytest.raises(errors.Refused):
        await provider.complete("claude-haiku-5-5", REQUEST)


async def test_anthropic_connection_error_is_retryable():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(errors.ConnectionFailed):
        await anthropic_provider(handler).complete("claude-haiku-5-5", REQUEST)


async def test_anthropic_without_credentials_falls_back(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    client = anthropic.AsyncAnthropic(
        base_url="https://api.anthropic.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda _r: httpx2.Response(200, json=MESSAGE))
        ),
    )

    with pytest.raises(errors.TargetUnavailable):
        await AnthropicProvider(client).complete("claude-haiku-5-5", REQUEST)


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


async def test_anthropic_stream_yields_deltas_then_final_usage():
    start = {**MESSAGE, "content": [], "stop_reason": None}
    start["usage"] = {"input_tokens": 12, "output_tokens": 1}
    body = sse(
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    provider = anthropic_provider(
        lambda _r: httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )

    events = [e async for e in provider.stream("claude-haiku-5-5", REQUEST)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.completion.text == "Hello" and done.completion.usage.output_tokens == 4


async def test_anthropic_stream_error_event_maps_to_overloaded():
    start = {**MESSAGE, "content": [], "stop_reason": None}
    body = sse(
        ("message_start", {"type": "message_start", "message": start}),
        ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
    )
    provider = anthropic_provider(
        lambda _r: httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )

    with pytest.raises(errors.ProviderOverloaded):
        [e async for e in provider.stream("claude-haiku-5-5", REQUEST)]


def ollama_provider(handler) -> OllamaProvider:
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaProvider(http=http)


async def test_ollama_complete_maps_counts_and_system_prompt():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hi."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 2,
            },
        )

    completion = await ollama_provider(handler).complete("llama3.2", REQUEST)

    assert completion.text == "Hi."
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (20, 2)
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["stream"] is False and sent["options"]["num_predict"] == 50


async def test_ollama_missing_model_falls_back():
    provider = ollama_provider(
        lambda _r: httpx.Response(404, json={"error": "model 'x' not found"})
    )

    with pytest.raises(errors.TargetUnavailable):
        await provider.complete("x", REQUEST)


async def test_ollama_non_object_body_is_an_error():
    provider = ollama_provider(lambda _r: httpx.Response(200, text="[]"))

    with pytest.raises(errors.ProviderOverloaded):
        await provider.complete("llama3.2", REQUEST)


async def test_ollama_stream_reads_ndjson():
    lines = [
        {"model": "llama3.2", "message": {"content": "Hel"}, "done": False},
        {"model": "llama3.2", "message": {"content": "lo"}, "done": False},
        {
            "model": "llama3.2",
            "message": {"content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 2,
        },
    ]
    body = "\n".join(json.dumps(line) for line in lines) + "\n"
    provider = ollama_provider(lambda _r: httpx.Response(200, text=body))

    events = [e async for e in provider.stream("llama3.2", REQUEST)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert events[-1].completion.text == "Hello"
    assert events[-1].completion.usage.output_tokens == 2


async def test_ollama_stream_cut_without_done_is_a_connection_failure():
    body = json.dumps({"message": {"content": "Hel"}, "done": False}) + "\n"
    provider = ollama_provider(lambda _r: httpx.Response(200, text=body))

    with pytest.raises(errors.ConnectionFailed):
        [e async for e in provider.stream("llama3.2", REQUEST)]
