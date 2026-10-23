"""Deterministic fault injection for tests and the outage benchmark.

A real outage cannot be scheduled, so the benchmark replays modelled ones: random
5xx, 429 bursts with Retry-After, hard outages and slow responses. The fault for
call N comes from a seeded RNG, so a run is reproducible.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from app.llm import errors
from app.llm.types import (
    Completion,
    CompletionRequest,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)

# Returns an exception to raise for this call, or None for a healthy call.
FaultFn = Callable[[int, random.Random], errors.LLMError | None]


def healthy(_call: int, _rng: random.Random) -> None:
    return None


def random_errors(rate: float, kind: type[errors.LLMError] = errors.ProviderOverloaded) -> FaultFn:
    def fault(_call: int, rng: random.Random) -> errors.LLMError | None:
        return kind("injected") if rng.random() < rate else None

    return fault


def rate_limited(rate: float, retry_after: float) -> FaultFn:
    def fault(_call: int, rng: random.Random) -> errors.LLMError | None:
        return (
            errors.RateLimited("injected", retry_after=retry_after) if rng.random() < rate else None
        )

    return fault


def down() -> FaultFn:
    def fault(_call: int, _rng: random.Random) -> errors.LLMError:
        return errors.ProviderOverloaded("injected outage")

    return fault


def bursts(period_s: float, down_s: float, phase_s: float = 0.0) -> FaultFn:
    """Down for `down_s` out of every `period_s`, counted from the first call.
    Every call inside a window fails: an outage, not bad luck per call."""
    started: float | None = None

    def fault(_call: int, _rng: random.Random) -> errors.LLMError | None:
        nonlocal started
        now = time.monotonic()
        if started is None:
            started = now
        if (now - started + phase_s) % period_s < down_s:
            return errors.ProviderOverloaded("injected burst outage")
        return None

    return fault


def fail_calls(*calls: int, kind: type[errors.LLMError] = errors.ProviderOverloaded) -> FaultFn:
    """Fail exactly the listed call numbers (0-based). For tests."""

    def fault(call: int, _rng: random.Random) -> errors.LLMError | None:
        return kind("injected") if call in calls else None

    return fault


@dataclass
class SimulatedProvider:
    name: str = "sim"
    fault: FaultFn = healthy
    latency_s: float = 0.0
    # Probability that a call takes slow_latency_s instead (models a tail).
    slow_rate: float = 0.0
    slow_latency_s: float = 0.0
    reply: str = "Simulated reply."
    # Mid-stream failure after this many chunks (None = never). For tests.
    fail_stream_after: int | None = None
    seed: int = 0
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _text(self, request: CompletionRequest) -> str:
        if request.json_schema is None:
            return self.reply
        # Satisfy the schema with the first enum value / a placeholder per field.
        schema = request.json_schema
        defs = schema.get("$defs", {})
        out: dict[str, object] = {}
        for key, spec in schema.get("properties", {}).items():
            if "$ref" in spec:
                spec = defs[spec["$ref"].rsplit("/", 1)[-1]]
            out[key] = spec["enum"][0] if "enum" in spec else _placeholder(spec.get("type"))
        return json.dumps(out)

    async def _prepare(self) -> None:
        call = self.calls
        self.calls += 1
        failure = self.fault(call, self._rng)
        slow = self._rng.random() < self.slow_rate
        delay = self.slow_latency_s if slow else self.latency_s
        if delay:
            await asyncio.sleep(delay)
        if failure is not None:
            raise failure

    def _completion(self, model: str, request: CompletionRequest, text: str) -> Completion:
        prompt = sum(len(m.content) for m in request.messages) + len(request.system or "")
        return Completion(
            text=text,
            provider=self.name,
            model=model,
            stop_reason="end_turn",
            usage=Usage(input_tokens=max(1, prompt // 4), output_tokens=max(1, len(text) // 4)),
        )

    async def complete(self, model: str, request: CompletionRequest) -> Completion:
        await self._prepare()
        return self._completion(model, request, self._text(request))

    async def stream(self, model: str, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        await self._prepare()
        text = self._text(request)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self.fail_stream_after is not None and i == self.fail_stream_after:
                raise errors.ConnectionFailed("injected mid-stream drop")
            yield TextDelta(word if i == 0 else " " + word)
            await asyncio.sleep(0)
        yield StreamDone(self._completion(model, request, text))


def _placeholder(kind: str | None) -> object:
    return {"integer": 0, "number": 0, "boolean": False, "array": []}.get(kind or "", "x")
