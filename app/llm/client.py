"""The resilient client: every LLM call is bounded, retried when retrying can help,
failed over when it cannot, and logged with its cost and latency."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace

from pydantic import BaseModel, ValidationError

from app.llm import errors
from app.llm.cost_log import CostLog
from app.llm.providers.base import Provider
from app.llm.types import (
    Attempt,
    Completion,
    CompletionRequest,
    RequestRecord,
    StreamDone,
    StreamEvent,
    Target,
    TextDelta,
)


@dataclass(frozen=True)
class RetryPolicy:
    attempts_per_target: int = 3
    attempt_timeout_s: float = 20.0
    # Wall-clock budget for the whole request, retries and fallbacks included.
    # For streams it bounds the wait for the first token, not the stream length.
    deadline_s: float = 45.0
    backoff_base_s: float = 0.25
    backoff_cap_s: float = 4.0
    first_token_timeout_s: float = 10.0
    idle_timeout_s: float = 10.0


@dataclass(frozen=True)
class Result[T]:
    value: T
    completion: Completion
    record: RequestRecord


class ResilientLLM:
    def __init__(
        self,
        providers: dict[str, Provider],
        chain: list[Target],
        policy: RetryPolicy | None = None,
        log: CostLog | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not chain:
            raise ValueError("fallback chain is empty")
        missing = {t.provider for t in chain} - providers.keys()
        if missing:
            raise ValueError(f"chain names unknown providers: {sorted(missing)}")
        self.providers = providers
        self.chain = chain
        self.policy = policy or RetryPolicy()
        self.log = log
        self._sleep = sleep
        self._rng = rng or random.Random()

    async def complete(
        self, request: CompletionRequest, *, endpoint: str, request_id: str | None = None
    ) -> Result[str]:
        return await self._run(request, endpoint, request_id, lambda c: c.text)

    async def complete_structured[M: BaseModel](
        self,
        request: CompletionRequest,
        schema: type[M],
        *,
        endpoint: str,
        request_id: str | None = None,
    ) -> Result[M]:
        request = replace(request, json_schema=schema.model_json_schema())

        def parse(completion: Completion) -> M:
            try:
                return schema.model_validate_json(completion.text)
            except ValidationError as exc:
                raise errors.InvalidOutput(str(exc)) from exc

        return await self._run(request, endpoint, request_id, parse)

    async def _run[T](
        self,
        request: CompletionRequest,
        endpoint: str,
        request_id: str | None,
        parse: Callable[[Completion], T],
    ) -> Result[T]:
        record = RequestRecord(request_id=request_id or uuid.uuid4().hex, endpoint=endpoint)
        start = time.monotonic()
        last: errors.LLMError | None = None
        try:
            for target in self.chain:
                provider = self.providers[target.provider]
                for attempt_no in range(self.policy.attempts_per_target):
                    remaining = self.policy.deadline_s - (time.monotonic() - start)
                    if remaining <= 0:
                        raise errors.DeadlineExceeded("request deadline exceeded")
                    t0 = time.monotonic()
                    completion: Completion | None = None
                    try:
                        async with asyncio.timeout(min(self.policy.attempt_timeout_s, remaining)):
                            completion = await provider.complete(target.model, request)
                        value = parse(completion)
                    except TimeoutError:
                        last = errors.ProviderTimeout("attempt timed out")
                    except errors.LLMError as exc:
                        last = exc
                    else:
                        record.attempts.append(
                            Attempt(target, "ok", _ms(t0), usage=completion.usage)
                        )
                        record.outcome = "ok"
                        return Result(value, completion, record)

                    record.attempts.append(
                        Attempt(
                            target,
                            "error",
                            _ms(t0),
                            error_kind=last.kind,
                            usage=completion.usage if completion else None,
                            cost_known=not isinstance(last, errors.ProviderTimeout),
                        )
                    )
                    if not last.fallback_ok:
                        raise last
                    if time.monotonic() - start >= self.policy.deadline_s:
                        raise errors.DeadlineExceeded(f"deadline exceeded; last error: {last.kind}")
                    delay = self._next_delay(last, attempt_no)
                    if delay is None:
                        break
                    if time.monotonic() - start + delay >= self.policy.deadline_s:
                        break
                    await self._sleep(delay)
            assert last is not None
            raise errors.AllTargetsFailed(last)
        except errors.LLMError as exc:
            record.error_kind = exc.kind
            raise
        finally:
            record.latency_ms = _ms(start)
            if self.log:
                self.log.write(record)

    async def stream(
        self, request: CompletionRequest, *, endpoint: str, request_id: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Retries and fallback happen only before the first token reaches the
        caller. After that, output from a second model cannot be spliced onto the
        first, so a failure ends the stream with StreamInterrupted."""
        record = RequestRecord(request_id=request_id or uuid.uuid4().hex, endpoint=endpoint)
        start = time.monotonic()
        last: errors.LLMError | None = None
        started = False
        try:
            for target in self.chain:
                provider = self.providers[target.provider]
                for attempt_no in range(self.policy.attempts_per_target):
                    remaining = self.policy.deadline_s - (time.monotonic() - start)
                    if remaining <= 0:
                        raise errors.DeadlineExceeded("no first token before the deadline")
                    t0 = time.monotonic()
                    ttft: float | None = None
                    events = provider.stream(target.model, request)
                    try:
                        while True:
                            timeout = (
                                self.policy.idle_timeout_s
                                if started
                                else min(self.policy.first_token_timeout_s, remaining)
                            )
                            try:
                                async with asyncio.timeout(timeout):
                                    event = await anext(events)
                            except StopAsyncIteration:
                                raise errors.ConnectionFailed("stream ended early") from None
                            if isinstance(event, TextDelta):
                                if not started:
                                    started = True
                                    ttft = _ms(t0)
                                    record.ttft_ms = _ms(start)
                                yield event
                            elif isinstance(event, StreamDone):
                                record.attempts.append(
                                    Attempt(
                                        target,
                                        "ok",
                                        _ms(t0),
                                        usage=event.completion.usage,
                                        ttft_ms=ttft,
                                    )
                                )
                                record.outcome = "ok"
                                yield event
                                return
                    except TimeoutError:
                        last = errors.ProviderTimeout("stream timed out")
                    except errors.LLMError as exc:
                        last = exc
                    finally:
                        await events.aclose()

                    record.attempts.append(
                        Attempt(
                            target,
                            "error",
                            _ms(t0),
                            error_kind=last.kind,
                            ttft_ms=ttft,
                            # Tokens were generated and billed, but usage only
                            # arrives with the final event we never got.
                            cost_known=not (started or isinstance(last, errors.ProviderTimeout)),
                        )
                    )
                    if started:
                        raise errors.StreamInterrupted(f"{last.kind} after first token")
                    if not last.fallback_ok:
                        raise last
                    if time.monotonic() - start >= self.policy.deadline_s:
                        raise errors.DeadlineExceeded(f"deadline exceeded; last error: {last.kind}")
                    delay = self._next_delay(last, attempt_no)
                    if delay is None:
                        break
                    if time.monotonic() - start + delay >= self.policy.deadline_s:
                        break
                    await self._sleep(delay)
            assert last is not None
            raise errors.AllTargetsFailed(last)
        except errors.LLMError as exc:
            record.error_kind = exc.kind
            raise
        except (GeneratorExit, asyncio.CancelledError):
            if record.outcome != "ok":
                record.error_kind = "client_disconnected"
            raise
        finally:
            record.latency_ms = _ms(start)
            if self.log:
                self.log.write(record)

    def _next_delay(self, error: errors.LLMError, attempt_no: int) -> float | None:
        """Seconds to wait before retrying this target, or None to move to the next one."""
        if not error.retryable or attempt_no + 1 >= self.policy.attempts_per_target:
            return None
        if error.retry_after is not None:
            # A Retry-After longer than our cap means this target is out for a
            # while; the next target is a better bet than waiting.
            return error.retry_after if error.retry_after <= self.policy.backoff_cap_s else None
        ceiling = min(self.policy.backoff_cap_s, self.policy.backoff_base_s * 2**attempt_no)
        return self._rng.uniform(0, ceiling)


def _ms(since: float) -> float:
    return (time.monotonic() - since) * 1000
