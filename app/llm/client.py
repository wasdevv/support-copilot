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
from app.llm.resilience import BreakerPolicy, BudgetPolicy, CircuitBreaker, RetryBudget
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

DEFAULT_BREAKER = BreakerPolicy()
DEFAULT_BUDGET = BudgetPolicy()


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
    # Each of these is off when None, which reproduces v1.0 behaviour.
    breaker: BreakerPolicy | None = DEFAULT_BREAKER
    budget: BudgetPolicy | None = DEFAULT_BUDGET
    # Send a copy to the next target when the first attempt is slower than this.
    # Set it near the primary's normal p95: hedging the slowest ~5% costs ~5% more calls.
    hedge_after_s: float | None = 4.0


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
        clock: Callable[[], float] = time.monotonic,
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
        breaker = self.policy.breaker
        self.breakers = {
            str(t): CircuitBreaker(breaker, clock) for t in chain if breaker is not None
        }
        self.budget = RetryBudget(self.policy.budget) if self.policy.budget else None

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
        if self.budget:
            self.budget.deposit()
        try:
            for index, target in enumerate(self.chain):
                if not self._allow(target):
                    record.breaker_skips += 1
                    continue
                for attempt_no in range(self.policy.attempts_per_target):
                    remaining = self.policy.deadline_s - (time.monotonic() - start)
                    if remaining <= 0:
                        raise errors.DeadlineExceeded("request deadline exceeded")
                    timeout = min(self.policy.attempt_timeout_s, remaining)
                    backup = self._hedge_target(index) if attempt_no == 0 else None
                    try:
                        if backup is None:
                            value, completion = await self._attempt(
                                target, request, parse, timeout, record
                            )
                        else:
                            value, completion = await self._hedged(
                                target, backup, request, parse, timeout, record
                            )
                    except errors.LLMError as exc:
                        last = exc
                    else:
                        record.outcome = "ok"
                        return Result(value, completion, record)

                    if not last.fallback_ok:
                        raise last
                    if time.monotonic() - start >= self.policy.deadline_s:
                        raise errors.DeadlineExceeded(f"deadline exceeded; last error: {last.kind}")
                    if not self._may_retry(target):
                        break
                    delay = self._next_delay(last, attempt_no)
                    if delay is None or time.monotonic() - start + delay >= self.policy.deadline_s:
                        break
                    if not self._retry_allowed(index):
                        break
                    await self._sleep(delay)
            if last is None:
                # Panic mode (Envoy's term): every breaker is open. A breaker on
                # the last resort can open on noise, and honoring all of them
                # would turn a partial outage into a total one. Try each target
                # once; only the last target keeps its retries (see _panic_attempts).
                record.panic = True
                for index, target in enumerate(self.chain):
                    for attempt_no in range(self._panic_attempts(index)):
                        remaining = self.policy.deadline_s - (time.monotonic() - start)
                        if remaining <= 0:
                            break
                        timeout = min(self.policy.attempt_timeout_s, remaining)
                        try:
                            value, completion = await self._attempt(
                                target, request, parse, timeout, record
                            )
                        except errors.LLMError as exc:
                            last = exc
                            if not exc.fallback_ok:
                                raise
                        else:
                            record.outcome = "ok"
                            return Result(value, completion, record)
                        delay = self._next_delay(last, attempt_no)
                        if (
                            delay is None
                            or time.monotonic() - start + delay >= self.policy.deadline_s
                        ):
                            break
                        await self._sleep(delay)
            if last is None:
                raise errors.DeadlineExceeded("request deadline exceeded")
            raise errors.AllTargetsFailed(last)
        except errors.LLMError as exc:
            record.error_kind = exc.kind
            raise
        finally:
            record.latency_ms = _ms(start)
            if self.log:
                self.log.write(record)

    async def _attempt[T](
        self,
        target: Target,
        request: CompletionRequest,
        parse: Callable[[Completion], T],
        timeout: float,
        record: RequestRecord,
    ) -> tuple[T, Completion]:
        """One call to one target. Records the attempt and feeds its breaker."""
        provider = self.providers[target.provider]
        breaker = self.breakers.get(str(target))
        t0 = time.monotonic()
        completion: Completion | None = None
        try:
            async with asyncio.timeout(timeout):
                completion = await provider.complete(target.model, request)
            value = parse(completion)
        except TimeoutError:
            error: errors.LLMError = errors.ProviderTimeout("attempt timed out")
        except errors.LLMError as exc:
            error = exc
        except asyncio.CancelledError:
            # Lost a hedge race. The provider may already be generating, so the
            # cost of this call is unknown, and it proves nothing about health.
            record.attempts.append(
                Attempt(target, "error", _ms(t0), error_kind="cancelled", cost_known=False)
            )
            if breaker:
                breaker.release()
            raise
        else:
            record.attempts.append(Attempt(target, "ok", _ms(t0), usage=completion.usage))
            if breaker:
                breaker.record_success()
            return value, completion

        record.attempts.append(
            Attempt(
                target,
                "error",
                _ms(t0),
                error_kind=error.kind,
                usage=completion.usage if completion else None,
                cost_known=not isinstance(error, errors.ProviderTimeout),
            )
        )
        if breaker:
            breaker.record_failure(error)
        raise error

    async def _hedged[T](
        self,
        primary: Target,
        backup: Target,
        request: CompletionRequest,
        parse: Callable[[Completion], T],
        timeout: float,
        record: RequestRecord,
    ) -> tuple[T, Completion]:
        """Race the primary against a delayed copy on the backup; first success wins."""
        assert self.policy.hedge_after_s is not None
        first = asyncio.create_task(self._attempt(primary, request, parse, timeout, record))
        done, _ = await asyncio.wait({first}, timeout=self.policy.hedge_after_s)
        if done or (self.budget and not self.budget.withdraw()):
            return await first
        record.hedged = True
        backup_timeout = max(0.0, timeout - self.policy.hedge_after_s)
        second = asyncio.create_task(self._attempt(backup, request, parse, backup_timeout, record))
        pending = {first, second}
        failure: BaseException | None = None
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    for loser in pending:
                        loser.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    return task.result()
                failure = task.exception()
        assert failure is not None
        raise failure

    def _panic_attempts(self, index: int) -> int:
        """In panic mode, one call per target bounds the cost of ignoring breakers.
        The last target keeps its retries: cutting them turns its normal failure
        rate (a CPU model's slow tail) straight into user-facing failures."""
        return self.policy.attempts_per_target if index == len(self.chain) - 1 else 1

    def _retry_allowed(self, index: int) -> bool:
        """The budget turns a retry into a fallback, never into a failure: on the
        last target there is nothing to fall back to, so the retry is not gated."""
        if self.budget is None or index == len(self.chain) - 1:
            return True
        return self.budget.withdraw()

    def _allow(self, target: Target) -> bool:
        breaker = self.breakers.get(str(target))
        return breaker is None or breaker.allow()

    def _may_retry(self, target: Target) -> bool:
        """A breaker that opened during this request also stops its retries."""
        breaker = self.breakers.get(str(target))
        return breaker is None or breaker.state == "closed"

    def _hedge_target(self, index: int) -> Target | None:
        if self.policy.hedge_after_s is None:
            return None
        for target in self.chain[index + 1 :]:
            breaker = self.breakers.get(str(target))
            # Only a closed breaker: a half-open target's single probe slot is
            # not spent on a speculative copy.
            if breaker is None or breaker.state == "closed":
                return target
        return None

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
        if self.budget:
            self.budget.deposit()
        try:
            # Second pass only in panic mode: every breaker was open (see _run).
            for panic in (False, True):
                if panic:
                    if last is not None:
                        break
                    record.panic = True
                for target in self.chain:
                    provider = self.providers[target.provider]
                    breaker = self.breakers.get(str(target))
                    if not panic and not self._allow(target):
                        record.breaker_skips += 1
                        continue
                    attempts = (
                        self._panic_attempts(self.chain.index(target))
                        if panic
                        else self.policy.attempts_per_target
                    )
                    for attempt_no in range(attempts):
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
                                    if breaker:
                                        breaker.record_success()
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
                                cost_known=not (
                                    started or isinstance(last, errors.ProviderTimeout)
                                ),
                            )
                        )
                        if breaker:
                            breaker.record_failure(last)
                        if started:
                            raise errors.StreamInterrupted(f"{last.kind} after first token")
                        if not last.fallback_ok:
                            raise last
                        if time.monotonic() - start >= self.policy.deadline_s:
                            raise errors.DeadlineExceeded(
                                f"deadline exceeded; last error: {last.kind}"
                            )
                        if not self._may_retry(target):
                            break
                        delay = self._next_delay(last, attempt_no)
                        if (
                            delay is None
                            or time.monotonic() - start + delay >= self.policy.deadline_s
                        ):
                            break
                        if not self._retry_allowed(self.chain.index(target)):
                            break
                        await self._sleep(delay)
            if last is None:
                raise errors.DeadlineExceeded("no first token before the deadline")
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
