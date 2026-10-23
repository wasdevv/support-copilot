"""State that outlives a single request: per-target circuit breakers and a retry budget.

Both take an injected clock and do no I/O, so tests drive time directly.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.llm import errors


def is_availability_failure(error: errors.LLMError) -> bool:
    """Failures that say the target is unhealthy. A bad request, an invalid answer
    or a refusal says nothing about whether the target is up."""
    if isinstance(error, errors.TargetUnavailable):
        return True
    return error.retryable and not isinstance(error, errors.InvalidOutput)


@dataclass(frozen=True)
class BreakerPolicy:
    # Opens when at least `failure_rate` of the last `window` calls failed, once
    # `min_calls` have been seen (window 100 is Resilience4j's default). Outcomes
    # land in completion order, and under concurrency timeouts all land together,
    # one attempt_timeout after the burst that caused them. A window smaller than
    # the concurrency then holds nothing but those timeouts and opens on a target
    # whose real failure rate is 27%.
    window: int = 100
    min_calls: int = 20
    failure_rate: float = 0.5
    open_s: float = 30.0


class CircuitBreaker:
    """closed -> open when the windowed failure rate crosses the threshold -> half-open
    after open_s, which lets exactly one probe through -> closed on success, open on failure."""

    def __init__(self, policy: BreakerPolicy, clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self._clock = clock
        self._outcomes: deque[bool] = deque(maxlen=policy.window)  # True = failed
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.policy.open_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """True when a call may go out. In half-open, only the first caller gets
        through, and it becomes the probe."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def record_success(self) -> None:
        if self._opened_at is not None:
            # A successful probe closes the breaker with a clean window.
            if not self._probing:
                return
            self._opened_at = None
            self._probing = False
            self._outcomes.clear()
        self._outcomes.append(False)

    def record_failure(self, error: errors.LLMError) -> None:
        if not is_availability_failure(error):
            # The call reached a healthy target.
            self.record_success()
            return
        if self._opened_at is not None:
            if self._probing:
                self._open()
            return
        self._outcomes.append(True)
        calls = len(self._outcomes)
        if (
            calls >= self.policy.min_calls
            and sum(self._outcomes) / calls >= self.policy.failure_rate
        ):
            self._open()

    def release(self) -> None:
        """A call that ended without a verdict (cancelled hedge loser) frees the probe slot."""
        self._probing = False

    def _open(self) -> None:
        self._opened_at = self._clock()
        self._probing = False
        self._outcomes.clear()


@dataclass(frozen=True)
class BudgetPolicy:
    # Each request earns `ratio` retry tokens: 0.1 caps retries + hedges at ~10%
    # of traffic once the floor is spent.
    ratio: float = 0.1
    # Tokens available from the start, so low traffic can still retry.
    min_tokens: float = 10.0
    max_tokens: float = 100.0


class RetryBudget:
    """Token bucket over retries, as in gRPC retry throttling and Finagle's RetryBudget.
    Under overload, retries stop multiplying load on a provider that is already failing."""

    def __init__(self, policy: BudgetPolicy) -> None:
        self.policy = policy
        self._tokens = policy.min_tokens

    @property
    def tokens(self) -> float:
        return self._tokens

    def deposit(self) -> None:
        self._tokens = min(self.policy.max_tokens, self._tokens + self.policy.ratio)

    def withdraw(self) -> bool:
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True
