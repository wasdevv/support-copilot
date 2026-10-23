"""Provider-neutral error taxonomy.

Each provider maps its own failures onto these classes, and the client decides
what to do from two flags only:

- ``retryable``: trying the same target again may succeed (timeouts, 429, 5xx).
- ``fallback_ok``: another target may succeed where this one cannot. False only
  for errors caused by the request itself (400): the same payload fails anywhere.
"""

from __future__ import annotations


class LLMError(Exception):
    kind = "error"
    retryable = False
    fallback_ok = True

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message or self.kind)
        self.retry_after = retry_after


class ProviderTimeout(LLMError):
    kind = "timeout"
    retryable = True


class ConnectionFailed(LLMError):
    kind = "connection"
    retryable = True


class RateLimited(LLMError):
    kind = "rate_limited"
    retryable = True


class ProviderOverloaded(LLMError):
    """5xx and Anthropic's 529."""

    kind = "overloaded"
    retryable = True


class InvalidOutput(LLMError):
    """The model answered, but not in the requested schema."""

    kind = "invalid_output"
    retryable = True


class TargetUnavailable(LLMError):
    """Auth failure, unknown model, permission: this target cannot serve, others may."""

    kind = "target_unavailable"


class Refused(LLMError):
    """The model declined. Asking the same model again rarely changes that."""

    kind = "refused"


class BadRequest(LLMError):
    kind = "bad_request"
    fallback_ok = False


class DeadlineExceeded(LLMError):
    kind = "deadline_exceeded"
    fallback_ok = False


class AllTargetsFailed(LLMError):
    kind = "all_targets_failed"
    fallback_ok = False

    def __init__(self, last: LLMError) -> None:
        super().__init__(f"all targets failed; last error: {last.kind}: {last}")
        self.last = last


class StreamInterrupted(LLMError):
    """The stream failed after tokens reached the caller; it cannot be retried."""

    kind = "stream_interrupted"
    fallback_ok = False
