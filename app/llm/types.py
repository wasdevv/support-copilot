from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    # JSON Schema the output must satisfy. Providers pass it to their native
    # constrained-decoding option; the client still validates the result.
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    usage: Usage
    stop_reason: str | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class StreamDone:
    completion: Completion


StreamEvent = TextDelta | StreamDone


@dataclass(frozen=True)
class Target:
    """One link of the fallback chain: which provider serves which model."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class Attempt:
    target: Target
    outcome: Literal["ok", "error"]
    latency_ms: float
    error_kind: str | None = None
    usage: Usage | None = None
    ttft_ms: float | None = None
    # A timed-out request may have been processed and billed by the provider;
    # we cannot know, so the cost of that attempt is unknown rather than zero.
    cost_known: bool = True


@dataclass
class RequestRecord:
    request_id: str
    endpoint: str
    attempts: list[Attempt] = field(default_factory=list)
    outcome: Literal["ok", "error"] = "error"
    error_kind: str | None = None
    latency_ms: float = 0.0
    ttft_ms: float | None = None
