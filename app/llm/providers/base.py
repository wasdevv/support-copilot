from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.llm.types import Completion, CompletionRequest, StreamEvent


class Provider(Protocol):
    """A provider makes exactly one attempt. Timeouts, retries and fallback belong
    to the client, so implementations must not retry on their own."""

    name: str

    async def complete(self, model: str, request: CompletionRequest) -> Completion: ...

    def stream(self, model: str, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
