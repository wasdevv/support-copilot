from __future__ import annotations

import random

import pytest

from app.llm.client import ResilientLLM, RetryPolicy
from app.llm.cost_log import CostLog
from app.llm.providers.simulated import SimulatedProvider
from app.llm.types import CompletionRequest, Message, Target

FAST = RetryPolicy(
    attempts_per_target=3,
    attempt_timeout_s=0.5,
    deadline_s=5.0,
    backoff_base_s=0.1,
    backoff_cap_s=1.0,
    first_token_timeout_s=0.5,
    idle_timeout_s=0.5,
    hedge_after_s=None,
)


class Sleeps(list[float]):
    """Records backoff sleeps instead of waiting them out."""

    async def __call__(self, seconds: float) -> None:
        self.append(seconds)


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
def log() -> CostLog:
    return CostLog()


@pytest.fixture
def build(sleeps: Sleeps, log: CostLog):
    def _build(*providers: tuple[SimulatedProvider, str], policy: RetryPolicy = FAST):
        return ResilientLLM(
            {p.name: p for p, _ in providers},
            [Target(p.name, model) for p, model in providers],
            policy,
            log,
            sleep=sleeps,
            rng=random.Random(0),
        )

    return _build


def ask(text: str = "Where is my order?") -> CompletionRequest:
    return CompletionRequest(messages=[Message("user", text)])
