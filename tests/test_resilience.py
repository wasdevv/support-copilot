from __future__ import annotations

import pytest

from app.llm import errors
from app.llm.client import RetryPolicy
from app.llm.providers.simulated import SimulatedProvider, down, fail_calls
from app.llm.resilience import BreakerPolicy, BudgetPolicy, CircuitBreaker, RetryBudget
from tests.conftest import FAST, ask


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def policy(**overrides) -> RetryPolicy:
    return RetryPolicy(**{**FAST.__dict__, **overrides})


# --- CircuitBreaker ---------------------------------------------------------


def test_breaker_opens_after_threshold_and_half_opens_with_one_probe():
    clock = Clock()
    breaker = CircuitBreaker(BreakerPolicy(window=3, min_calls=3, open_s=10), clock)

    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure(errors.ProviderOverloaded())

    assert breaker.state == "open" and not breaker.allow()
    clock.now = 10
    assert breaker.state == "half_open"
    assert breaker.allow()  # the probe
    assert not breaker.allow()  # everyone else waits for its verdict


def test_probe_success_closes_and_probe_failure_reopens():
    clock = Clock()
    breaker = CircuitBreaker(BreakerPolicy(window=1, min_calls=1, open_s=10), clock)
    breaker.record_failure(errors.ProviderTimeout())
    clock.now = 10
    assert breaker.allow()
    breaker.record_failure(errors.ProviderTimeout())
    assert breaker.state == "open"

    clock.now = 20
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()


@pytest.mark.parametrize("error", [errors.BadRequest(), errors.InvalidOutput(), errors.Refused()])
def test_non_availability_failures_do_not_trip_the_breaker(error):
    breaker = CircuitBreaker(BreakerPolicy(window=1, min_calls=1))

    breaker.record_failure(error)

    assert breaker.state == "closed"


def test_failure_rate_below_threshold_stays_closed():
    # A last-resort model with a normal slow tail must not trip its own breaker.
    breaker = CircuitBreaker(BreakerPolicy(window=10, min_calls=4, failure_rate=0.5))

    for failed in [False, True, False, False, True, False, False]:
        if failed:
            breaker.record_failure(errors.ProviderTimeout())
        else:
            breaker.record_success()

    assert breaker.state == "closed"


def test_breaker_waits_for_min_calls_before_judging():
    breaker = CircuitBreaker(BreakerPolicy(window=10, min_calls=5))

    for _ in range(4):
        breaker.record_failure(errors.ProviderOverloaded())

    assert breaker.state == "closed"
    breaker.record_failure(errors.ProviderOverloaded())
    assert breaker.state == "open"


# --- RetryBudget ------------------------------------------------------------


def test_budget_spends_the_floor_then_earns_a_ratio_per_request():
    budget = RetryBudget(BudgetPolicy(ratio=0.5, min_tokens=1, max_tokens=10))

    assert budget.withdraw()
    assert not budget.withdraw()
    budget.deposit()
    assert not budget.withdraw()
    budget.deposit()
    assert budget.withdraw()


# --- client wiring ----------------------------------------------------------


async def test_open_breaker_skips_the_dead_target_without_calling_it(build, log):
    primary = SimulatedProvider("a", fault=down())
    secondary = SimulatedProvider("b")
    llm = build(
        (primary, "sim-primary"),
        (secondary, "sim-secondary"),
        policy=policy(breaker=BreakerPolicy(window=3, min_calls=3, open_s=60)),
    )

    await llm.complete(ask(), endpoint="answer")  # 3 failures open the breaker
    calls_after_first = primary.calls
    result = await llm.complete(ask(), endpoint="answer", request_id="r2")

    assert calls_after_first == 3 and primary.calls == 3
    assert [a.target.provider for a in result.record.attempts] == ["b"]
    assert log.get("r2")["breaker_skips"] == 1


async def test_all_breakers_open_enters_panic_mode_with_one_call_per_target(build, log):
    a = SimulatedProvider("a", fault=down())
    b = SimulatedProvider("b", fault=down())
    llm = build(
        (a, "sim-primary"),
        (b, "sim-secondary"),
        policy=policy(breaker=BreakerPolicy(window=3, min_calls=3, open_s=60)),
    )
    with pytest.raises(errors.AllTargetsFailed):
        await llm.complete(ask(), endpoint="answer")
    assert llm.breakers["a/sim-primary"].state == "open"
    assert llm.breakers["b/sim-secondary"].state == "open"

    with pytest.raises(errors.AllTargetsFailed):
        await llm.complete(ask(), endpoint="answer", request_id="r2")

    row = log.get("r2")
    # One call to the first target, full retries on the last.
    assert row["panic"] == 1 and row["attempts"] == 1 + 3 and row["breaker_skips"] == 2


async def test_panic_mode_serves_from_a_recovered_last_resort(build):
    a = SimulatedProvider("a", fault=down())
    local = SimulatedProvider("b", fault=fail_calls(0, 1, 2))
    llm = build(
        (a, "sim-primary"),
        (local, "sim-local"),
        policy=policy(breaker=BreakerPolicy(window=3, min_calls=3, open_s=60)),
    )
    with pytest.raises(errors.AllTargetsFailed):
        await llm.complete(ask(), endpoint="answer")

    result = await llm.complete(ask(), endpoint="answer")

    assert result.completion.provider == "b" and result.record.panic


async def test_stream_panic_mode_tries_each_target_once(build):
    a = SimulatedProvider("a", fault=down())
    b = SimulatedProvider("b", fault=fail_calls(0, 1, 2), reply="back up")
    llm = build(
        (a, "sim-primary"),
        (b, "sim-secondary"),
        policy=policy(breaker=BreakerPolicy(window=3, min_calls=3, open_s=60)),
    )
    with pytest.raises(errors.AllTargetsFailed):
        [e async for e in llm.stream(ask(), endpoint="answer")]

    events = [e async for e in llm.stream(ask(), endpoint="answer")]

    assert events[-1].completion.text == "back up"
    assert a.calls == 4 and b.calls == 4  # 3 to open each breaker, then 1 in panic


async def test_breaker_opening_mid_request_stops_its_retries(build):
    a = SimulatedProvider("a", fault=down())
    b = SimulatedProvider("b")
    llm = build(
        (a, "sim-primary"),
        (b, "sim-secondary"),
        policy=policy(breaker=BreakerPolicy(window=1, min_calls=1)),
    )

    await llm.complete(ask(), endpoint="answer")

    assert a.calls == 1 and b.calls == 1


async def test_empty_budget_skips_retries_and_falls_back(build, sleeps):
    a = SimulatedProvider("a", fault=fail_calls(0))
    b = SimulatedProvider("b")
    llm = build(
        (a, "sim-primary"),
        (b, "sim-secondary"),
        policy=policy(budget=BudgetPolicy(ratio=0, min_tokens=0)),
    )

    result = await llm.complete(ask(), endpoint="answer")

    assert a.calls == 1 and sleeps == []
    assert result.completion.provider == "b"


async def test_hedge_fires_after_delay_and_fastest_success_wins(build, log):
    slow = SimulatedProvider("a", latency_s=0.4, reply="slow")
    fast = SimulatedProvider("b", latency_s=0.01, reply="fast")
    llm = build((slow, "sim-primary"), (fast, "sim-secondary"), policy=policy(hedge_after_s=0.05))

    result = await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert result.value == "fast"
    kinds = {a.target.provider: a.error_kind for a in result.record.attempts}
    assert kinds == {"a": "cancelled", "b": None}
    row = log.get("r1")
    assert row["hedged"] == 1 and row["attempts"] == 2
    # The cancelled call may have been billed: the total is a floor.
    assert row["cost_complete"] == 0
    assert row["latency_ms"] < 300


async def test_no_hedge_when_primary_answers_in_time(build, log):
    a = SimulatedProvider("a", latency_s=0.01)
    b = SimulatedProvider("b")
    llm = build((a, "sim-primary"), (b, "sim-secondary"), policy=policy(hedge_after_s=0.2))

    await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert b.calls == 0 and log.get("r1")["hedged"] == 0


async def test_no_hedge_when_budget_is_empty(build):
    a = SimulatedProvider("a", latency_s=0.1, reply="primary")
    b = SimulatedProvider("b", reply="backup")
    llm = build(
        (a, "sim-primary"),
        (b, "sim-secondary"),
        policy=policy(hedge_after_s=0.02, budget=BudgetPolicy(ratio=0, min_tokens=0)),
    )

    result = await llm.complete(ask(), endpoint="answer")

    assert result.value == "primary" and b.calls == 0


async def test_hedge_where_primary_fails_still_takes_the_backup(build):
    a = SimulatedProvider("a", latency_s=0.1, fault=down())
    b = SimulatedProvider("b", latency_s=0.2, reply="backup")
    llm = build((a, "sim-primary"), (b, "sim-secondary"), policy=policy(hedge_after_s=0.02))

    result = await llm.complete(ask(), endpoint="answer")

    assert result.value == "backup"


async def test_empty_budget_still_retries_the_last_target(build, sleeps):
    # A denied retry must become a fallback; on the last target it would be a failure.
    only = SimulatedProvider("a", fault=fail_calls(0))
    llm = build((only, "sim-primary"), policy=policy(budget=BudgetPolicy(ratio=0, min_tokens=0)))

    result = await llm.complete(ask(), endpoint="answer")

    assert only.calls == 2 and len(sleeps) == 1
    assert result.record.outcome == "ok"


async def test_breaker_opening_mid_request_does_not_cut_last_target_retries(build):
    only = SimulatedProvider("a", fault=fail_calls(0))
    llm = build(
        (only, "sim-primary"),
        policy=policy(breaker=BreakerPolicy(window=1, min_calls=1, open_s=60)),
    )

    result = await llm.complete(ask(), endpoint="answer")

    assert result.record.outcome == "ok" and only.calls == 2
