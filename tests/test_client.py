from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from app.llm import errors
from app.llm.providers.simulated import SimulatedProvider, down, fail_calls
from app.llm.types import StreamDone, TextDelta
from tests.conftest import FAST, ask


async def test_healthy_primary_answers_in_one_attempt(build, log):
    primary = SimulatedProvider("a", reply="Your order ships today.")
    llm = build((primary, "sim-primary"))

    result = await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert result.value == "Your order ships today."
    assert [a.outcome for a in result.record.attempts] == ["ok"]
    row = log.get("r1")
    assert row["outcome"] == "ok" and row["attempts"] == 1 and row["fallback"] == 0


async def test_retries_transient_errors_with_capped_jittered_backoff(build, sleeps):
    primary = SimulatedProvider("a", fault=fail_calls(0, 1))
    llm = build((primary, "sim-primary"))

    result = await llm.complete(ask(), endpoint="answer")

    assert [a.error_kind for a in result.record.attempts] == ["overloaded", "overloaded", None]
    assert len(sleeps) == 2
    # Full jitter: attempt n waits in [0, min(cap, base * 2**n)].
    assert 0 <= sleeps[0] <= 0.1 and 0 <= sleeps[1] <= 0.2


async def test_falls_back_when_primary_exhausts_its_attempts(build, log):
    primary = SimulatedProvider("a", fault=down())
    secondary = SimulatedProvider("b", reply="from secondary")
    llm = build((primary, "sim-primary"), (secondary, "sim-secondary"))

    result = await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert result.value == "from secondary"
    assert primary.calls == FAST.attempts_per_target
    row = log.get("r1")
    assert (row["provider"], row["model"], row["fallback"]) == ("b", "sim-secondary", 1)


async def test_target_unavailable_skips_retries_and_falls_back(build, sleeps):
    primary = SimulatedProvider("a", fault=fail_calls(0, kind=errors.TargetUnavailable))
    secondary = SimulatedProvider("b")
    llm = build((primary, "sim-primary"), (secondary, "sim-secondary"))

    await llm.complete(ask(), endpoint="answer")

    assert primary.calls == 1 and secondary.calls == 1 and sleeps == []


async def test_bad_request_stops_the_chain(build, log):
    primary = SimulatedProvider("a", fault=fail_calls(0, kind=errors.BadRequest))
    secondary = SimulatedProvider("b")
    llm = build((primary, "sim-primary"), (secondary, "sim-secondary"))

    with pytest.raises(errors.BadRequest):
        await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert secondary.calls == 0
    row = log.get("r1")
    assert row["outcome"] == "error" and row["error_kind"] == "bad_request"


async def test_retry_after_within_cap_is_honored(build, sleeps):
    primary = SimulatedProvider(
        "a", fault=lambda call, _rng: errors.RateLimited(retry_after=0.7) if call == 0 else None
    )
    llm = build((primary, "sim-primary"))

    await llm.complete(ask(), endpoint="answer")

    assert sleeps == [0.7]


async def test_retry_after_beyond_cap_moves_to_next_target(build, sleeps):
    primary = SimulatedProvider("a", fault=lambda _c, _r: errors.RateLimited(retry_after=30))
    secondary = SimulatedProvider("b")
    llm = build((primary, "sim-primary"), (secondary, "sim-secondary"))

    result = await llm.complete(ask(), endpoint="answer")

    assert primary.calls == 1 and sleeps == []
    assert result.completion.provider == "b"


async def test_all_targets_failing_raises_with_last_error(build, log):
    llm = build((SimulatedProvider("a", fault=down()), "sim-primary"))

    with pytest.raises(errors.AllTargetsFailed) as exc:
        await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert exc.value.last.kind == "overloaded"
    assert log.get("r1")["attempts"] == 3


async def test_slow_attempt_times_out_and_its_cost_is_unknown(build, log):
    slow = SimulatedProvider("a", latency_s=2.0)
    fast = SimulatedProvider("b")
    llm = build((slow, "sim-primary"), (fast, "sim-secondary"))

    result = await llm.complete(ask(), endpoint="answer", request_id="r1")

    assert result.completion.provider == "b"
    assert {a.error_kind for a in result.record.attempts[:-1]} == {"timeout"}
    row = log.get("r1")
    # The timed-out attempts may have been billed; the total is only a floor.
    assert row["cost_complete"] == 0


async def test_deadline_bounds_the_whole_request(build):
    policy = FAST.__class__(**{**FAST.__dict__, "attempt_timeout_s": 0.2, "deadline_s": 0.5})
    llm = build(
        (SimulatedProvider("a", latency_s=1.0), "sim-primary"),
        (SimulatedProvider("b", latency_s=1.0), "sim-secondary"),
        policy=policy,
    )

    with pytest.raises(errors.DeadlineExceeded):
        await llm.complete(ask(), endpoint="answer")


async def test_unpriced_model_has_unknown_cost_not_zero(build, log):
    llm = build((SimulatedProvider("a"), "model-nobody-priced"))

    await llm.complete(ask(), endpoint="answer", request_id="r1")

    row = log.get("r1")
    assert row["cost_usd"] is None and row["cost_complete"] == 0


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    score: int


async def test_structured_output_is_validated(build):
    llm = build((SimulatedProvider("a"), "sim-primary"))

    result = await llm.complete_structured(ask(), Verdict, endpoint="triage")

    assert result.value == Verdict(label="x", score=0)


async def test_invalid_structured_output_retries_then_falls_back(build, log):
    broken = SimulatedProvider("a", reply="not json")
    broken._text = lambda _request: "not json"  # ignores the schema, like a weak model
    good = SimulatedProvider("b")
    llm = build((broken, "sim-primary"), (good, "sim-secondary"))

    result = await llm.complete_structured(ask(), Verdict, endpoint="triage", request_id="r1")

    assert result.completion.provider == "b"
    assert [a.error_kind for a in result.record.attempts] == ["invalid_output"] * 3 + [None]
    # Invalid answers still consumed tokens, and the log charges for them.
    assert result.record.attempts[0].usage is not None
    assert log.get("r1")["cost_complete"] == 1


async def collect(stream):
    return [event async for event in stream]


async def test_stream_falls_back_before_first_token(build, log):
    llm = build(
        (SimulatedProvider("a", fault=down()), "sim-primary"),
        (SimulatedProvider("b", reply="hello there"), "sim-secondary"),
    )

    events = await collect(llm.stream(ask(), endpoint="answer", request_id="r1"))

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "hello there"
    assert isinstance(events[-1], StreamDone)
    row = log.get("r1")
    assert row["outcome"] == "ok" and row["fallback"] == 1 and row["ttft_ms"] is not None


async def test_stream_failure_after_first_token_is_not_retried(build, log):
    flaky = SimulatedProvider("a", reply="one two three", fail_stream_after=2)
    secondary = SimulatedProvider("b")
    llm = build((flaky, "sim-primary"), (secondary, "sim-secondary"))

    seen: list[str] = []
    with pytest.raises(errors.StreamInterrupted):
        async for event in llm.stream(ask(), endpoint="answer", request_id="r1"):
            if isinstance(event, TextDelta):
                seen.append(event.text)

    assert "".join(seen) == "one two"
    assert secondary.calls == 0
    row = log.get("r1")
    assert row["error_kind"] == "stream_interrupted" and row["cost_complete"] == 0


async def test_stream_first_token_timeout_falls_back(build):
    llm = build(
        (SimulatedProvider("a", latency_s=2.0), "sim-primary"),
        (SimulatedProvider("b", reply="ok"), "sim-secondary"),
    )

    events = await collect(llm.stream(ask(), endpoint="answer"))

    assert events[-1].completion.provider == "b"
