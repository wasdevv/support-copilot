# PRD — Stage 1.1: circuit breakers, hedged requests, retry budget

Follow-up to `stage-1-reliable-llm-client.md`. Same module (`app/llm/`), same
benchmark, plus one harder scenario.

## Problema

v1.0 measured three weaknesses of its own:

1. **Wasted attempts on a dead model.** In a full primary outage every request
   spends 3 attempts on the primary before falling back (4.05 attempts per
   request, p95 296 ms in the scaled benchmark). The client has no memory across
   requests.
2. **Slow tail.** When 30% of primary calls are slow, p95 is 735 ms: a request
   waits the full attempt timeout before anything else is tried.
3. **Retry amplification.** Under overload, every failing request retries up to
   3 times per target, multiplying load on a provider that is already failing.
   In a total outage the client spends 9 attempts and 1.3 s per request before failing.

The benchmark also draws faults independently per call, which is the easy case.
Real outages come in bursts.

## Requisitos

1. **Circuit breaker per target** (closed → open → half-open):
   - opens when the failure rate over a sliding window of recent calls reaches
     `failure_rate`, once `min_calls` were seen. Availability failures are
     timeout, connection, 429, 5xx and target unavailable. Invalid output,
     refusals and bad requests are not availability failures and do not count.
   - while open, the target is skipped without a call; after `open_s` it lets
     **one** probe through (half-open). Success closes it, failure re-opens it.
   - when every target is open, the request fails fast with `circuit_open`.
2. **Hedged requests** (non-streaming only): if the first attempt on a target has
   not answered after `hedge_after_s`, send the same request to the next
   available target and take the first success. The loser is cancelled. Its
   cost is unknown, because a cancelled call may still be billed, so the request
   is marked `cost_complete = false`.
3. **Retry budget** (token bucket, as in gRPC retry throttling): each request
   deposits `ratio` tokens, each retry or hedge withdraws one, and there is a
   floor of `min_tokens`. With no token, the client skips the retry and moves to the next
   target, or skips the hedge.
4. Each feature is optional in `RetryPolicy`, so v1.0 behaviour is reproducible
   with the new fields set to `None`.
5. **Benchmark**: compare three clients per scenario: naive, v1.0 and v1.1.
   Report failure rate, p95 latency, attempts per request, and amplification
   (provider calls per request). Add a `burst_outages` scenario in which the primary
   and the secondary go down in time windows, not per call.
6. The log records breaker skips and hedges per request.

## Critérios de aceite

- [ ] Tests: the breaker opens after N failures and skips the target with no
      call; it half-opens after `open_s` with a single probe; probe success
      closes it and probe failure re-opens it; bad request and invalid output do
      not trip it; all open → `circuit_open` without calling any provider.
- [ ] Tests: a hedge fires only after `hedge_after_s`, the fastest success wins,
      the loser is cancelled and logged with unknown cost; no hedge when the
      primary answers in time; no hedge when the budget is empty.
- [ ] Tests: an empty retry budget skips retries and falls back.
- [ ] Benchmark table with naive / v1.0 / v1.1 for every scenario, fixed seed,
      reproduced in CI; README and CHANGELOG quote only measured numbers.
- [ ] Existing 40 tests still pass; ruff clean.

## Fora de escopo

- Hedging streams (the second stream would need to be consumed and discarded;
  revisit if TTFT tail matters in v3).
- Breaker state shared across processes (single process by design).
- Adaptive hedge delay from live latency percentiles (fixed delay for now).

## Arquitetura e decisões

- `app/llm/resilience.py`: `CircuitBreaker` and `RetryBudget`, both with an
  injected clock and no I/O. The client owns one breaker per target and one budget.
- `RetryPolicy` gains `breaker: BreakerPolicy | None`, `hedge_after_s: float | None`
  and `budget: BudgetPolicy | None`. The defaults turn all three on.
- Hedging is done in `_hedged_attempt()`: race the primary attempt against
  `sleep(hedge_after_s)` and then a backup attempt, using `asyncio.wait(FIRST_COMPLETED)`.
- Breaker bookkeeping happens in the attempt path, so retries, hedges and
  fallbacks all feed it.
- `burst_outages` fault: down for `down_s` out of every `period_s`, each
  provider with its own phase, so windows partly overlap.

## Riscos

- **Hedging costs money.** Every hedge is a second call. The benchmark reports
  amplification so the latency gain is shown next to its price.
- **A breaker that is too eager** drops a healthy model after a few unlucky errors. Threshold 5,
  with a short open window, and the half-open probe recovers it.
- **Benchmark tuning.** The fault model of the five v1.0 scenarios is unchanged.
  The new scenario is added, not swapped in.

## Found during implementation

Three versions of v1.1 were measured before this one. Each was worse than v1.0
somewhere, and each failure had a structural cause:

1. **Breaker that counted consecutive failures (5):** v1.1 failed 94% in
   `cloud_outage`. The last-resort local model has a 25% slow tail, and under
   concurrency its breaker kept opening, so every request failed fast.
2. **Windowed failure rate (20 calls) plus fail-fast when all breakers were
   open:** still 26%. Outcomes are recorded in completion order. Timeouts from a
   burst of 100 concurrent requests all land together, 300 ms after the burst,
   so a 20-call window held only timeouts. Fixed with a window of 100
   (Resilience4j's default) and min 20 calls.
3. **Fail-fast when every breaker is open** turns a partial outage into a total
   one if the last resort opened on noise. It was replaced by **panic mode**
   (Envoy's term): when every breaker is open, ignore them and try each target
   once.
4. **The retry budget starved the last target.** A denied retry is meant to
   become a fallback, but on the last target it became a failure (+2 pp in
   `rate_limit_storm`). Rule: the budget does not gate the last target in the
   chain.

5. **Panic mode with one call per target** let `cloud_outage` reach 5.2% on
   the slower CI runner, against 1.8% for v1.0, whenever the local breaker opened.
   The last target keeps its retries in panic mode too, the same rule as the budget.

6. **The breaker cut the last target's retries mid-request.** When the local
   breaker opened during a request, `_may_retry` ended it with retries left and
   without going through panic mode. On the CI runner this produced 3.4% in
   `cloud_outage`, against 2.0% for v1.0. It was reproduced by pinning the
   benchmark to one contended core. All "last target" exceptions now live in
   `ResilientLLM._is_last`.

## Passos

1. `resilience.py` + unit tests.
2. Wire the breaker, the budget and hedging into `client.py` + client tests.
3. Log columns (`hedged`, `breaker_skips`) + stats.
4. Benchmark: three clients, burst scenario, amplification column.
5. README/CHANGELOG with measured numbers; commit; CI.
