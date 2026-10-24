# support-copilot

A customer-support copilot for an online store, built the way LLM features
have to be built in production: as backend calls that time out, fail, cost
money and need to be measured. It grows in five stages, one release each:

| Stage | Theme | Status |
|---|---|---|
| v1 | LLM calls are backend calls: timeouts, retries, fallback, circuit breakers, hedging, cost log | **done** (v1.1) |
| v2 | Retrieval: hybrid search, reranking, citations, freshness | planned |
| v3 | Cost and latency: prompt/semantic caching, model routing | planned |
| v4 | Agent: refunds with step budgets, idempotent tools, human approval | planned |
| v5 | Evals in CI and end-to-end tracing | planned |

## v1: a resilient LLM client

`app/llm/` wraps every model call in one policy:

- **Timeouts**: per attempt, plus a total deadline per request. Streams get a
  time-to-first-token timeout and an idle timeout between chunks.
- **Retries**: exponential backoff with full jitter, capped. `Retry-After` is
  honored when it fits under the cap; a longer one skips to the next model
  instead of stalling the request. Only retryable failures retry: timeouts,
  connection errors, 408/409/429/5xx/529.
- **Model fallback**: an ordered chain, by default
  `claude-haiku-5-5 → claude-sonnet-5-5 → llama3.2 (local Ollama)`. Auth
  failures, unknown models and refusals skip straight to the next target. A
  `400` stops the chain, because the same payload fails everywhere.
- **Streaming**: retries and fallback happen only before the first token. After
  output reaches the caller, a failure ends the stream with an `error` event.
  Text from two models is never spliced together.
- **Structured outputs**: a JSON schema goes to the provider's constrained
  decoding (`output_config.format` / Ollama `format`) and the result is checked
  with Pydantic. Invalid output is a failed attempt, so it is retried and then
  falls back like any other failure.
- **Cost and latency log**: SQLite, one row per request and one per attempt,
  with tokens, USD cost, latency and TTFT. `GET /v1/stats` reports per-endpoint
  error rate, fallback rate, attempts, cost and p50/p95 latency.
- **Cost is never invented.** An unpriced model logs `NULL`, not `$0`. A
  timed-out attempt or a cancelled hedge may have been billed, so the request
  is flagged `cost_complete = false` and its total is shown as a floor.

v1.1 adds state that lasts across requests (`app/llm/resilience.py`):

- **Circuit breaker per model**: opens when 50% of the last 100 calls failed
  (minimum 20 calls), skips the model without calling it, and half-opens after
  a cooldown with a single probe. Bad requests, invalid output and refusals do
  not count, because they say nothing about whether the model is up.
- **Panic mode**: when every breaker is open, the client ignores them and tries
  each model once, keeping retries only on the last one (Envoy uses the same idea). A breaker on the last-resort model can open on noise,
  and honoring it would turn a partial outage into a total one.
- **Hedged requests** ("The Tail at Scale"): if the first model has not answered
  by `hedge_after_s` (set near its normal p95), the same request goes to the next
  model and the first answer wins. The loser is cancelled and its cost is
  logged as unknown.
- **Retry budget** (token bucket, as in gRPC retry throttling): retries and
  hedges are capped at about 10% of traffic, so a failing provider is not hit
  with 3× the load. A denied retry becomes a fallback. The budget never gates
  the last model, where a denied retry would fail the user.

The Anthropic SDK runs with `max_retries=0`. Two stacked retry policies
multiply attempts (3 × 3 = 9 calls), so the client keeps the only one.

### Measured: failures, tail latency and provider load under simulated outages

`python -m evals.outage_bench` sends the same 2,000 support questions per
scenario through three clients over fault-injected providers:

- **naive**: one model, one attempt, a timeout. What most code does.
- **v1.0**: retries with backoff and the fallback chain.
- **v1.1**: v1.0 plus circuit breakers, panic mode, hedging and the retry budget.

2000 requests per scenario per client, seed 7:

| scenario | naive failed | v1.0 failed | v1.1 failed |
|---|---:|---:|---:|
| flaky_primary: 20% 5xx | 21.3% | 0.0% | 0.1% |
| rate_limit_storm: 40% 429, Retry-After 0.2 s | 41.9% | 0.0% | 0.3% |
| slow_primary: 30% of calls past the timeout | 32.4% | 0.0% | 0.1% |
| primary_outage | 100.0% | 0.0% | 0.1% |
| cloud_outage: both Claude models down | 100.0% | 2.0% | 1.8% |
| burst_outages: primary down 40% of the time, secondary 25%, in windows | 47.5% | 0.1% | 0.1% |
| total_outage: everything down | 100.0% | 100.0% | 100.0% |

| scenario | p95 v1.0 | p95 v1.1 | change | provider calls/request v1.0 → v1.1 | hedged |
|---|---:|---:|---:|---:|---:|
| flaky_primary | 110 ms | 93 ms | −15% | 1.27 → 1.27 | 3.5% |
| rate_limit_storm | 554 ms | 214 ms | −61% | 1.67 → 1.34 | 9.0% |
| slow_primary | 730 ms | 362 ms | −51% | 1.45 → 1.21 | 10.2% |
| primary_outage | 291 ms | 120 ms | −59% | 4.05 → 1.22 | 9.9% |
| cloud_outage | 1225 ms | 811 ms | −34% | 7.34 → 1.46 | 1.8% |
| burst_outages | 230 ms | 211 ms | −8% | 1.39 → 1.28 | 4.5% |
| total_outage | 1299 ms | 995 ms | −23% | 9.00 → 4.75 | 1.0% |

Results across the six partial-outage scenarios:

- **Failed requests: 57.2% (naive) → 0.4% (v1.1).**
- **p95 latency: −42% on average versus v1.0.** In a full primary outage it drops
  from 291 ms to 120 ms (−59%). With a 429 storm it drops from 554 ms to 214 ms (−61%).
- **Load on providers during outages:** calls per request fall by 70% when the
  primary is down (4.05 → 1.22) and by 80% when both cloud models are down
  (7.34 → 1.46). That is the retry storm v1.0 would put on a provider that is
  already failing.

v1.1 fails as often as v1.0 (0.3–0.4%). What it buys is latency and provider
load, not success rate. Runs are not bit-for-bit reproducible, because asyncio
scheduling decides which timeouts land together. Across repeated runs, the p95
gain stays between −42% and −45%, and `cloud_outage` stays between 1.4% and 2.6% for
both clients. Every CI run publishes its own table in the workflow summary.

How to read these numbers:

- **Scale:** latencies are scaled down. A 30 ms call, a 300 ms attempt timeout and a 2 s
  deadline stand for roughly 2 s, 20 s and 2 min. Backoff, breaker
  cooldowns and outage windows are scaled by the same factor.
- **Fault model:** five scenarios draw faults independently per call, which is
  the easy case. `burst_outages` takes models down in time windows. The local model
  has a 25% slow tail of its own, as CPU inference does, so the last fallback can
  fail too. That tail is the ~2% in `cloud_outage`.
- **The price of hedging:** 1–10% of requests send a second call. The calls/request
  column includes those calls and the cancelled losers.
- **Where the policy stops:** in `total_outage` nothing can succeed. v1.1 still
  gives the last model its retries in panic mode, so it spends 4.75 calls instead
  of 9 and fails in about 1 s instead of 1.3 s.
- **Rejected designs:** five versions of v1.1 made some scenario worse than
  v1.0. The worst was 94% failures in `cloud_outage`, and two of them only showed
  up on the slower CI runner. They all had the same root cause: some mechanism
  gave up on the **last** model. The causes and fixes are in
  `docs/prd/stage-1.1-breakers-hedging-budget.md`.

### Live run against a real model

With no Anthropic credentials on the machine, the first two targets fail with
`target_unavailable` and the local `llama3.2` serves the request:

```
POST /v1/answer   {"question": "I was charged twice for the same order."}
200  {"answer": "That's not a good situation. I'm happy to help you resolve it. ...",
      "provider": "ollama", "model": "llama3.2", "attempts": 3, "cost_usd": "0", ...}

POST /v1/triage   {"message": "I was charged twice for order #1182 and need the money back before my rent is due tomorrow!"}
200  {"triage": {"category": "account", "priority": "urgent", "sentiment": "negative",
                 "summary": "Customer requests refund for duplicate charge on order #1182 ..."}, ...}
```

The triage is schema-valid but has the wrong category (`account` instead of
`billing`). Validation guarantees the shape of the output, not that it is
correct. Measuring quality is what the eval set in `evals/` is for, and it grows
from v2 onward.

## API

| Method | Path | |
|---|---|---|
| POST | `/v1/answer` | `{"question": "...", "stream": false}`; with `stream: true` the reply is SSE (`delta`, then `done` or `error`) |
| POST | `/v1/triage` | `{"message": "..."}` → category, priority, sentiment, summary |
| GET | `/v1/stats` | per endpoint: requests, error/fallback/hedge/panic rate, calls per request, cost, p50/p95 latency, p95 TTFT |
| GET | `/healthz` | liveness |

Every response carries `request_id`, the `provider`/`model` that answered,
`attempts`, `cost_usd` (`null` when it could not be measured in full) and
`latency_ms`. Provider failures return `503` with the error kind. A provider
`400` returns `502`, because a malformed request is our bug.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                      # 59 tests, about 5 s
.venv/bin/python -m evals.outage_bench
.venv/bin/uvicorn app.api:from_env --factory --port 8000
```

| Env | Default |
|---|---|
| `SC_CHAIN` | `anthropic:claude-haiku-5-5,anthropic:claude-sonnet-5-5,ollama:llama3.2` |
| `SC_ATTEMPT_TIMEOUT_S` / `SC_DEADLINE_S` | `20` / `45` |
| `SC_HEDGE_AFTER_S` | `4` (`off` disables hedging) |
| `SC_DB` | `var/copilot.db` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `ANTHROPIC_API_KEY` | read by the Anthropic SDK |

## Layout

```
app/
  llm/          v1: client (retry, fallback, timeouts, hedging), resilience
                (breakers, retry budget), providers, pricing, cost log
  api.py        FastAPI endpoints
  prompts.py    system prompts and the triage schema
evals/          support questions (grow every stage) and the outage benchmark
tests/          client, providers (real SDK over a mock transport), API
docs/prd/       one PRD per stage
```

Later stages add `app/retrieval/` (v2), `app/cache/` (v3), `app/agent/` (v4)
and the store's documents in `data/docs/`.

## Tests

The Anthropic provider is tested through the real `anthropic` SDK over an
`httpx` mock transport. The transport returns the documented wire format:
status codes, error bodies, `retry-after`, and SSE events, including an
`error` event after a `200`. Ollama is tested the same way with its
NDJSON stream. The client is tested against the fault-injecting simulated
provider with an injected sleep, so backoff assertions do not wait.
