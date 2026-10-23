# support-copilot

A customer-support copilot for an online store, built the way LLM features
have to be built in production: as backend calls that time out, fail, cost
money and need to be measured. It grows in five stages, one release each:

| Stage | Theme | Status |
|---|---|---|
| v1 | LLM calls are backend calls: timeouts, retries, fallback, cost log | **done** |
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
  timed-out attempt may have been billed, so the request is flagged
  `cost_complete = false` and its total is shown as a floor.

The Anthropic SDK runs with `max_retries=0`. Two stacked retry policies
multiply attempts (3 × 3 = 9 calls), so the client keeps the only one.

### Measured: failure rate under simulated outages

`python -m evals.outage_bench` sends the same 2,000 support questions per
scenario through two clients over fault-injected providers. The naive client is
one model, one attempt, a timeout. The resilient client is the one above.

2000 requests per scenario per client, seed 7:

| scenario | naive failed | resilient failed | resilient p95 | avg attempts | fallback |
|---|---:|---:|---:|---:|---:|
| flaky_primary: 20% 5xx | 21.3% | 0.0% | 111 ms | 1.27 | 0.9% |
| rate_limit_storm: 40% 429, Retry-After 0.2 s | 41.9% | 0.0% | 554 ms | 1.67 | 7.0% |
| slow_primary: 30% of calls past the timeout | 32.4% | 0.0% | 735 ms | 1.46 | 3.0% |
| primary_outage | 100.0% | 0.0% | 296 ms | 4.05 | 100.0% |
| cloud_outage: both Claude models down | 100.0% | 1.8% | 1237 ms | 7.35 | 98.2% |
| total_outage: everything down | 100.0% | 100.0% | 1292 ms | 9.00 | 0.0% |

Across the five partial-outage scenarios, the share of failed requests drops
from **59.1% to 0.4%**. `total_outage` is listed to show where the policy stops
helping: when every target is down, no client-side retry policy can serve the request.

How to read these numbers:

- Latencies are scaled down (a 30 ms call, 300 ms attempt timeout, 2 s deadline)
  so a run takes under a minute. Timeouts and backoff are scaled with them.
- Faults are drawn per call, not correlated in time. Real outages come in
  bursts, so these numbers are an upper bound on what retries alone can fix.
  Fallback to a different model is what covers correlated failures.
- The local fallback has a slow tail (25% of calls exceed the timeout) because
  CPU inference does. That tail causes the 1.8% in `cloud_outage`.
- Retries cost latency. During a full primary outage, every request spends
  three attempts on the dead primary before falling back (4.05 attempts on
  average). A circuit breaker would skip it. That is noted for v3.

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
| GET | `/v1/stats` | per-endpoint requests, error and fallback rate, attempts, cost, p50/p95 latency, p95 TTFT |
| GET | `/healthz` | liveness |

Every response carries `request_id`, the `provider`/`model` that answered,
`attempts`, `cost_usd` (`null` when it could not be measured in full) and
`latency_ms`. Provider failures return `503` with the error kind. A provider
`400` returns `502`, because a malformed request is our bug.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                      # 40 tests, about 5 s
.venv/bin/python -m evals.outage_bench
.venv/bin/uvicorn app.api:from_env --factory --port 8000
```

| Env | Default |
|---|---|
| `SC_CHAIN` | `anthropic:claude-haiku-5-5,anthropic:claude-sonnet-5-5,ollama:llama3.2` |
| `SC_ATTEMPT_TIMEOUT_S` / `SC_DEADLINE_S` | `20` / `45` |
| `SC_DB` | `var/copilot.db` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `ANTHROPIC_API_KEY` | read by the Anthropic SDK |

## Layout

```
app/
  llm/          v1: client (retry, fallback, timeouts), providers, pricing, cost log
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
