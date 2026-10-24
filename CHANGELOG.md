# Changelog

## v1.1.1 (2026-10-23)

- Panic mode keeps retries on the last model. The first v1.1.0 CI run showed
  `cloud_outage` at 5.2%, against 1.8% for v1.0: when the local model's breaker
  opened, panic mode gave it a single attempt. Benchmark numbers below are from
  v1.1.1.

## v1.1.0 (2026-10-23): circuit breakers, hedged requests, retry budget

- Per-model circuit breaker (failure rate over the last 100 calls, minimum 20,
  half-open single probe). Invalid output, refusals and bad requests do not trip it.
- Panic mode: when every breaker is open, ignore them and try each model once
  (the last one keeps its retries).
- Hedged requests: after `hedge_after_s` the same request goes to the next model,
  first answer wins, and the loser is cancelled and logged with unknown cost.
- Retry budget (token bucket, ~10% of traffic). A denied retry becomes a
  fallback, and the last model is never gated.
- Log and `/v1/stats` gain hedge rate, panic rate and breaker skips.
- Benchmark compares naive / v1.0 / v1.1 and adds `burst_outages`. Over six
  partial-outage scenarios: failed requests 57.2% → 0.4%, p95 −45% versus
  v1.0, and provider calls per request −70% in a primary outage and −80% in a
  cloud outage.

## v1.0.0 (2026-10-23): LLM calls are backend calls

- Resilient client (`app/llm/client.py`) with per-attempt timeouts, a request
  deadline, exponential backoff with full jitter, `Retry-After` support and an
  ordered model fallback chain (Claude Haiku 5.5 → Claude Sonnet 5.5 → local llama3.2).
- Streaming with first-token and idle timeouts; fallback only before the first token.
- Structured outputs validated with Pydantic; invalid output is retried, then falls back.
- Providers: Anthropic (official SDK, SDK retries off), Ollama, and a
  fault-injecting simulator.
- SQLite cost and latency log per request and per attempt; `GET /v1/stats` per endpoint.
  Unknown cost is logged as `NULL`, never `0`.
- FastAPI endpoints `/v1/answer` (JSON or SSE), `/v1/triage`, `/v1/stats`.
- `evals/questions.jsonl` (25 support questions) and `evals/outage_bench.py`:
  across five partial-outage scenarios, the share of failed requests drops from
  59.1% to 0.4%.
