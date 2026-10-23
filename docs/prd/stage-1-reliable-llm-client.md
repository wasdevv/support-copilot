# PRD — Stage 1: LLM calls are backend calls

Project: **support-copilot**, a customer-support copilot built in five stages
(reliability → RAG → cost/latency → agent → evals/observability). This PRD
covers **stage 1 only**. Later stages get their own PRD.

## Problema

An LLM API is a remote dependency with worse failure modes than a database:
multi-second latency, rate limits (429), overload (529/503), timeouts that may
or may not have been billed, and outputs that can be malformed. Code that calls
`messages.create()` directly and returns its result fails the user every time
the provider hiccups, and nobody can say what a request cost or which endpoint
spends the money.

Stage 1 treats the LLM call as a backend call: bounded in time, retried with
backoff when retrying can help, failed over to another model when retrying
cannot, and logged with tokens, cost and latency per request.

## Requisitos

1. **Provider abstraction** with two real providers and one simulated:
   - `anthropic` — official `anthropic` Python SDK (async), SDK retries **off**
     (`max_retries=0`), because the retry policy is ours and stacking two
     policies multiplies attempts.
   - `ollama` — local model over HTTP (`httpx`), the offline fallback.
   - `simulated` — deterministic fault injection (error rate, outage windows,
     latency distribution, error kinds) used by tests and the outage benchmark.
2. **Timeouts**: per-attempt timeout and a total deadline per request. For
   streaming, a time-to-first-token timeout and an idle timeout between chunks.
3. **Retries**: exponential backoff with full jitter, capped; `Retry-After`
   honored when it fits within the cap, otherwise skip straight to fallback.
   Only retryable errors retry: timeouts, connection errors, 408/409/429/5xx/529.
4. **Model fallback**: an ordered chain of `(provider, model)` targets. When a
   target exhausts its attempts or fails with a target-specific error (auth,
   model not found), the next target runs. A `400 invalid request` stops the
   chain — the same payload would fail everywhere.
5. **Streaming**: retry/fallback is allowed only **before the first token** is
   sent to the caller. After that, a failure ends the stream with an error
   event; never splice output from two models.
6. **Structured outputs**: request a JSON schema (Anthropic `output_config.format`,
   Ollama `format`), validate with Pydantic; invalid output counts as a failed
   attempt and goes through the same retry/fallback path.
7. **Cost & latency log**: one row per request and one per attempt in SQLite:
   request id, endpoint, provider, model, outcome, error kind, input/output/cache
   tokens, cost in USD, latency, time to first token.
8. **Cost is never invented**: a model missing from the price table has cost
   `NULL`, not zero; a timed-out attempt may have been billed, so its cost is
   unknown and the request is flagged `cost_complete = false`.
9. **HTTP API (FastAPI)**: `POST /v1/answer` (support reply, optional SSE
   streaming), `POST /v1/triage` (structured: category, priority, sentiment),
   `GET /v1/stats` (per-endpoint requests, error rate, cost, p50/p95 latency),
   `GET /healthz`.
10. **Outage benchmark**: `evals/outage_bench.py` runs the same workload
    through a naive client (single model, timeout only) and the resilient client
    under simulated outage scenarios, and prints failure rates — these fill the
    `[X%] → [Y%]` in the README bullet.

## Critérios de aceite

- [ ] `pytest` green, covering: retry on 429/5xx/timeout, no retry on 400,
      fallback on exhausted attempts and on auth/not-found, `Retry-After`
      honored and capped, total deadline enforced, streaming fallback before
      first token and no fallback after it, structured-output validation
      failure → retry → fallback, unknown model price → `NULL` cost, timeout
      attempt → `cost_complete = false`.
- [ ] Anthropic provider tested through the real SDK against an
      `httpx.MockTransport` speaking the real wire format (status codes, error
      body, `retry-after` header, SSE events) — no mocks of our own code.
- [ ] `GET /v1/stats` returns per-endpoint count, error rate, total cost and
      p50/p95 latency computed from the SQLite log.
- [ ] `evals/outage_bench.py` prints, per scenario, the naive and resilient
      failure rates with N ≥ 1000 requests and a fixed seed; README quotes the
      measured numbers, not targets.
- [ ] `ruff check` and `ruff format --check` clean; GitHub Actions runs lint,
      tests and the benchmark on push.
- [ ] Live smoke against local Ollama (`llama3.2`) through the API documented
      in the README with its real output.

## Fora de escopo

- Retrieval, embeddings, caching, model routing by difficulty, agents, evals,
  tracing — stages 2–5.
- Circuit breaker across requests (per-request retries only in stage 1;
  revisit if the benchmark shows retry storms).
- Auth, multi-tenancy, deployment.
- A live Anthropic run in CI (no API key in this environment; the provider is
  exercised through the SDK with a mock transport).

## Arquitetura e decisões

Layout fixed by the user for all five stages (`app/retrieval`, `app/cache`,
`app/agent` and `data/docs/` arrive in v2–v4):

```
app/
  llm/
    types.py        CompletionRequest, Completion, Usage, StreamEvent
    errors.py       LLMError hierarchy with .retryable / .fallback_ok / .retry_after
    pricing.py      $/MTok table, cost_usd() -> Decimal | None
    providers/      anthropic.py, ollama.py, simulated.py (Provider protocol)
    client.py       ResilientLLM: timeouts, backoff, fallback chain, streaming, structured
    cost_log.py     SQLite requests + attempts, per-endpoint stats
  api.py            FastAPI endpoints
  prompts.py        system prompts, triage schema
  config.py         chain and policy from env
evals/
  questions.jsonl   support questions; grows every stage
  outage_bench.py   requirement 10
tests/
CHANGELOG.md
.github/workflows/
```

Found during implementation:

- With no credentials, the Anthropic SDK raises a bare `TypeError` at request
  time, not an `AnthropicError`. Unmapped, it turned a missing key into a 500
  instead of a fallback (caught by the live smoke, now tested).
- An SSE `error` event after a `200` surfaces as `APIStatusError` with status
  200; it is mapped from the event's error type, as a server-side failure.
- Pydantic emits `$ref`/`$defs` for enums; the Anthropic schema goes through
  the SDK's `transform_schema`.
- The local model's slow tail is part of the benchmark's fault model, otherwise
  the last fallback is free and every partial outage reads 0.0%.

- **Python 3.12+, FastAPI, Pydantic v2, httpx, official `anthropic` SDK.**
  `pyproject.toml` with hatchling; `ruff`, `pytest`, `pytest-asyncio`.
- **Default chain**: `anthropic/claude-haiku-5-5` → `anthropic/claude-sonnet-5-5`
  → `ollama/llama3.2`. Haiku first because support replies are short and
  latency-sensitive; Sonnet as a different-capacity pool on the same provider;
  Ollama as the provider-independent last resort. Configurable via env.
- **No `temperature`** in requests: since 2026-10-21 the Messages API returns
  400 for sampling parameters.
- **Backoff**: `sleep = random(0, min(cap, base * 2**attempt))` (full jitter).
  Sleep and clock are injected so tests do not wait.
- **SQLite** (stdlib) for the log: one process, no server to run; the schema
  is plain enough to move to Postgres later.
- **Cost columns are `NULL`able on purpose** (requirement 8).

## Riscos

- **Simulated outages flatter the resilient client** if the fault model is
  unrealistic. Mitigation: several scenarios (random 5xx, 429 bursts with
  `Retry-After`, full primary outage, slow provider/timeouts, correlated
  outage of both Anthropic targets) and report each one, including the one
  where fallback cannot help.
- **Retries raise cost and tail latency.** The benchmark reports p95 latency
  and attempts per request next to failure rate.
- **Python 3.14 locally vs 3.12 in CI** — pin the CI matrix to both.

## Passos

1. Scaffold repo: pyproject, ruff, pytest, README skeleton, CI workflow.
2. `types`, `errors`, `pricing` + tests.
3. Simulated provider + `ResilientLLM.complete` with retry/fallback/deadline + tests.
4. Streaming path + tests; structured output path + tests.
5. Anthropic provider via SDK + MockTransport tests; Ollama provider + tests.
6. `cost_log` + stats + tests.
7. FastAPI app + request tests.
8. Outage benchmark; run it; write numbers into README.
9. Live smoke against Ollama; commit; create public GitHub repo; push; CI green.
