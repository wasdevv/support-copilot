# Changelog

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
