# CLAUDE.md

support-copilot: customer-support copilot built in five stages (see README).
One PRD per stage in `docs/prd/`; read the current stage's PRD before coding.

- Python 3.12+, FastAPI, Pydantic v2, official `anthropic` SDK (v2, which uses
  `httpx2` internally: mock it with `httpx2.MockTransport`, not `httpx`).
- `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`; `.venv/bin/pytest`;
  `.venv/bin/ruff check . && .venv/bin/ruff format --check .`.
- Providers make exactly one attempt; retries, timeouts and fallback live only in
  `app/llm/client.py`. Keep the Anthropic SDK at `max_retries=0`.
- Unknown cost is `None`/`NULL`, never `0`: unpriced model, timed-out attempt,
  stream cut before its final usage event.
- No `temperature`/`top_p`/`top_k` in Messages API requests (400 since 2026-10-21).
- Local Ollama on this machine: `OLLAMA_MODELS=~/.local/ollama/models ~/.local/ollama/bin/ollama serve`
  (models: `llama3.2`, `nomic-embed-text`). Nothing in tests needs it.
- Benchmark numbers in README/CHANGELOG are measured; rerun
  `python -m evals.outage_bench --seed 7` and update them together when the
  client or fault model changes.
