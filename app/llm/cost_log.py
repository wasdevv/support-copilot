"""Per-request cost and latency log in SQLite: one row per request, one per attempt.

Cost columns are nullable on purpose. NULL means "not measurable" (unpriced
model, timed-out attempt the provider may have billed); summing it as zero would
under-report spend exactly when something went wrong.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.llm.pricing import cost_usd
from app.llm.types import Attempt, RequestRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id    TEXT PRIMARY KEY,
    endpoint      TEXT NOT NULL,
    created_at    REAL NOT NULL,
    outcome       TEXT NOT NULL,
    error_kind    TEXT,
    provider      TEXT,
    model         TEXT,
    attempts      INTEGER NOT NULL,
    fallback      INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      TEXT,
    cost_complete INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    ttft_ms       REAL,
    hedged        INTEGER NOT NULL,
    breaker_skips INTEGER NOT NULL,
    panic         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    request_id    TEXT NOT NULL REFERENCES requests(request_id),
    seq           INTEGER NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    error_kind    TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      TEXT,
    latency_ms    REAL NOT NULL,
    ttft_ms       REAL,
    PRIMARY KEY (request_id, seq)
);
CREATE INDEX IF NOT EXISTS requests_endpoint ON requests(endpoint, created_at);
"""


def attempt_cost(attempt: Attempt) -> Decimal | None:
    if not attempt.cost_known:
        return None
    if attempt.usage is None:
        # Rejected before generation (5xx, 429, 4xx): providers do not bill these.
        return Decimal(0)
    return cost_usd(attempt.target.model, attempt.usage)


def request_cost(record: RequestRecord) -> tuple[Decimal, bool]:
    """Sum of the attempts we can price, and whether that sum is the whole bill."""
    costs = [attempt_cost(a) for a in record.attempts]
    known = sum((c for c in costs if c is not None), Decimal(0))
    return known, all(c is not None for c in costs)


class CostLog:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def write(self, record: RequestRecord) -> None:
        cost, complete = request_cost(record)
        served = next((a for a in record.attempts if a.outcome == "ok"), None)
        usages = [a.usage for a in record.attempts if a.usage]
        first_target = record.attempts[0].target if record.attempts else None
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.request_id,
                    record.endpoint,
                    time.time(),
                    record.outcome,
                    record.error_kind,
                    served.target.provider if served else None,
                    served.target.model if served else None,
                    len(record.attempts),
                    int(served is not None and served.target != first_target),
                    sum(u.input_tokens for u in usages),
                    sum(u.output_tokens for u in usages),
                    str(cost) if (complete or cost) else None,
                    int(complete),
                    record.latency_ms,
                    record.ttft_ms,
                    int(record.hedged),
                    record.breaker_skips,
                    int(record.panic),
                ),
            )
            self._db.executemany(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        record.request_id,
                        seq,
                        a.target.provider,
                        a.target.model,
                        a.outcome,
                        a.error_kind,
                        a.usage.input_tokens if a.usage else None,
                        a.usage.output_tokens if a.usage else None,
                        None if (c := attempt_cost(a)) is None else str(c),
                        a.latency_ms,
                        a.ttft_ms,
                    )
                    for seq, a in enumerate(record.attempts)
                ],
            )

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict[str, dict[str, Any]]:
        rows = self._db.execute(
            "SELECT endpoint, outcome, cost_usd, cost_complete, latency_ms, ttft_ms,"
            " attempts, fallback, hedged, breaker_skips, panic FROM requests"
        ).fetchall()
        by_endpoint: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_endpoint.setdefault(row["endpoint"], []).append(row)
        return {endpoint: _summarize(group) for endpoint, group in sorted(by_endpoint.items())}


def _summarize(rows: list[sqlite3.Row]) -> dict[str, Any]:
    n = len(rows)
    errors = sum(r["outcome"] != "ok" for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    ttfts = sorted(r["ttft_ms"] for r in rows if r["ttft_ms"] is not None)
    cost = sum((Decimal(r["cost_usd"]) for r in rows if r["cost_usd"] is not None), Decimal(0))
    return {
        "requests": n,
        "errors": errors,
        "error_rate": errors / n,
        "fallback_rate": sum(r["fallback"] for r in rows) / n,
        # Provider calls per request, cancelled hedge losers included: the load
        # this endpoint puts on providers relative to a single call.
        "avg_attempts": sum(r["attempts"] for r in rows) / n,
        "hedge_rate": sum(r["hedged"] for r in rows) / n,
        "breaker_skips": sum(r["breaker_skips"] for r in rows),
        "panic_rate": sum(r["panic"] for r in rows) / n,
        "cost_usd": str(cost),
        # Requests whose cost is only a floor. When non-zero, cost_usd is a minimum.
        "cost_incomplete_requests": sum(not r["cost_complete"] for r in rows),
        "cost_per_request_usd": str(cost / n),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "ttft_p95_ms": percentile(ttfts, 95),
    }


def percentile(sorted_values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]
