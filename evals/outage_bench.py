"""Failure rate, tail latency and provider load of three LLM clients under simulated outages.

    python -m evals.outage_bench [--requests 2000] [--seed 7] [--json out.json]

All clients send the same support questions to the same fault-injected providers:

- naive: what most code does. One model, one attempt, a timeout.
- v1.0:  retries with backoff, fallback primary -> secondary -> local.
- v1.1:  v1.0 plus per-target circuit breakers, hedged requests and a retry budget.

Five scenarios draw faults independently per call. `burst_outages` takes
providers down in time windows, which is closer to a real incident. The local
model has a slow tail of its own (CPU inference), so the last fallback can fail too.
Latencies are scaled down: a 30 ms normal call, a 300 ms attempt timeout and a 2 s
deadline stand for roughly 2 s, 20 s and 2 min. Timeouts, backoff, breaker
windows and outage windows are scaled by the same factor.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from app.llm import errors
from app.llm.client import ResilientLLM, RetryPolicy
from app.llm.cost_log import CostLog
from app.llm.providers.simulated import (
    SimulatedProvider,
    bursts,
    down,
    random_errors,
    rate_limited,
)
from app.llm.resilience import BreakerPolicy, BudgetPolicy
from app.llm.types import CompletionRequest, Message, Target

QUESTIONS = [
    json.loads(line)["question"]
    for line in (Path(__file__).parent / "questions.jsonl").read_text().splitlines()
    if line.strip()
]

V10 = RetryPolicy(
    attempts_per_target=3,
    attempt_timeout_s=0.3,
    deadline_s=2.0,
    backoff_base_s=0.05,
    backoff_cap_s=0.4,
    breaker=None,
    budget=None,
    hedge_after_s=None,
)
V11 = replace(
    V10,
    breaker=BreakerPolicy(window=100, min_calls=20, failure_rate=0.5, open_s=0.25),
    budget=BudgetPolicy(ratio=0.1, min_tokens=10),
    # Normal primary latency is 30 ms, so 60 ms is past its healthy tail.
    hedge_after_s=0.06,
)
NAIVE = replace(V10, attempts_per_target=1)
CLIENTS = {"naive": NAIVE, "v1.0": V10, "v1.1": V11}

# Each scenario builds fresh fault functions per run: burst windows keep state.
Faults = Callable[[], tuple[dict, dict, dict]]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    faults: Faults
    # Paced so time-windowed outages span several windows per run.
    concurrency: int = 100


SCENARIOS = [
    Scenario(
        "flaky_primary",
        "primary 5xx on 20% of calls",
        lambda: (
            {"fault": random_errors(0.20)},
            {"fault": random_errors(0.05)},
            {"fault": random_errors(0.02)},
        ),
    ),
    Scenario(
        "rate_limit_storm",
        "primary 429 on 40% of calls, Retry-After 0.2 s",
        lambda: (
            {"fault": rate_limited(0.40, retry_after=0.2)},
            {"fault": rate_limited(0.10, retry_after=0.2)},
            {"fault": random_errors(0.02)},
        ),
    ),
    Scenario(
        "slow_primary",
        "30% of primary calls past the timeout",
        lambda: (
            {"slow_rate": 0.30, "slow_latency_s": 1.0},
            {"fault": random_errors(0.05)},
            {"fault": random_errors(0.02)},
        ),
    ),
    Scenario(
        "primary_outage",
        "primary fully down",
        lambda: ({"fault": down()}, {"fault": random_errors(0.05)}, {"fault": random_errors(0.02)}),
    ),
    Scenario(
        "cloud_outage",
        "both Claude models down",
        lambda: ({"fault": down()}, {"fault": down()}, {"fault": random_errors(0.02)}),
    ),
    Scenario(
        "burst_outages",
        "primary down 40% of the time in windows, secondary 25%, overlapping",
        lambda: (
            {"fault": bursts(period_s=1.0, down_s=0.4)},
            {"fault": bursts(period_s=1.2, down_s=0.3, phase_s=0.5)},
            {"fault": random_errors(0.02)},
        ),
        concurrency=25,
    ),
    Scenario(
        "total_outage",
        "every target down",
        lambda: ({"fault": down()}, {"fault": down()}, {"fault": down()}),
    ),
]
PARTIAL = [s.name for s in SCENARIOS if s.name != "total_outage"]


def providers(scenario: Scenario, seed: int) -> list[tuple[SimulatedProvider, str]]:
    primary, secondary, local = scenario.faults()
    return [
        (SimulatedProvider("primary", latency_s=0.03, seed=seed, **primary), "sim-primary"),
        (
            SimulatedProvider("secondary", latency_s=0.06, seed=seed + 1, **secondary),
            "sim-secondary",
        ),
        (
            SimulatedProvider(
                "local",
                latency_s=0.15,
                # CPU inference has a fat tail: a quarter of local calls blow
                # past the attempt timeout, so the last resort is not free.
                slow_rate=0.25,
                slow_latency_s=0.5,
                seed=seed + 2,
                **local,
            ),
            "sim-local",
        ),
    ]


async def measure(scenario: Scenario, client: str, n: int, seed: int) -> dict:
    targets = providers(scenario, seed)
    if client == "naive":
        targets = targets[:1]
    log = CostLog()
    llm = ResilientLLM(
        {p.name: p for p, _ in targets},
        [Target(p.name, model) for p, model in targets],
        CLIENTS[client],
        log,
        rng=random.Random(seed),
    )
    gate = asyncio.Semaphore(scenario.concurrency)

    async def one(i: int) -> None:
        request = CompletionRequest(messages=[Message("user", QUESTIONS[i % len(QUESTIONS)])])
        async with gate:
            with contextlib.suppress(errors.LLMError):
                await llm.complete(request, endpoint="answer")

    await asyncio.gather(*(one(i) for i in range(n)))
    return log.stats()["answer"]


async def main(n: int, seed: int) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        row = {"scenario": scenario.name, "description": scenario.description}
        for client in CLIENTS:
            row[client] = await measure(scenario, client, n, seed)
        rows.append(row)
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def change(before: float, after: float) -> str:
    return f"{(after - before) / before * 100:+.0f}%" if before else "n/a"


def mean(rows: list[dict], client: str, key: str) -> float:
    picked = [r[client][key] for r in rows if r["scenario"] in PARTIAL]
    return sum(picked) / len(picked)


def report(rows: list[dict], n: int, seed: int) -> str:
    lines = [
        f"{n} requests per scenario per client, seed {seed}",
        "",
        "Failed requests",
        "",
        "| scenario | naive | v1.0 | v1.1 |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {pct(r['naive']['error_rate'])} "
            f"| {pct(r['v1.0']['error_rate'])} | {pct(r['v1.1']['error_rate'])} |"
        )
    lines += [
        "",
        "Tail latency and provider load (v1.0 → v1.1)",
        "",
        "| scenario | p95 v1.0 | p95 v1.1 | p95 change | calls/req v1.0 | calls/req v1.1 "
        "| hedged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        a, b = r["v1.0"], r["v1.1"]
        lines.append(
            f"| {r['scenario']} | {a['latency_p95_ms']:.0f} ms | {b['latency_p95_ms']:.0f} ms "
            f"| {change(a['latency_p95_ms'], b['latency_p95_ms'])} "
            f"| {a['avg_attempts']:.2f} | {b['avg_attempts']:.2f} | {pct(b['hedge_rate'])} |"
        )
    naive, v10, v11 = (mean(rows, c, "error_rate") for c in CLIENTS)
    p95_10, p95_11 = mean(rows, "v1.0", "latency_p95_ms"), mean(rows, "v1.1", "latency_p95_ms")
    lines += [
        "",
        f"Mean over the {len(PARTIAL)} partial-outage scenarios: failed requests "
        f"naive {pct(naive)}, v1.0 {pct(v10)}, v1.1 {pct(v11)}; "
        f"p95 latency v1.0 {p95_10:.0f} ms → v1.1 {p95_11:.0f} ms ({change(p95_10, p95_11)}).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = asyncio.run(main(args.requests, args.seed))
    print(report(results, args.requests, args.seed))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
