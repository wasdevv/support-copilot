"""Failure rate of a naive LLM call vs the resilient client under simulated outages.

    python -m evals.outage_bench [--requests 2000] [--seed 7] [--json out.json]

Both clients send the same questions to the same fault-injected providers. The
naive client is what most code does: one model, one attempt, a timeout. The
resilient client retries with backoff and falls back primary -> secondary -> local.
Faults are drawn per call (not correlated in time), and the local model has a
slow tail of its own, so the last resort can fail too.
Latencies are scaled down (tens of ms instead of seconds) so a run takes seconds;
timeouts and backoff are scaled with them, so the ratios hold.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.llm import errors
from app.llm.client import ResilientLLM, RetryPolicy
from app.llm.cost_log import CostLog
from app.llm.providers.simulated import SimulatedProvider, down, random_errors, rate_limited
from app.llm.types import CompletionRequest, Message, Target

QUESTIONS = [
    json.loads(line)["question"]
    for line in (Path(__file__).parent / "questions.jsonl").read_text().splitlines()
    if line.strip()
]

# Scaled: a 30 ms "normal" call, 300 ms attempt timeout, 2 s request deadline.
POLICY = RetryPolicy(
    attempts_per_target=3,
    attempt_timeout_s=0.3,
    deadline_s=2.0,
    backoff_base_s=0.05,
    backoff_cap_s=0.4,
)
NAIVE = RetryPolicy(attempts_per_target=1, attempt_timeout_s=0.3, deadline_s=2.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    primary: dict
    secondary: dict
    local: dict


SCENARIOS = [
    Scenario(
        "flaky_primary",
        "primary returns 5xx on 20% of calls",
        {"fault": random_errors(0.20)},
        {"fault": random_errors(0.05)},
        {"fault": random_errors(0.02)},
    ),
    Scenario(
        "rate_limit_storm",
        "primary 429s 40% of calls with Retry-After 0.2s",
        {"fault": rate_limited(0.40, retry_after=0.2)},
        {"fault": rate_limited(0.10, retry_after=0.2)},
        {"fault": random_errors(0.02)},
    ),
    Scenario(
        "slow_primary",
        "30% of primary calls take 1s (past the 0.3s timeout)",
        {"slow_rate": 0.30, "slow_latency_s": 1.0},
        {"fault": random_errors(0.05)},
        {"fault": random_errors(0.02)},
    ),
    Scenario(
        "primary_outage",
        "primary model fully down",
        {"fault": down()},
        {"fault": random_errors(0.05)},
        {"fault": random_errors(0.02)},
    ),
    Scenario(
        "cloud_outage",
        "both cloud models down, only the local model is up",
        {"fault": down()},
        {"fault": down()},
        {"fault": random_errors(0.02)},
    ),
    Scenario(
        "total_outage",
        "every target down: fallback cannot help",
        {"fault": down()},
        {"fault": down()},
        {"fault": down()},
    ),
]


def providers(scenario: Scenario, seed: int) -> list[tuple[SimulatedProvider, str]]:
    return [
        (
            SimulatedProvider("primary", latency_s=0.03, seed=seed, **scenario.primary),
            "sim-primary",
        ),
        (
            SimulatedProvider("secondary", latency_s=0.06, seed=seed + 1, **scenario.secondary),
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
                **scenario.local,
            ),
            "sim-local",
        ),
    ]


async def run(llm: ResilientLLM, n: int, concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        request = CompletionRequest(messages=[Message("user", QUESTIONS[i % len(QUESTIONS)])])
        async with gate:
            with contextlib.suppress(errors.LLMError):
                await llm.complete(request, endpoint="answer")

    await asyncio.gather(*(one(i) for i in range(n)))


async def measure(scenario: Scenario, n: int, seed: int, resilient: bool) -> dict:
    targets = providers(scenario, seed)
    if not resilient:
        targets = targets[:1]
    log = CostLog()
    llm = ResilientLLM(
        {p.name: p for p, _ in targets},
        [Target(p.name, model) for p, model in targets],
        POLICY if resilient else NAIVE,
        log,
        rng=random.Random(seed),
    )
    await run(llm, n, concurrency=100)
    return log.stats()["answer"]


async def main(n: int, seed: int) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        naive = await measure(scenario, n, seed, resilient=False)
        resilient = await measure(scenario, n, seed, resilient=True)
        rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "naive": naive,
                "resilient": resilient,
            }
        )
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def report(rows: list[dict], n: int, seed: int) -> str:
    lines = [
        f"{n} requests per scenario per client, seed {seed}",
        "",
        "| scenario | naive failed | resilient failed | resilient p95 | avg attempts | fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        nv, rs = r["naive"], r["resilient"]
        lines.append(
            f"| {r['scenario']} | {pct(nv['error_rate'])} | {pct(rs['error_rate'])} "
            f"| {rs['latency_p95_ms']:.0f} ms | {rs['avg_attempts']:.2f} "
            f"| {pct(rs['fallback_rate'])} |"
        )
    # Headline: the outages fallback is designed for. total_outage is reported
    # above but left out, since no client-side policy can serve it.
    partial = [r for r in rows if r["scenario"] != "total_outage"]
    naive = sum(r["naive"]["error_rate"] for r in partial) / len(partial)
    resilient = sum(r["resilient"]["error_rate"] for r in partial) / len(partial)
    lines += [
        "",
        f"Mean over the {len(partial)} partial-outage scenarios: "
        f"naive {pct(naive)} failed, resilient {pct(resilient)} failed.",
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
