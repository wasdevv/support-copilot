"""USD per million tokens, keyed by model id.

A model missing here has an unknown cost (None), never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.llm.types import Usage

MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    input: Decimal
    output: Decimal
    cache_read: Decimal
    # 5-minute TTL write: 1.25x input.
    cache_write: Decimal


def _anthropic(input_: str, output: str, cache_read: str | None = None) -> Price:
    inp = Decimal(input_)
    return Price(
        input=inp,
        output=Decimal(output),
        cache_read=Decimal(cache_read) if cache_read else inp / 10,
        cache_write=inp * Decimal("1.25"),
    )


ZERO = Price(Decimal(0), Decimal(0), Decimal(0), Decimal(0))

# Anthropic first-party prices, checked 2026-10-23. Haiku 5.5 is the <=100K-token tier.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": _anthropic("4", "20", "0.20"),
    "claude-sonnet-5-5": _anthropic("2", "10", "0.20"),
    "claude-haiku-5-5": _anthropic("0.10", "0.50"),
    # Local inference: no per-token bill. Listed explicitly so "free" is a fact
    # we recorded, not the default for anything we forgot to price.
    "llama3.2": ZERO,
    # Simulated providers used by tests and the outage benchmark.
    "sim-primary": _anthropic("0.10", "0.50"),
    "sim-secondary": _anthropic("2", "10", "0.20"),
    "sim-local": ZERO,
}


def cost_usd(model: str, usage: Usage) -> Decimal | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / MTOK
