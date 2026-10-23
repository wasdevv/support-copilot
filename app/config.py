from __future__ import annotations

import os

from app.llm.client import ResilientLLM, RetryPolicy
from app.llm.cost_log import CostLog
from app.llm.providers.base import Provider
from app.llm.types import Target

DEFAULT_CHAIN = "anthropic:claude-haiku-5-5,anthropic:claude-sonnet-5-5,ollama:llama3.2"


def parse_chain(raw: str) -> list[Target]:
    targets = []
    for item in raw.split(","):
        provider, sep, model = item.strip().partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"bad chain entry {item!r}; expected provider:model")
        targets.append(Target(provider, model))
    return targets


def llm_from_env() -> tuple[ResilientLLM, CostLog]:
    chain = parse_chain(os.environ.get("SC_CHAIN", DEFAULT_CHAIN))
    providers: dict[str, Provider] = {}
    names = {t.provider for t in chain}
    if "anthropic" in names:
        from app.llm.providers.anthropic import AnthropicProvider

        providers["anthropic"] = AnthropicProvider()
    if "ollama" in names:
        from app.llm.providers.ollama import OllamaProvider

        providers["ollama"] = OllamaProvider(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    policy = RetryPolicy(
        attempt_timeout_s=float(os.environ.get("SC_ATTEMPT_TIMEOUT_S", 20)),
        deadline_s=float(os.environ.get("SC_DEADLINE_S", 45)),
    )
    log = CostLog(os.environ.get("SC_DB", "var/copilot.db"))
    return ResilientLLM(providers, chain, policy, log), log
