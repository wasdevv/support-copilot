from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

ANSWER_SYSTEM = """You are a customer-support assistant for an online store.
Answer the customer's question in plain language, in at most 120 words.
If the answer depends on a policy or an order detail you were not given, say what
you would need to check instead of guessing a policy."""

TRIAGE_SYSTEM = """You triage incoming customer-support messages.
Classify the message and write a one-sentence summary for the agent who picks it up.
Priority is "urgent" only for payment failures, security issues or an order that
must arrive within 48 hours."""


class Category(StrEnum):
    billing = "billing"
    shipping = "shipping"
    returns = "returns"
    account = "account"
    product = "product"
    other = "other"


class Priority(StrEnum):
    low = "low"
    normal = "normal"
    urgent = "urgent"


class Sentiment(StrEnum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class Triage(BaseModel):
    # extra="forbid" emits additionalProperties: false, which strict JSON-schema
    # output modes require.
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    sentiment: Sentiment
    summary: str
