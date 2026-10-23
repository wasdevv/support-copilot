from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.llm.providers.simulated import SimulatedProvider, down, fail_calls


@pytest.fixture
def client_for(build, log):
    def _client(*providers):
        return TestClient(create_app(build(*providers), log))

    return _client


def test_answer_returns_text_and_request_metadata(client_for):
    client = client_for((SimulatedProvider("a", reply="It ships in 2 days."), "sim-primary"))

    body = client.post("/v1/answer", json={"question": "When does it ship?"}).json()

    assert body["answer"] == "It ships in 2 days."
    assert body["provider"] == "a" and body["model"] == "sim-primary" and body["attempts"] == 1
    assert float(body["cost_usd"]) > 0


def test_answer_rejects_empty_question(client_for):
    client = client_for((SimulatedProvider("a"), "sim-primary"))

    assert client.post("/v1/answer", json={"question": ""}).status_code == 422


def test_answer_returns_503_when_every_target_fails(client_for):
    client = client_for((SimulatedProvider("a", fault=down()), "sim-primary"))

    response = client.post("/v1/answer", json={"question": "hi"})

    assert response.status_code == 503
    assert response.json()["error"] == "all_targets_failed"


def test_triage_returns_validated_structure(client_for):
    client = client_for((SimulatedProvider("a"), "sim-primary"))

    body = client.post("/v1/triage", json={"message": "I was charged twice"}).json()

    assert set(body["triage"]) == {"category", "priority", "sentiment", "summary"}


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_answer_stream_sends_deltas_then_done_with_cost(client_for):
    client = client_for(
        (SimulatedProvider("a", fault=down()), "sim-primary"),
        (SimulatedProvider("b", reply="on its way"), "sim-secondary"),
    )

    with client.stream("POST", "/v1/answer", json={"question": "hi", "stream": True}) as r:
        events = sse_events(r.read().decode())

    names = [name for name, _ in events]
    assert names == ["delta", "delta", "delta", "done"]
    assert "".join(d["text"] for n, d in events if n == "delta") == "on its way"
    assert events[-1][1]["provider"] == "b"


def test_stats_aggregates_per_endpoint(client_for):
    client = client_for((SimulatedProvider("a", fault=fail_calls(1, 2, 3)), "sim-primary"))

    client.post("/v1/answer", json={"question": "one"})  # call 0: ok
    client.post("/v1/answer", json={"question": "two"})  # calls 1-3: fail
    client.post("/v1/triage", json={"message": "three"})  # call 4: ok

    stats = client.get("/v1/stats").json()["endpoints"]

    assert stats["answer"]["requests"] == 2 and stats["answer"]["error_rate"] == 0.5
    assert stats["answer"]["avg_attempts"] == 2.0
    assert stats["triage"]["requests"] == 1 and stats["triage"]["errors"] == 0
    assert stats["answer"]["latency_p95_ms"] >= stats["answer"]["latency_p50_ms"]
