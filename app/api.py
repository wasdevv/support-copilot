from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.llm import errors
from app.llm.client import ResilientLLM, Result
from app.llm.cost_log import CostLog
from app.llm.types import CompletionRequest, Message, StreamDone, TextDelta
from app.prompts import ANSWER_SYSTEM, TRIAGE_SYSTEM, Triage


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    stream: bool = False


class TriageIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


def create_app(llm: ResilientLLM, log: CostLog) -> FastAPI:
    app = FastAPI(title="support-copilot")

    def meta(request_id: str) -> dict[str, Any]:
        row = log.get(request_id) or {}
        return {
            "request_id": request_id,
            "provider": row.get("provider"),
            "model": row.get("model"),
            "attempts": row.get("attempts"),
            # null when the request's cost could not be measured in full.
            "cost_usd": row.get("cost_usd") if row.get("cost_complete") else None,
            "latency_ms": row.get("latency_ms"),
        }

    def failure(exc: errors.LLMError, request_id: str) -> JSONResponse:
        # A 400 from the provider means we built a bad request: our bug, not theirs.
        status = 502 if isinstance(exc, errors.BadRequest) else 503
        return JSONResponse(
            {"error": exc.kind, "detail": str(exc), "request_id": request_id}, status_code=status
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/answer", response_model=None)
    async def answer(body: AnswerIn) -> Any:
        request_id = uuid.uuid4().hex
        request = CompletionRequest(
            messages=[Message("user", body.question)], system=ANSWER_SYSTEM, max_tokens=400
        )
        if body.stream:
            return StreamingResponse(
                _sse(llm, request, request_id, meta), media_type="text/event-stream"
            )
        try:
            result: Result[str] = await llm.complete(
                request, endpoint="answer", request_id=request_id
            )
        except errors.LLMError as exc:
            return failure(exc, request_id)
        return {"answer": result.value, **meta(request_id)}

    @app.post("/v1/triage", response_model=None)
    async def triage(body: TriageIn) -> Any:
        request_id = uuid.uuid4().hex
        request = CompletionRequest(
            messages=[Message("user", body.message)], system=TRIAGE_SYSTEM, max_tokens=300
        )
        try:
            result = await llm.complete_structured(
                request, Triage, endpoint="triage", request_id=request_id
            )
        except errors.LLMError as exc:
            return failure(exc, request_id)
        return {"triage": result.value.model_dump(mode="json"), **meta(request_id)}

    @app.get("/v1/stats")
    async def stats() -> dict[str, Any]:
        return {"endpoints": log.stats()}

    return app


async def _sse(llm, request, request_id, meta) -> AsyncIterator[str]:  # type: ignore[no-untyped-def]
    try:
        async for event in llm.stream(request, endpoint="answer", request_id=request_id):
            if isinstance(event, TextDelta):
                yield _event("delta", {"text": event.text})
            elif isinstance(event, StreamDone):
                pass
    except errors.LLMError as exc:
        # Headers are already sent, so the failure travels as an event.
        yield _event("error", {"error": exc.kind, "request_id": request_id})
        return
    # The record is written when the generator finishes, so meta reads it here.
    yield _event("done", meta(request_id))


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def from_env() -> FastAPI:
    from app.config import llm_from_env

    return create_app(*llm_from_env())
