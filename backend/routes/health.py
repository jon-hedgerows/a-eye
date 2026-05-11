from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class PullRequest(BaseModel):
    name: str


@router.get("/health")
async def health_check(request: Request):
    """App status + Ollama connection check."""
    llm_client = request.app.state.llm_client
    connected = await llm_client.check_connection()

    models = []
    if connected:
        models = await llm_client.list_models()

    model_names = [m.get("name", "?") for m in models]
    settings = request.app.state.settings

    return {
        "status": "ok",
        "ollama": {
            "connected": connected,
            "host": settings.ollama_host,
            "models": model_names,
            "vision_model": settings.vision_model,
            "llm_model": settings.llm_model or None,
        },
    }


@router.get("/models")
async def list_models(request: Request):
    """List available LLM models, categorised by capability."""
    llm_client = request.app.state.llm_client
    return await llm_client.list_models_by_capability()


@router.post("/models/pull")
async def pull_model(request: Request, body: PullRequest):
    """Stream model pull progress from Ollama (NDJSON)."""
    llm_client = request.app.state.llm_client
    model_name = body.name.strip()
    if not model_name:
        raise HTTPException(400, "Model name is required")
    return StreamingResponse(
        llm_client.pull_model_stream(model_name),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
