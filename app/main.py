"""
FastAPI application entry point.
Exposes POST /chat and GET /health endpoints.
Implements Phase 7 features: catch-all error responses and 25s timeout budgets.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import ChatRequest, ChatResponse
from app.agent.orchestrator import handle_turn
from app.retrieval.index import load_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retrieval index once at startup asynchronously so health checks are ready."""
    logger.info("Initializing search indices at startup...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_index)
    logger.info("Search indices initialized.")
    yield

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent that recommends SHL Individual Test Solutions.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.
    Applies 25s timeout and safe fallback returns on exceptions.
    """
    try:
        # Enforce soft internal timeout budget of 25s
        response = await asyncio.wait_for(
            handle_turn(request.messages),
            timeout=25.0
        )
        return response
    except asyncio.TimeoutError:
        logger.error("Request timed out after 25s budget")
        return ChatResponse(
            reply="I'm sorry, my retrieval search took longer than expected. Could you try specifying the role name again?",
            recommendations=[],
            end_of_conversation=False
        )
    except Exception as exc:
        logger.exception("Global exception handler caught inside /chat: %s", exc)
        return ChatResponse(
            reply="I ran into an unexpected issue processing that request. Could we clarify the assessment skills you are looking for?",
            recommendations=[],
            end_of_conversation=False
        )
