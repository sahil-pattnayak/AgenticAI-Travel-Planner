from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.planner import TravelPlannerAgent
from app.core.config import settings
from app.ingestion.pipeline import ingest_location
from app.memory.conversation import ConversationMemory
from app.rag.retriever import TravelRetriever
from app.rag.vector_store import VectorStoreManager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("travel_planner_api")

app = FastAPI(title="VoyagePilot AI - Smart Travel Planner API", version="1.0.0")
FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

memory = ConversationMemory()
vector_store = VectorStoreManager()
retriever = TravelRetriever(vector_store=vector_store)
agent = TravelPlannerAgent(retriever=retriever, memory=memory)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    location: str | None = None
    weather_consent: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    actions_taken: list[str]
    suggested_followups: list[str] = []


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest) -> dict[str, Any]:
    query = payload.query.strip()
    location = payload.location.strip() if payload.location else None
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    if location and not vector_store.has_location(location):
        logger.info("Location %s not found in vector DB. Triggering ingestion.", location)
        try:
            chunks = await ingest_location(location_name=location, vector_store=vector_store)
            logger.info("Auto-ingestion complete for %s with %s chunks.", location, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Auto-ingestion failed for %s", location)
            raise HTTPException(
                status_code=500,
                detail=f"Auto-ingestion failed for location '{location}': {exc}",
            ) from exc

    try:
        result = agent.run(
            query=query,
            location=location,
            weather_consent=payload.weather_consent,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc
