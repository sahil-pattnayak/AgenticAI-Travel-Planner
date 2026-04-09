# Backend Architecture Guide

This document explains the backend of **VoyagePilot AI Travel Planner** in a way that is easy to present to reviewers, interviewers, and teammates.

## 1) Backend Objective

The backend provides an **agentic RAG planning service** that:

- Ingests travel data into a FAISS knowledge base.
- Decides when to use retrieval vs live tools (weather/news).
- Executes tools safely with explicit behavior rules.
- Maintains short conversation memory and follow-up continuity.
- Returns grounded travel plans with source links.

## 2) High-Level Architecture

```text
Client (Frontend/REST)
        |
        v
FastAPI (app/main.py)
        |
        v
TravelPlannerAgent (LangGraph flow)
  planner_node -> (optional) tool_node -> responder_node
        |                     |               |
        |                     |               +--> LLM synthesis + source formatting + follow-up suggestions
        |                     |
        |                     +--> ToolExecutor --> WeatherTool / NewsTool
        |
        +--> TravelRetriever --> VectorStoreManager (FAISS)
                                    ^
                                    |
                         Ingestion Pipeline (crawl/clean/chunk/index)
```

## 3) Module-by-Module Breakdown

### API Layer

- File: `app/main.py`
- Responsibilities:
- Exposes `POST /query`, `GET /health`, and serves `frontend/index.html` on `/`.
- Validates request shape using Pydantic (`QueryRequest`).
- Auto-triggers ingestion when destination is missing in FAISS.
- Calls `TravelPlannerAgent.run(...)` and returns structured response.

### Agent Layer

- Files: `app/agent/planner.py`, `app/agent/executor.py`, `app/agent/prompts.py`
- Responsibilities:
- **Planner node** decides action: `rag_only`, `weather_tool`, `news_tool`, or permission flow.
- **Tool node** executes tools through `ToolExecutor`.
- **Responder node** merges RAG context + tool output + memory into final answer.
- Enforces grounding rules:
- Removes duplicate citation blocks.
- Appends a clean `Sources:` section with direct URLs.
- Keeps follow-up responses query-focused (avoids unrelated sections).
- Applies hotel recommendation logic only when explicitly required.

### RAG Layer

- Files: `app/rag/retriever.py`, `app/rag/vector_store.py`, `app/rag/embedder.py`
- Responsibilities:
- Embeds text with OpenAI embeddings.
- Stores/retrieves chunks in FAISS local index.
- Supports metadata filtering by `location`.
- Falls back to broader retrieval when strict location filtering returns nothing.

### Ingestion Layer

- Files: `app/ingestion/pipeline.py`, `app/ingestion/crawler.py`, `app/ingestion/cleaner.py`, `app/ingestion/chunker.py`
- Responsibilities:
- Builds multi-source URLs for each destination:
- Wikivoyage
- Wikipedia
- Nomadic Matt
- Incredible India (direct paths + search fallback)
- Crawls source pages (crawl4ai path + robust fallback behavior).
- Cleans text/HTML and semantically chunks content.
- Adds metadata and indexes chunks into FAISS.

### Tools Layer

- Files: `app/tools/weather.py`, `app/tools/news.py`
- Responsibilities:
- **WeatherTool**:
- Geocodes location, falls back to nearby match when exact location is unavailable.
- Fetches current weather + forecast from OpenWeatherMap.
- Returns structured data and nearby-location notes.
- **NewsTool**:
- Queries NewsAPI for `<location> travel news`.
- Returns top recent articles with URL/source/date.

### Memory Layer

- File: `app/memory/conversation.py`
- Responsibilities:
- Stores recent turns (`user_query`, `answer`, `actions_taken`).
- Stores tool call history.
- Stores pending permission state (e.g., weather approval handshake).

### Config Layer

- File: `app/core/config.py`
- Responsibilities:
- Loads env configuration and defaults.
- Centralizes keys/models/index path/log settings.
- Handles platform-sensitive defaults (`CRAWL4AI_ENABLED`).

## 4) Request Lifecycle (Step-by-Step)

1. Client sends `POST /query`.
2. API validates payload.
3. If location missing in vector DB, API runs `ingest_location(...)`.
4. Agent planner evaluates user intent:
5. Weather intent -> weather flow (with consent behavior).
6. Safety/news intent -> news tool.
7. Otherwise -> RAG-only.
8. Optional tool execution runs via `ToolExecutor`.
9. Responder retrieves top-k docs, merges context/tool data, formats final response.
10. Agent returns:
- `answer`
- `sources`
- `actions_taken`
- `suggested_followups`

## 5) Backend Data Contracts

### Query Request

```json
{
  "query": "Plan my trip to Puri",
  "location": "Puri",
  "weather_consent": true
}
```

### Query Response

```json
{
  "answer": "Day-wise plan ...\n\nSources:\nhttps://...",
  "sources": ["https://..."],
  "actions_taken": ["weather_tool:auto_first_answer"],
  "suggested_followups": ["...", "...", "..."]
}
```

## 6) Why This Design Works

- **Separation of concerns**: API, agent, retrieval, ingestion, tools, and memory are isolated.
- **Production-friendly behavior**: graceful fallbacks for crawl/tool failures.
- **Grounded outputs**: source URLs are carried through to final answer.
- **Extensibility**: new tools or retrievers can be introduced without rewriting API.
- **Low operational overhead**: FAISS local persistence keeps POC simple and fast.

## 7) Operational Notes

- Startup command:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

- Health endpoint: `GET /health`
- API docs: `GET /docs`
- FAISS index directory (default): `data/faiss_store`

## 8) Known Constraints and Practical Improvements

- Memory is in-process (not persistent across service restarts).
- Tool reliability depends on external API keys and quotas.
- Crawl quality depends on source page structure.

Recommended next upgrades:

- Persist memory in Redis/Postgres.
- Add API auth and request rate limiting.
- Add async job queue for ingestion.
- Add observability (structured tracing + metrics).
- Add automated evaluation suite for answer grounding.
