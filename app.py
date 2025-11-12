"""Contracts Copilot FastAPI Service."""

import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contract_llm import Config, ContractInsightsEngine


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("contract-insights-api")

app = FastAPI(
    title="Contracts Copilot",
    description="Ask questions about ILWU/PMA maritime agreements (2022-2028).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[ContractInsightsEngine] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    query: str
    response_type: Optional[str] = None
    content: str = ""
    answer_points: List[str] = Field(default_factory=list)
    disclaimer: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    total_matches: int = 0
    query_classification: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    has_vector_access: bool
    llama_available: bool


@app.on_event("startup")
async def startup_event() -> None:
    global engine
    try:
        Config.validate()
        engine = ContractInsightsEngine()
        logger.info("Contracts Copilot initialized.")
    except Exception as exc:  # pragma: no cover - startup failure logging.
        logger.exception("Failed to initialize Contracts Copilot: %s", exc)
        engine = None


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    return {
        "message": "Contracts Copilot",
        "version": "1.0.0",
        "description": "Embed questions, retrieve clauses from the 'contracts' collection, and synthesize answers.",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    has_vector = engine is not None and engine.vector_db.client is not None
    llama_available = engine is not None and engine.llama.available()
    status = "healthy" if has_vector else "degraded"
    return HealthResponse(
        status=status,
        has_vector_access=has_vector,
        llama_available=llama_available,
    )


@app.post("/query", response_model=QueryResponse, tags=["Contracts"])
async def handle_query(request: QueryRequest) -> QueryResponse:
    if not engine:
        raise HTTPException(status_code=503, detail="Contract engine not initialized.")
    try:
        result = engine.handle_query(request.query, top_k=request.top_k or 5)
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("Error processing query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
