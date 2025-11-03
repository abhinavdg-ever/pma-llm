"""
Sleep Coach LLM FastAPI Service
FastAPI application for Sleep Coach LLM system with wearable data processing
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from dotenv import load_dotenv

# Import the sleep coach LLM system
from sleep_coach_llm import SleepCoachLLM, Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Coach LLM API",
    description="AI-powered Sleep Coach service with wearable data processing and LLM integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Sleep Coach LLM system
sleep_coach: Optional[SleepCoachLLM] = None

# Request/Response models
class QueryRequest(BaseModel):
    user_id: str
    query: str

class QueryResponse(BaseModel):
    query: str
    response_type: Optional[str] = None
    content: Optional[str] = None
    charts: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None

class WearableDataRequest(BaseModel):
    user_id: str
    date: str
    sleep_duration: Optional[float] = None
    deep_sleep: Optional[float] = None
    rem_sleep: Optional[float] = None
    light_sleep: Optional[float] = None
    awake_time: Optional[float] = None
    heart_rate: Optional[List[int]] = None
    movement: Optional[List[float]] = None
    respiration: Optional[List[float]] = None

class KnowledgeDocumentRequest(BaseModel):
    documents: List[Dict[str, str]]  # List of {content: str, source: str}

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    llm_available: bool
    sleep_coach_initialized: bool

@app.on_event("startup")
async def startup_event():
    """Initialize Sleep Coach LLM system on startup"""
    global sleep_coach
    logger.info("Starting up Sleep Coach LLM service...")
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize Sleep Coach LLM
        sleep_coach = SleepCoachLLM()
        logger.info("Sleep Coach LLM system initialized successfully")
        
        # Optionally add some default knowledge documents
        try:
            sleep_coach.add_knowledge_documents([
                {
                    "content": "Adults need 7-9 hours of sleep per night for optimal health.",
                    "source": "National Sleep Foundation"
                },
                {
                    "content": "Deep sleep is crucial for physical recovery and immune function.",
                    "source": "Journal of Sleep Research"
                },
                {
                    "content": "REM sleep plays a vital role in memory consolidation and emotional processing.",
                    "source": "Neuroscience & Biobehavioral Reviews"
                }
            ])
        except Exception as e:
            logger.warning(f"Could not add default knowledge documents: {e}")
            
    except Exception as e:
        logger.error(f"Error initializing Sleep Coach LLM: {e}")
        sleep_coach = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down Sleep Coach LLM service...")
    global sleep_coach
    if sleep_coach and sleep_coach.analytics_db.connection:
        sleep_coach.analytics_db.connection.close()
        logger.info("Database connection closed")

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Sleep Coach LLM API",
        "version": "2.0.0",
        "description": "AI-powered Sleep Coach service",
        "endpoints": {
            "/health": "Health check",
            "/query": "Ask questions about sleep data",
            "/wearable": "Submit wearable sleep data",
            "/knowledge": "Add knowledge documents to vector DB",
            "/stats": "Get database statistics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    global sleep_coach
    
    db_connected = False
    llm_available = False
    
    if sleep_coach:
        db_connected = sleep_coach.analytics_db.connection is not None and sleep_coach.analytics_db.connection.is_connected()
        # Test LLM by checking if agent is initialized
        llm_available = sleep_coach.agent.llm is not None
    
    sleep_coach_initialized = sleep_coach is not None
    
    status = "healthy" if (db_connected and llm_available and sleep_coach_initialized) else "degraded"
    
    return HealthResponse(
        status=status,
        database_connected=db_connected,
        llm_available=llm_available,
        sleep_coach_initialized=sleep_coach_initialized
    )

@app.post("/query", response_model=QueryResponse, tags=["Sleep Coach"])
async def handle_query(request: QueryRequest):
    """
    Handle user query about sleep data
    
    Examples:
    - "How was my sleep over the last 7 days?"
    - "What's my average sleep duration?"
    - "How does my deep sleep compare to others?"
    - "What's the importance of REM sleep?"
    """
    global sleep_coach
    
    if not sleep_coach:
        raise HTTPException(status_code=503, detail="Sleep Coach LLM system not initialized")
    
    try:
        logger.info(f"Processing query from user {request.user_id}: {request.query}")
        response = sleep_coach.handle_user_query(request.user_id, request.query)
        
        return QueryResponse(
            query=request.query,
            response_type=response.get("response_type"),
            content=response.get("content"),
            charts=response.get("charts"),
            debug=response.get("debug")
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/wearable", tags=["Data Ingestion"])
async def process_wearable_data(request: WearableDataRequest):
    """
    Process and store wearable sleep data
    
    Accepts raw wearable device data and stores it in the analytics database.
    """
    global sleep_coach
    
    if not sleep_coach:
        raise HTTPException(status_code=503, detail="Sleep Coach LLM system not initialized")
    
    try:
        logger.info(f"Processing wearable data for user {request.user_id} on {request.date}")
        
        # Prepare raw data dict
        raw_data = {
            "user_id": request.user_id,
            "date": request.date,
            "sleep_duration": request.sleep_duration or 0,
            "deep_sleep": request.deep_sleep or 0,
            "rem_sleep": request.rem_sleep or 0,
            "light_sleep": request.light_sleep or 0,
            "awake_time": request.awake_time or 0,
            "heart_rate": request.heart_rate or [],
            "movement": request.movement or [],
            "respiration": request.respiration or []
        }
        
        sleep_coach.process_wearable_data(raw_data)
        
        return {
            "status": "success",
            "message": f"Wearable data processed for user {request.user_id}",
            "date": request.date
        }
    except Exception as e:
        logger.error(f"Error processing wearable data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing wearable data: {str(e)}")

@app.post("/knowledge", tags=["Knowledge Base"])
async def add_knowledge_documents(request: KnowledgeDocumentRequest):
    """
    Add knowledge documents to the vector database
    
    This allows you to add sleep-related research and knowledge
    that the LLM can reference when answering questions.
    """
    global sleep_coach
    
    if not sleep_coach:
        raise HTTPException(status_code=503, detail="Sleep Coach LLM system not initialized")
    
    try:
        logger.info(f"Adding {len(request.documents)} knowledge documents")
        sleep_coach.add_knowledge_documents(request.documents)
        
        return {
            "status": "success",
            "message": f"Added {len(request.documents)} documents to knowledge base"
        }
    except Exception as e:
        logger.error(f"Error adding knowledge documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding knowledge documents: {str(e)}")

@app.get("/stats", tags=["Analytics"])
async def get_database_stats():
    """Get database statistics and customer overview"""
    global sleep_coach
    
    if not sleep_coach:
        raise HTTPException(status_code=503, detail="Sleep Coach LLM system not initialized")
    
    try:
        overview = sleep_coach.analytics_db.customers_overview(limit=20)
        return {
            "total_customers": len(overview),
            "customers": overview
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends/{user_id}", tags=["Analytics"])
async def get_user_trends(user_id: str, days: int = 7):
    """Get sleep trends for a specific user over specified days"""
    global sleep_coach
    
    if not sleep_coach:
        raise HTTPException(status_code=503, detail="Sleep Coach LLM system not initialized")
    
    try:
        trends = sleep_coach.analytics_db.calculate_trends(user_id, days)
        return trends
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
