"""
FastAPI Application for Astro Conversational Insight Agent
Enhanced with MongoDB logging, session-wise configuration, and comprehensive logging
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging FIRST
from app.logging_config import setup_logging, RequestLogger
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

import logging
logger = logging.getLogger(__name__)

from app.llm_wrapper import create_llm, LLMWrapper
from app.retrieval import RetrievalEngine
from app.translation import TranslationService
from app.memory import SessionManager, get_session_manager
from app.agent import AstroConversationalAgent
from app.astrology_utils import AstrologyCalculator
from app.mongodb_logger import get_mongodb_logger
from app.session_config import (
    get_config_manager, 
    SessionConfig, 
    ConfigUpdateRequest,
    LLMProvider
)

logger.info("=" * 60)
logger.info("Starting Astro Conversational Insight Agent")
logger.info("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="Astro Conversational Insight Agent",
    description="RAG + Personalization based Astrological Insight API with MongoDB logging",
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

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models
class UserProfile(BaseModel):
    """User profile model"""
    name: str = Field(..., description="User's name")
    birth_date: str = Field(..., description="Birth date in YYYY-MM-DD format")
    birth_time: str = Field(..., description="Birth time in HH:MM format")
    birth_place: str = Field(..., description="Birth place")
    gender: Optional[str] = Field(None, description="Gender (optional)")
    goals: Optional[str] = Field(None, description="User goals (optional)")
    preferred_language: Optional[str] = Field("en", description="Preferred language (en/hi)")
    
    @validator('birth_date')
    def validate_birth_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("birth_date must be in YYYY-MM-DD format")
    
    @validator('birth_time')
    def validate_birth_time(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
            return v
        except ValueError:
            raise ValueError("birth_time must be in HH:MM format")
    
    @validator('preferred_language')
    def validate_language(cls, v):
        if v not in ['en', 'hi']:
            raise ValueError("preferred_language must be 'en' or 'hi'")
        return v


class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message/question")
    user_profile: UserProfile = Field(..., description="User's astrological profile")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Agent's response")
    context_used: List[str] = Field(..., description="Sources used for response")
    zodiac: str = Field(..., description="User's zodiac sign")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    error: Optional[str] = Field(None, description="Error message if any")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    active_sessions: int
    mongodb_status: str = "unknown"


# Global variables for services
session_manager: SessionManager = None
config_manager = None
mongodb_logger = None
llm_wrapper: LLMWrapper = None
retrieval_engine: RetrievalEngine = None
translation_service: TranslationService = None
agent: AstroConversationalAgent = None
request_logger: RequestLogger = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global session_manager, config_manager, mongodb_logger, llm_wrapper
    global retrieval_engine, translation_service, agent, request_logger
    
    logger.info("Initializing Astro Conversational Agent...")
    
    # Initialize request logger
    request_logger = RequestLogger()
    logger.info("✓ Request logger initialized")
    
    # Initialize MongoDB logger
    mongodb_logger = get_mongodb_logger()
    if mongodb_logger.connected:
        logger.info("✓ MongoDB logger connected")
    else:
        logger.warning("⚠ MongoDB logger not connected - logging to files only")
    
    # Initialize session manager
    session_manager = get_session_manager()
    logger.info("✓ Session manager initialized")
    
    # Initialize configuration manager with Ollama as default
    default_config = SessionConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model=os.getenv("LLM_MODEL", "llama3.1"),
        temperature=0.7
    )
    config_manager = get_config_manager()
    config_manager.default_config = default_config
    logger.info(f"✓ Config manager initialized (default: Ollama/{default_config.llm_model})")
    
    # Initialize LLM with default config
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    llm_model = os.getenv("LLM_MODEL", "llama3.1")
    
    try:
        if llm_provider == "openai":
            llm_wrapper = create_llm(
                provider="openai",
                model_name=llm_model or "gpt-3.5-turbo",
                temperature=0.7
            )
            logger.info(f"✓ OpenAI LLM initialized ({llm_model})")
        else:
            llm_wrapper = create_llm(
                provider="ollama",
                model_name=llm_model,
                temperature=0.7
            )
            logger.info(f"✓ Ollama LLM initialized ({llm_model})")
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        logger.info("Attempting fallback to Ollama...")
        try:
            llm_wrapper = create_llm(
                provider="ollama",
                model_name="llama3.1",
                temperature=0.7
            )
            logger.info("✓ Ollama LLM initialized (fallback)")
        except Exception as e2:
            logger.critical(f"Both LLM providers failed: {e2}")
            raise RuntimeError("Failed to initialize LLM")
    
    # Initialize retrieval engine
    try:
        retrieval_engine = RetrievalEngine(
            data_dir="./data",
            vector_store_type=os.getenv("VECTOR_STORE", "faiss"),
            persist_directory="./vector_store"
        )
        
        # Try to load existing vector store, otherwise build it
        try:
            retrieval_engine.load_vector_store()
            logger.info("✓ Vector store loaded from disk")
        except:
            logger.info("Building new vector store...")
            retrieval_engine.build_vector_store()
            logger.info("✓ Vector store built successfully")
    except Exception as e:
        logger.error(f"Error initializing retrieval engine: {e}")
        raise RuntimeError("Failed to initialize retrieval engine")
    
    # Initialize translation service
    translation_service = TranslationService()
    logger.info("✓ Translation service initialized")
    
    # Initialize agent
    agent = AstroConversationalAgent(
        llm_wrapper=llm_wrapper,
        retrieval_engine=retrieval_engine,
        translation_service=translation_service,
        enable_hallucination_filter=True
    )
    logger.info("✓ Agent initialized")
    
    logger.info("=" * 60)
    logger.info("🌟 Astro Conversational Agent ready!")
    logger.info(f"MongoDB: {'Connected' if mongodb_logger.connected else 'Not connected'}")
    logger.info(f"LLM Provider: {llm_provider}")
    logger.info(f"Model: {llm_model}")
    logger.info(f"Vector Store: {os.getenv('VECTOR_STORE', 'faiss')}")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """Serve the HTML UI"""
    return FileResponse("static/index.html")


@app.get("/api", response_model=Dict)
async def api_info():
    """API information endpoint"""
    return {
        "message": "Astro Conversational Insight Agent API",
        "version": "2.0.0",
        "features": [
            "MongoDB logging",
            "Session-wise configuration",
            "Ollama + OpenAI support",
            "Multi-language (English/Hindi)",
            "RAG with FAISS/ChromaDB"
        ],
        "endpoints": {
            "ui": "/ (HTML Interface)",
            "health": "/health",
            "chat": "/chat (POST)",
            "profile": "/profile (POST)",
            "session": "/session/{session_id} (GET, DELETE)",
            "config": "/config/{session_id} (GET, PUT)",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    mongodb_status = "connected" if mongodb_logger and mongodb_logger.connected else "disconnected"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        active_sessions=session_manager.get_active_sessions_count(),
        mongodb_status=mongodb_status
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for conversational astrological insights
    
    Args:
        request: Chat request containing session_id, message, and user_profile
        
    Returns:
        ChatResponse with agent's response and metadata
    """
    start_time = time.time()
    
    try:
        # Log request
        request_logger.log_request(request.session_id, "/chat", request.dict())
        logger.info(f"Chat request from session: {request.session_id}")

        # Get or create session
        session = session_manager.get_session(request.session_id)
        
        # Get session configuration
        session_config = config_manager.get_config(request.session_id)
        
        # Build or update user profile
        profile_dict = request.user_profile.dict()
        
        # Calculate astrological data if not in session yet
        if session.get_user_profile() is None:
            full_profile = AstrologyCalculator.build_user_profile(
                name=profile_dict["name"],
                birth_date=profile_dict["birth_date"],
                birth_time=profile_dict["birth_time"],
                birth_place=profile_dict["birth_place"],
                gender=profile_dict.get("gender"),
                goals=profile_dict.get("goals")
            )
            session.set_user_profile(full_profile)
            logger.info(f"Created profile for {full_profile['name']} - {full_profile['sun_sign']}")
        else:
            full_profile = session.get_user_profile()
        
        # Add user message to history
        session.add_message("user", request.message)
        
        # Get conversation history
        history = session.get_messages(last_n=10)
        
        # Process query through agent
        result = agent.process_query(
            query=request.message,
            user_profile=full_profile,
            conversation_history=history,
            language=profile_dict.get("preferred_language", "en")
        )
        
        # Add assistant response to history
        session.add_message("assistant", result["response"], {
            "context_used": result["context_used"],
            "confidence_score": result["confidence_score"]
        })
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log to MongoDB in background if enabled
        if session_config.log_to_mongodb and mongodb_logger and mongodb_logger.connected:
            background_tasks.add_task(
                mongodb_logger.log_request_response,
                session_id=request.session_id,
                request_data={
                    "message": request.message,
                    "user_profile": profile_dict,
                    "language": profile_dict.get("preferred_language", "en")
                },
                response_data=result,
                metadata={
                    "duration": duration,
                    "session_config": session_config.dict()
                }
            )
        
        # Log response
        request_logger.log_response(request.session_id, "/chat", 200, duration)
        logger.info(f"Response generated in {duration:.2f}s - Confidence: {result['confidence_score']:.2f}")
        
        # Return response
        return ChatResponse(
            response=result["response"],
            context_used=result["context_used"],
            zodiac=result["zodiac"],
            confidence_score=result["confidence_score"],
            error=result.get("error")
        )
        
    except Exception as e:
        duration = time.time() - start_time
        request_logger.log_error(request.session_id, "/chat", str(e))
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses
    
    Args:
        request: Chat request containing session_id, message, and user_profile
        
    Returns:
        Server-Sent Events stream
    """
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        try:
            # Get or create session
            session = session_manager.get_session(request.session_id)
            
            # Build or update user profile
            profile_dict = request.user_profile.dict()
            
            if session.get_user_profile() is None:
                full_profile = AstrologyCalculator.build_user_profile(
                    name=profile_dict["name"],
                    birth_date=profile_dict["birth_date"],
                    birth_time=profile_dict["birth_time"],
                    birth_place=profile_dict["birth_place"],
                    gender=profile_dict.get("gender"),
                    goals=profile_dict.get("goals")
                )
                session.set_user_profile(full_profile)
            else:
                full_profile = session.get_user_profile()
            
            # Add user message
            session.add_message("user", request.message)
            history = session.get_messages(last_n=10)
            
            # Send metadata first
            yield f"data: {json.dumps({'type': 'metadata', 'zodiac': full_profile.get('sun_sign', 'Unknown')})}\n\n"
            
            # Stream response from agent
            full_response = ""
            final_response = ""
            context_used = []
            confidence_score = 0.0
            
            async for event in agent.process_query_stream(
                query=request.message,
                user_profile=full_profile,
                conversation_history=history,
                language=profile_dict.get("preferred_language", "en")
            ):
                yield f"data: {json.dumps(event)}\n\n"
                
                if event.get("type") == "chunk":
                    full_response += event.get("content", "")
                elif event.get("type") == "clear":
                    full_response = ""
                elif event.get("type") == "done":
                    # Get final response (could be translated)
                    final_response = event.get("final_response", full_response)
                    context_used = event.get("context_used", [])
                    confidence_score = event.get("confidence_score", 0.0)
                    
                    # Add to history with final response
                    session.add_message("assistant", final_response, {
                        "context_used": context_used,
                        "confidence_score": confidence_score
                    })
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/profile")
async def build_profile(profile: UserProfile):
    """
    Build astrological profile from birth details
    
    Args:
        profile: User profile with birth details
        
    Returns:
        Complete astrological profile
    """
    try:
        full_profile = AstrologyCalculator.build_user_profile(
            name=profile.name,
            birth_date=profile.birth_date,
            birth_time=profile.birth_time,
            birth_place=profile.birth_place,
            gender=profile.gender,
            goals=profile.goals
        )
        
        return {
            "status": "success",
            "profile": full_profile
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error building profile: {str(e)}")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get session information
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session data
    """
    try:
        session = session_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "message_count": len(session.get_messages()),
            "user_profile": session.get_user_profile(),
            "last_updated": session.last_updated.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        session_manager.delete_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session (keeping profile)
    
    Args:
        session_id: Session identifier
        
    Returns:
        Confirmation message
    """
    try:
        session = session_manager.get_session(session_id)
        session.clear()
        logger.info(f"Cleared session: {session_id}")
        return {
            "status": "success",
            "message": f"Session {session_id} history cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/config/{session_id}")
async def get_session_config(session_id: str):
    """
    Get configuration for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session configuration
    """
    try:
        config = config_manager.get_config(session_id)
        return {
            "session_id": session_id,
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@app.put("/config/{session_id}")
async def update_session_config(session_id: str, config_update: ConfigUpdateRequest):
    """
    Update configuration for a session
    
    Args:
        session_id: Session identifier
        config_update: Configuration fields to update
        
    Returns:
        Updated configuration
    """
    try:
        # Get updates that are not None
        updates = {k: v for k, v in config_update.dict().items() if v is not None}
        
        # Update config
        config_manager.update_config(session_id, **updates)
        
        # Get updated config
        updated_config = config_manager.get_config(session_id)
        
        logger.info(f"Updated config for session {session_id}: {updates}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "config": updated_config.dict()
        }
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """
    Get system statistics
    
    Returns:
        Statistics including MongoDB data
    """
    try:
        stats = {
            "active_sessions": session_manager.get_active_sessions_count(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add MongoDB statistics if connected
        if mongodb_logger and mongodb_logger.connected:
            mongo_stats = mongodb_logger.get_statistics()
            stats["mongodb"] = mongo_stats
        else:
            stats["mongodb"] = {"connected": False}
        
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )