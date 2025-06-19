import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.endpoint import router as api_router
from app.core.config import settings
from app.services.chatbot_service import ChatbotService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global chatbot service instance
chatbot_service: Optional[ChatbotService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global chatbot_service
    try:
        # Initialize chatbot service on startup
        logger.info("Initializing chatbot service...")
        chatbot_service = ChatbotService()
        await chatbot_service.initialize()
        logger.info("Chatbot service initialized successfully")
        
        # Store in app state for access in endpoints
        app.state.chatbot_service = chatbot_service
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot service: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down chatbot service...")
        if chatbot_service:
            await chatbot_service.cleanup()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Medical ChatBot API",
    description="FastAPI backend for medical chatbot using Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8001", "https://blood-donation-frontend-seven.vercel.app/"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Medical ChatBot API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "medical-chatbot-api"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
