from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional
import logging

from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.chatbot_service import ChatbotService

logger = logging.getLogger(__name__)

router = APIRouter()

def get_chatbot_service(request: Request) -> ChatbotService:
    """Dependency to get chatbot service from app state."""
    return request.app.state.chatbot_service

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
):
    """
    Chat endpoint for interacting with the medical chatbot.
    
    Args:
        request: ChatRequest containing user message and options
        chatbot_service: Injected chatbot service
        
    Returns:
        ChatResponse with AI assistant's reply
    """
    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(
                status_code=400, 
                detail="Message cannot be empty"
            )
          # Get response from chatbot service
        response = await chatbot_service.get_chat_response(
            message=request.message,
            include_sources=request.include_sources
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.get("/chat/health")
async def chat_health():
    """Health check endpoint for chat service."""
    return {"status": "healthy", "service": "chat-endpoint"}

@router.delete("/chat/session/{session_id}")
async def clear_session(
    session_id: str,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
):
    """
    Clear a specific chat session.
    
    Args:
        session_id: ID of the session to clear
        chatbot_service: Injected chatbot service
    """
    try:
        # Remove session from chatbot service
        if session_id in chatbot_service.chat_sessions:
            del chatbot_service.chat_sessions[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
            
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while clearing the session"
        )
