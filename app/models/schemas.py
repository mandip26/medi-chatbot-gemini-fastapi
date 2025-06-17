from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message model."""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question or message")
    include_sources: bool = Field(True, description="Whether to include source documents in response")

class SourceDocument(BaseModel):
    """Source document model for RAG responses."""
    content: str = Field(..., description="Content of the source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="AI assistant's response")
    sources: Optional[List[SourceDocument]] = Field(None, description="Source documents used for response")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
