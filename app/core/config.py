import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
      # Gemini Model Configuration
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TEMPERATURE: float = 0.5
    GEMINI_MAX_OUTPUT_TOKENS: int = 512
    
    # Vector Store Configuration
    DB_FAISS_PATH: str = "vectorstore/db_faiss"
    EMBEDDING_MODEL: str = "models/embedding-001"
    RETRIEVAL_K: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
