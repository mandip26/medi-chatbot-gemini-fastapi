import os
import logging
import google.generativeai as genai
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from core.config import settings
from models.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)

class GoogleEmbeddings(Embeddings):
    """Custom Google GenerativeAI Embeddings wrapper for langchain compatibility."""
    
    def __init__(self, model_name: str):
        """Initialize Google embedding model."""
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = model_name
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text
            )
            embeddings.append(result['embedding'])
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = genai.embed_content(
            model=self.model_name,
            content=text
        )
        return result['embedding']

class ChatbotService:
    """Service class for handling chatbot operations with Gemini AI."""
    
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.embeddings: Optional[GoogleEmbeddings] = None
        
    async def initialize(self):
        """Initialize the chatbot service."""
        try:
            # Configure Gemini AI
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
                
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Initialize embeddings
            self.embeddings = GoogleEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )            
            # Load vectorstore
            await self._load_vectorstore()
            
            logger.info("ChatbotService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotService: {e}")
            raise
    
    async def _load_vectorstore(self):
        """Load the FAISS vectorstore."""
        try:
            # Get absolute path to vectorstore
            current_dir = os.path.dirname(os.path.abspath(__file__))
            app_dir = os.path.dirname(current_dir)
            vectorstore_path = os.path.join(app_dir, settings.DB_FAISS_PATH)
            
            logger.info(f"Looking for vectorstore at: {vectorstore_path}")
            
            if not os.path.exists(vectorstore_path):
                logger.warning(f"Vectorstore path not found: {vectorstore_path}")
                logger.info("Running without RAG capabilities - using Gemini model only")
                return
                
            # Check if required files exist
            index_file = os.path.join(vectorstore_path, "index.faiss")
            pkl_file = os.path.join(vectorstore_path, "index.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(pkl_file):
                logger.warning(f"Vectorstore files missing in {vectorstore_path}")
                logger.info("Running without RAG capabilities - using Gemini model only")
                return
                
            self.vectorstore = FAISS.load_local(
                vectorstore_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vectorstore loaded successfully from {vectorstore_path}")
            
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            logger.info("Continuing without RAG capabilities")
    
    async def get_chat_response(
        self, 
        message: str, 
        include_sources: bool = False
    ) -> ChatResponse:
        """        Get a chat response using Gemini AI with optional RAG.
        
        Args:
            message: User's message
            include_sources: Whether to include source documents
            
        Returns:
            ChatResponse object
        """
        try:
            # Get context from vectorstore if available
            context = ""
            sources = []
            
            if self.vectorstore:
                context, sources = await self._get_context_and_sources(message)                
                logger.info(f"Retrieved context from vectorstore: {len(context)} characters")
            else:
                logger.info("No vectorstore available, using Gemini without RAG")
            
            # Prepare the prompt
            prompt = self._create_prompt(message, context)
            
            # Get response from Gemini
            response_text = await self._get_gemini_response(prompt)
            
            return ChatResponse(
                message=response_text,
                sources=sources if include_sources else None
            )
            
        except Exception as e:
            logger.error(f"Error getting chat response: {e}")
            raise

    async def _get_context_and_sources(self, query: str) -> tuple[str, List[SourceDocument]]:
        """Retrieve context and sources from vectorstore."""
        try:
            logger.info(f"Searching vectorstore for query: {query[:50]}...")
            
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(
                query, 
                k=settings.RETRIEVAL_K
            )
            
            logger.info(f"Found {len(docs)} relevant documents")
            
            # Extract context and create source documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs):
                logger.debug(f"Document {i}: {doc.page_content[:100]}...")
                context_parts.append(doc.page_content)
                sources.append(SourceDocument(
                    content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    metadata=doc.metadata
                ))
            
            context = "\n\n".join(context_parts)
            logger.info(f"Total context length: {len(context)} characters")
            return context, sources
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "", []
    
    def _create_prompt(self, question: str, context: str = "") -> str:
        """Create a prompt for Gemini AI."""
        prompt = f"""
        Use the pieces of information provided in the context to answer user's questions.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide accurate and helpful medical information while emphasizing that this is not a substitute for professional medical advice.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        return prompt
    
    async def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini AI."""
        try:
            # Create a new model instance for each request (no session management)
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=settings.GEMINI_TEMPERATURE,
                    max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
                )
            )
            
            # Send message and get response
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("ChatbotService cleanup completed")
