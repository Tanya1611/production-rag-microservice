"""Embeddings Module to generate embeddings using OpenAI embeddings."""

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

def get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embedding instance
    
    Returns:
        Configured OpenAIEmbedding Instance
    """

    settings = get_settings()
    logger.debug(f"Initializing embedding model : {settings.embedding_model}.")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )

    logger.info("Initialized Embedding model successfully.")
    return embeddings

class embeddingService():
    """Service for generating embeddings."""

    def __init__(self):
        # Initialize embedding service.
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, query_text:str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query_text: Query text

        Returns:
            Embedding vector as list of floats
        """

        logger.debug(f"Generating embedding for query: {query_text[:50]}.")
        return self.embeddings.embed_query(query_text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        
        logger.debug(f"Generating embeddings for {len(texts)} documents.")
        return self.embeddings.embed_documents(texts)

