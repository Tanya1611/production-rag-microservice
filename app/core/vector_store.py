"""Vector Store Module for Qdrant operations"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.utils.logger import get_logger
from app.core.embeddings import get_embeddings

logger = get_logger(__name__)
settings = get_settings()

#Embedding dimension
EMBEDDING_DIMENSION = 1536

def get_qdrant_client() -> QdrantClient:
    """ Get Qdrant client instance.
    
    Returns:
        Configured QdrantClient instance
    """
    logger.debug(f"Connecting to Qdrant at url: {settings.qdrant_url}")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

    logger.info("Qdrant lient connected successfully.")
    return client

class VectorStoreService:
    """Vector Store Service for managing its operations."""

    def __init__(self, collection_name: str|None = None):
        """Initialize vector store service.

        Args:
            collection_name: Name of the Qdrant collection 
        """

        self.connection_name = collection_name or settings.colllection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

        # initialize LangChain Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client= self.client,
            collection_name= self.connection_name,
            embedding= self.embeddings
        )

        logger.info(f"VectorStoreService initialized for collection: {self.connection_name} ")

    def check_collection(self) -> None:
        """Ensure the connection exists otherwise create it."""

        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists with {collection_info.points_count} points"
            )
        except UnexpectedResponse:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")


