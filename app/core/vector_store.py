"""Vector Store Module for Qdrant operations"""

from uuid import uuid4

from langchain_core.documents import Document
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

        # Ensure collection exists
        self.check_collection()

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

    def add_document(self, documents : list[Document]) -> list[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of doucments objects to add.

        Return:
            List of Document IDs
        """

        if not documents:
            logger.warning("No documents to add!")
            return []

        logger.debug(f"Adding {len(documents)} documents to the collection.")

        # Generate unique IDs for each document
        ids = [str(uuid4()) for _ in documents]

        # Add to vector store
        self.vector_store.add_documents(documents, ids)

        logger.info(f"Successfully added {len(documents)} documents.")
        return ids
        

    def search(self, query:str, k:int|None = None) -> list[Document]:
        """Search for the similar documents.

        Args:
            query: Search query
            k: Number of results to return(default from settings)

        Return:
            List of similar document objects
        """

        k = k or settings.retrieval_k

        logger.debug(f"Searching for {query[:20]}.. ")

        results = self.vector_store.similarity_search(query, k)

        logger.info(f"Found {len(results)} results.")
        return results

    def search_with_score(self, query:str, k:int|None=None) -> list[tuple[Document,int]]:
        """Search for the similar documents with relevance score.
        
        Args:
            query: Search the query
            k: Number of results to return(default from settings)

        Return:
            List of tuple of similar document with scores
        """

        k = k or settings.retrieval_k
        logger.debug(f"Searching for {query[:25]}...")

        results = self.vector_store.similarity_search_with_score(query, k=k)

        logger.info(f"Found {len(results)} results with scores.")
        return results
