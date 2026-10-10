"""RAG Chain Module using LancgChain LCEL"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI

from app.config import get_settings
from app.utils.logger import get_logger
from app.core.vector_store import VectorStoreService

logger = get_logger(__name__)
settings = get_settings()

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the provided context, say 'I don't have enough information to answer the question!' 

Do not make up the information. Only use the provided context.

Context: {context}
Question: {question}
Answer:
"""

def format_docs(docs: list[Document]) -> str:
    """Format documents into a single context string
    
    Args:
        Formatted context string
    """

    return "\n\n-------\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG Chain for Question-Answer"""

    def __init__(self, vector_store_service : VectorStoreService | None = None):
        """ Initialize RAG Chain

        Args:
        """

        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key
            )

        # Create Prompt Template
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Build LCEL Chain
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"RAGChain initialized with model={settings.llm_model}, retrieval_k={settings.retrieval_k}.")

    async def query(self, question:str) -> str:
        """Execute async RAG query.
            
        Args:
            question: User question
                
        Return:
            Generated answer
        """
    
        logger.debug(f"Processing async query: {question[:50]}...")

        try:
            answer = await self.chain.ainvoke(question)
            logger.info("Async query processed successfully.")
            return answer

        except Exception as e:
            logger.error(f"Error processing async query: {e}")
            raise

    async def query_with_sources(self, question:str) -> dict:
        """Execute an async RAG query and return sources.
        
        Args:
            question: User question
        
        Return:
            Dictionary with answer and source documents
        """

        logger.debug(f"Processing async query with sources: {question[:50]}...")

        try:
            answer = await self.chain.ainvoke(question)
            source_docs = self.retriever.invoke(question)

            #Format sources
            sources = [
                {
                    "content": (doc.page_content[:500]+'...' if len(doc.page_content)>500 else doc.page_content),
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]

            logger.info(f"Async query processed with {len(sources)} sources.")

            return{
                "answer":answer,
                "sources":sources
            }
        except Exception as e:
            logger.error(f"Error processing async query with sources: {e}")
            raise

    async def stream(self, question:str):
        """Stram RAG Response
        
        Args:
            question: User question
            
        Yield:
            Response chunks
        """

        logger.debug(f"Streaming query: {question[:50]}..")

        try:
            for chunk in self.chain.astream(question):
                yield f"\n\n{chunk}"

        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise

