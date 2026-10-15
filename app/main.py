"""FastAPI application entry point."""

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from app import __version__
from app.api.routes import document, health, query
from app.config import get_settings
from app.utils.logger import get_logger, setup_logger
from app.core.document_processor import DocumentProcessor

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logger(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting RAG Q&A v{__version__}")
    logger.info(f"Log level: {settings.log_level}")

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    description="""
## RAG Q&A System API

A Retrieval-Augmented Generation (RAG) question-answering system built with:
- **FastAPI** for the API layer
- **LangChain** for RAG orchestration
- **Qdrant Cloud** for vector storage
- **OpenAI** for embeddings and LLM

### Features
- Upload PDF documents
- Ask questions and get AI-powered answers
- View source documents for transparency
- Streaming responses for real-time feedback
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# Include routers
app.include_router(health.router)
app.include_router(document.router)
app.include_router(query.router)


@app.post("/upload-document/")
async def upload_document(file: UploadFile = File()):
    processor = DocumentProcessor()
    result_message = processor.process_upload(file=file.file, filename=file.filename)

    return{
        "status": "Success"   
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        reload=True,
    )