"""Pydantic Schemas for API request/response models"""

from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field


#------------Health Schemas-----------------

class HealthResponse(BaseModel):
    # Health Check Response

    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Response")
    version: str = Field(description="application version")

class ReadinessResponse(BaseModel):
    # Readiness Check Response
    status: str = Field(description="Service Status")
    qdrant_connected: bool = Field(description="Qdrant connection status")
    collection_info: dict = Field(description="Collection information")


#------------Document Schemas-----------------------

class DocumentUploadResponse(BaseModel):
    # Response after document upload.

    message: str = Field(description="Document status message")
    filename: str = Field(description="Uploaded filename")
    chunks_created: int = Field(description="Number of chunks created")
    document_ids: list[str] = Field(description="List of document IDs")

class DocumentInfo(BaseModel):
    # Document Information

    source: str = Field(description="Document source/filename")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )

class DocumentListResponse(BaseModel):
    # Response for listing documents

    collection_name: str = Field(description="Collection name")
    total_documents: int = Field(description="Total document count")
    status: str = Field(description="Collection status")

#-------------Query Schemas-----------------

class QueryRequest(BaseModel):
    # Request for RAG query

    question : str = Field(min_length=1, max_length=1000)
    include_sources : bool = Field(description="Include source documents in response")

    model_config = {
        "json_schema_extra":{
            "examples": [
                {
                    "question":"What is RAG?",
                    "incluse_sources":True
                }
            ]
        }
    }

class SourceDocument(BaseModel):
    # Source document information

    content : str = Field(description="Document content expected")
    metadata : dict[str, Any] = Field(description="Document metadata")

class QueryResponse(BaseModel):
    # Response for RAG Query

    question : str = Field(description="Original question")
    answer : str = Field(description="Generated answer")
    sources : list[SourceDocument] | None = Field(None, description="Source Document used")
    processing_time_ms : float = Field(description="Query processing time in milliseconds")


#-------------Error Schemas-----------------

class ErrorResponse(BaseModel):
    # Error Response

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: str|None = Field(default=None, description="Detailed error information")

class ValidationErrorResponse(BaseModel):
    # Validation error response

    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(description="Error message")
    errors: list[dict] = Field(description="Validation errors")