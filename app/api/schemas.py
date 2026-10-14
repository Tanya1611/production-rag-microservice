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