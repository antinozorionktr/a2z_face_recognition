"""
Pydantic models for API request/response schemas

Updated to support partner_id and partner_name for PostgreSQL integration.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


class FaceRecordCreate(BaseModel):
    """Schema for creating a new face record"""
    partner_id: int = Field(..., description="Partner ID (integer identifier)")
    partner_name: str = Field(..., min_length=1, max_length=255, description="Partner name")

    class Config:
        json_schema_extra = {
            "example": {
                "partner_id": 12345,
                "partner_name": "Acme Corporation"
            }
        }


class FaceRecord(BaseModel):
    """Schema for a face record response"""
    id: str = Field(..., description="Unique UUID identifier for the record")
    partner_id: int = Field(..., description="Partner ID")
    partner_name: str = Field(..., description="Partner name")
    image_name: str = Field(..., description="Original image filename")
    embedding_index: int = Field(..., description="Index in FAISS vector store")
    is_active: bool = Field(default=True, description="Whether the record is active")
    created_at: datetime = Field(..., description="Timestamp when record was created")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "partner_id": 12345,
                "partner_name": "Acme Corporation",
                "image_name": "partner_photo.jpg",
                "embedding_index": 0,
                "is_active": True,
                "created_at": "2024-01-15T10:30:00"
            }
        }


class FaceRecordList(BaseModel):
    """Schema for listing all face records"""
    total_count: int = Field(..., description="Total number of records")
    records: List[FaceRecord] = Field(..., description="List of face records")


class MatchResult(BaseModel):
    """Schema for a single match result with complete partner data"""
    id: str = Field(..., description="Record UUID of the matched face")
    partner_id: int = Field(..., description="Partner ID of the matched record")
    partner_name: str = Field(..., description="Partner name of the matched record")
    image_name: str = Field(..., description="Original image filename")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1, higher is better)")
    distance: float = Field(..., description="Distance/dissimilarity score (lower is better)")
    is_active: bool = Field(default=True, description="Whether the record is active")
    created_at: Optional[datetime] = Field(default=None, description="Record creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "partner_id": 12345,
                "partner_name": "Acme Corporation",
                "image_name": "partner_photo.jpg",
                "confidence": 0.92,
                "distance": 0.08,
                "is_active": True,
                "created_at": "2024-01-15T10:30:00"
            }
        }


class MatchResponse(BaseModel):
    """Schema for match API response"""
    recognized: bool = Field(..., description="Whether a match was found above threshold")
    best_match: Optional[MatchResult] = Field(default=None, description="Best matching record if recognized")
    top_matches: List[MatchResult] = Field(..., description="Top-K nearest matches")
    face_detected: bool = Field(..., description="Whether a face was detected in input")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "recognized": True,
                "best_match": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "partner_id": 12345,
                    "partner_name": "Acme Corporation",
                    "image_name": "partner_photo.jpg",
                    "confidence": 0.92,
                    "distance": 0.08,
                    "is_active": True,
                    "created_at": "2024-01-15T10:30:00"
                },
                "top_matches": [],
                "face_detected": True,
                "processing_time_ms": 245.5
            }
        }


class CreateResponse(BaseModel):
    """Schema for create record response"""
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Status message")
    record: Optional[FaceRecord] = Field(default=None, description="Created record details")


class DeleteResponse(BaseModel):
    """Schema for delete record response"""
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Status message")
    deleted_id: Optional[str] = Field(default=None, description="UUID of deleted record")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "No face detected in the provided image"
            }
        }