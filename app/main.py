"""
Face Recognition API

A robust facial identification and recognition API powered by DeepFace and FAISS,
with PostgreSQL for metadata storage.

Endpoints:
- POST /records - Create a new face record
- GET /records - List all face records
- POST /records/match - Match a face against the database
- DELETE /records/{record_id} - Delete a face record
"""
import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    SUPPORTED_FORMATS,
    TOP_K_MATCHES,
    RECOGNITION_THRESHOLD,
    FACE_RECOGNITION_MODEL,
    FACE_DETECTOR_BACKEND
)
from app.schemas import (
    FaceRecord,
    FaceRecordList,
    MatchResult,
    MatchResponse,
    CreateResponse,
    DeleteResponse,
    ErrorResponse
)
from app.face_service import face_service
from app.vector_store import vector_store
from app.database import async_session_maker, init_db, close_db
from app.repository import FaceRecordRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Face Recognition API...")
    logger.info(f"Model: {FACE_RECOGNITION_MODEL}")
    logger.info(f"Detector: {FACE_DETECTOR_BACKEND}")
    
    # Initialize database
    await init_db()
    
    logger.info(f"FAISS index has {vector_store.count} vectors")
    yield
    
    # Shutdown
    await close_db()
    logger.info("Shutting down Face Recognition API...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    # Check file extension
    ext = "." + file.filename.lower().split(".")[-1] if "." in file.filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )


@app.get("/", include_in_schema=False)
async def root(db: AsyncSession = Depends(get_db)):
    """Root endpoint with API info."""
    total_records = await FaceRecordRepository.count(db)
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "model": FACE_RECOGNITION_MODEL,
        "detector": FACE_DETECTOR_BACKEND,
        "total_records": total_records,
        "faiss_vectors": vector_store.count,
        "endpoints": {
            "create": "POST /records",
            "list": "GET /records",
            "match": "POST /records/match",
            "delete": "DELETE /records/{record_id}"
        }
    }


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint."""
    try:
        db_count = await FaceRecordRepository.count(db)
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_count = 0
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "model_loaded": face_service._model_loaded,
        "database_status": db_status,
        "total_records": db_count,
        "faiss_vectors": vector_store.count
    }


# ============================================================================
# API 1: CREATE RECORD
# ============================================================================
@app.post(
    "/records",
    response_model=CreateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "No face detected"}
    },
    summary="Create a new face record",
    description="""
    Register a new face in the system with partner information.
    
    **Pipeline:**
    1. Image preprocessing (resize, convert to RGB)
    2. Face detection using RetinaFace
    3. Face extraction and alignment
    4. Embedding generation using ArcFace
    5. Store embedding in FAISS index
    6. Store partner metadata in PostgreSQL
    
    **Requirements:**
    - Image must contain exactly one clear face
    - Supported formats: JPG, PNG, WebP, BMP
    - Face should be frontal or near-frontal for best results
    """
)
async def create_record(
    image: UploadFile = File(..., description="Face image file"),
    partner_id: int = Form(..., description="Partner ID (integer identifier)"),
    partner_name: str = Form(..., min_length=1, max_length=255, description="Partner name"),
    db: AsyncSession = Depends(get_db)
):
    """Create a new face record with the provided image and partner info."""
    start_time = time.time()
    
    # Validate image
    validate_image_file(image)
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
    
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # Generate embedding
    success, embedding = face_service.generate_embedding_from_bytes(image_bytes)
    
    if not success or embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the provided image. Please ensure the image contains a clear, frontal face."
        )
    
    # Generate record ID
    record_id = uuid.uuid4()
    
    # Add embedding to FAISS
    try:
        embedding_index = vector_store.add(embedding=embedding)
    except Exception as e:
        logger.error(f"Failed to add embedding to FAISS: {e}")
        raise HTTPException(status_code=500, detail="Failed to store face embedding")
    
    # Store metadata in PostgreSQL
    try:
        db_record = await FaceRecordRepository.create(
            session=db,
            record_id=record_id,
            partner_id=partner_id,
            partner_name=partner_name,
            image_name=image.filename or "unknown",
            embedding_index=embedding_index
        )
    except Exception as e:
        # Rollback FAISS addition if DB fails
        vector_store.delete(embedding_index)
        logger.error(f"Failed to store record in database: {e}")
        raise HTTPException(status_code=500, detail="Failed to store face record in database")
    
    # Convert to response schema
    record = FaceRecordRepository.db_to_schema(db_record)
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Created record for partner {partner_id} ({record.id}) in {processing_time:.1f}ms")
    
    return CreateResponse(
        success=True,
        message=f"Face record created successfully for partner '{partner_name}'",
        record=record
    )


# ============================================================================
# API 2: LIST RECORDS
# ============================================================================
@app.get(
    "/records",
    response_model=FaceRecordList,
    summary="List all face records",
    description="Retrieve all registered face records from the database."
)
async def list_records(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db)
):
    """List all registered face records with pagination."""
    total_count = await FaceRecordRepository.count(db)
    db_records = await FaceRecordRepository.get_all(db, skip=skip, limit=limit)
    
    records = [FaceRecordRepository.db_to_schema(r) for r in db_records]
    
    return FaceRecordList(
        total_count=total_count,
        records=records
    )


# ============================================================================
# API 3: MATCH RECORD
# ============================================================================
@app.post(
    "/records/match",
    response_model=MatchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "No face detected"}
    },
    summary="Match a face against the database",
    description="""
    Find matching faces for an input image and return complete partner data.
    
    **Pipeline:**
    1. Image preprocessing
    2. Face detection and extraction
    3. Generate query embedding
    4. FAISS similarity search (cosine distance)
    5. Retrieve partner data from PostgreSQL
    6. Return Top-K nearest matches with full partner info
    
    **Output:**
    - `recognized`: True if best match is above threshold
    - `best_match`: Best matching record with complete partner data
    - `top_matches`: Top-K similar faces with confidence scores and partner info
    """
)
async def match_record(
    image: UploadFile = File(..., description="Face image to match"),
    top_k: int = Query(TOP_K_MATCHES, ge=1, le=20, description="Number of top matches to return"),
    threshold: float = Query(
        RECOGNITION_THRESHOLD, 
        ge=0.0, 
        le=1.0, 
        description="Recognition threshold (lower = stricter)"
    ),
    db: AsyncSession = Depends(get_db)
):
    """Match an input face against all registered faces."""
    start_time = time.time()
    
    # Validate image
    validate_image_file(image)
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
    
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # Check if we have any records
    if vector_store.count == 0:
        processing_time = (time.time() - start_time) * 1000
        return MatchResponse(
            recognized=False,
            best_match=None,
            top_matches=[],
            face_detected=False,
            processing_time_ms=round(processing_time, 2)
        )
    
    # Generate embedding
    success, embedding = face_service.generate_embedding_from_bytes(image_bytes)
    
    if not success or embedding is None:
        processing_time = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=422,
            detail="No face detected in the provided image. Please ensure the image contains a clear, frontal face."
        )
    
    # Search in FAISS
    recognized, faiss_matches = vector_store.search(
        query_embedding=embedding,
        top_k=top_k,
        threshold=threshold
    )
    
    if not faiss_matches:
        processing_time = (time.time() - start_time) * 1000
        return MatchResponse(
            recognized=False,
            best_match=None,
            top_matches=[],
            face_detected=True,
            processing_time_ms=round(processing_time, 2)
        )
    
    # Get embedding indices from matches
    embedding_indices = [m[0] for m in faiss_matches]
    
    # Fetch partner data from PostgreSQL
    db_records_map = await FaceRecordRepository.get_multiple_by_embedding_indices(db, embedding_indices)
    
    # Build match results with full partner data
    top_matches = []
    for embedding_index, confidence, distance in faiss_matches:
        db_record = db_records_map.get(embedding_index)
        if db_record:
            match_result = MatchResult(
                id=str(db_record.id),
                partner_id=db_record.partner_id,
                partner_name=db_record.partner_name,
                image_name=db_record.image_name,
                confidence=confidence,
                distance=distance,
                is_active=db_record.is_active,
                created_at=db_record.created_at
            )
            top_matches.append(match_result)
    
    # Determine best match
    best_match = top_matches[0] if recognized and top_matches else None
    
    processing_time = (time.time() - start_time) * 1000
    
    if recognized and best_match:
        logger.info(
            f"Matched face to partner {best_match.partner_id} ('{best_match.partner_name}') "
            f"(confidence: {best_match.confidence:.2%}) in {processing_time:.1f}ms"
        )
    else:
        top_score = top_matches[0].confidence if top_matches else "N/A"
        logger.info(f"No match found (top score: {top_score}) in {processing_time:.1f}ms")
    
    return MatchResponse(
        recognized=recognized,
        best_match=best_match,
        top_matches=top_matches,
        face_detected=True,
        processing_time_ms=round(processing_time, 2)
    )


# ============================================================================
# API 4: DELETE RECORD
# ============================================================================
@app.delete(
    "/records/{record_id}",
    response_model=DeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Record not found"}
    },
    summary="Delete a face record",
    description="Remove a registered face from the database by its UUID."
)
async def delete_record(
    record_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a face record by its ID."""
    # Check if record exists
    existing = await FaceRecordRepository.get_by_id(db, record_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Record with ID '{record_id}' not found"
        )
    
    # Delete from FAISS
    vector_store.delete(existing.embedding_index)
    
    # Soft delete from database
    success = await FaceRecordRepository.soft_delete(db, record_id)
    
    if success:
        logger.info(f"Deleted record {record_id} (partner: {existing.partner_name})")
        return DeleteResponse(
            success=True,
            message=f"Successfully deleted record for partner '{existing.partner_name}'",
            deleted_id=record_id
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete record"
        )


# ============================================================================
# Additional Utility Endpoints
# ============================================================================
@app.get(
    "/records/{record_id}",
    response_model=FaceRecord,
    responses={
        404: {"model": ErrorResponse, "description": "Record not found"}
    },
    summary="Get a specific face record",
    description="Retrieve details of a specific face record by its UUID."
)
async def get_record(
    record_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific face record by ID."""
    db_record = await FaceRecordRepository.get_by_id(db, record_id)
    if not db_record:
        raise HTTPException(
            status_code=404,
            detail=f"Record with ID '{record_id}' not found"
        )
    return FaceRecordRepository.db_to_schema(db_record)


@app.get(
    "/partners/{partner_id}/records",
    response_model=FaceRecordList,
    summary="Get all records for a partner",
    description="Retrieve all face records associated with a specific partner ID."
)
async def get_partner_records(
    partner_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all face records for a specific partner."""
    db_records = await FaceRecordRepository.get_by_partner_id(db, partner_id)
    records = [FaceRecordRepository.db_to_schema(r) for r in db_records]
    
    return FaceRecordList(
        total_count=len(records),
        records=records
    )


@app.post(
    "/maintenance/rebuild-index",
    summary="Rebuild FAISS index",
    description="Rebuild the FAISS index to reclaim space from deleted entries. Use during maintenance."
)
async def rebuild_index(db: AsyncSession = Depends(get_db)):
    """Rebuild the FAISS index to optimize storage."""
    start_time = time.time()
    
    count_before = vector_store.count
    count_after = vector_store.rebuild_index()
    
    processing_time = (time.time() - start_time) * 1000
    
    db_count = await FaceRecordRepository.count(db)
    
    return {
        "success": True,
        "message": "Index rebuilt successfully",
        "faiss_vectors_before": count_before,
        "faiss_vectors_after": count_after,
        "database_records": db_count,
        "processing_time_ms": round(processing_time, 2)
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)