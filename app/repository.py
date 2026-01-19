"""
Face Records Repository

Database operations for face_records table using SQLAlchemy async.
Provides CRUD operations and query methods.
"""
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.models import FaceRecordDB
from app.schemas import FaceRecord

logger = logging.getLogger(__name__)


class FaceRecordRepository:
    """
    Repository class for face_records database operations.
    
    All methods are async and require an AsyncSession.
    """
    
    @staticmethod
    async def create(
        session: AsyncSession,
        record_id: uuid.UUID,
        partner_id: int,
        partner_name: str,
        image_name: str,
        embedding_index: int
    ) -> FaceRecordDB:
        """
        Create a new face record in the database.
        
        Args:
            session: Database session
            record_id: UUID for the record
            partner_id: Partner identifier
            partner_name: Partner name
            image_name: Original image filename
            embedding_index: Index in FAISS vector store
            
        Returns:
            Created FaceRecordDB instance
        """
        db_record = FaceRecordDB(
            id=record_id,
            partner_id=partner_id,
            partner_name=partner_name,
            image_name=image_name,
            embedding_index=embedding_index,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        session.add(db_record)
        await session.commit()
        await session.refresh(db_record)
        
        logger.info(f"Created DB record {record_id} for partner {partner_id}")
        return db_record
    
    @staticmethod
    async def get_by_id(session: AsyncSession, record_id: str) -> Optional[FaceRecordDB]:
        """Get a face record by its UUID."""
        try:
            record_uuid = uuid.UUID(record_id)
        except ValueError:
            return None
        
        result = await session.execute(
            select(FaceRecordDB)
            .where(FaceRecordDB.id == record_uuid)
            .where(FaceRecordDB.is_active == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_embedding_index(session: AsyncSession, embedding_index: int) -> Optional[FaceRecordDB]:
        """Get a face record by its FAISS embedding index."""
        result = await session.execute(
            select(FaceRecordDB)
            .where(FaceRecordDB.embedding_index == embedding_index)
            .where(FaceRecordDB.is_active == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_partner_id(session: AsyncSession, partner_id: int) -> List[FaceRecordDB]:
        """Get all face records for a partner."""
        result = await session.execute(
            select(FaceRecordDB)
            .where(FaceRecordDB.partner_id == partner_id)
            .where(FaceRecordDB.is_active == True)
            .order_by(FaceRecordDB.created_at.desc())
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_all(
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = True
    ) -> List[FaceRecordDB]:
        """Get all face records with pagination."""
        query = select(FaceRecordDB)
        
        if active_only:
            query = query.where(FaceRecordDB.is_active == True)
        
        query = query.order_by(FaceRecordDB.created_at.desc()).offset(skip).limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def count(session: AsyncSession, active_only: bool = True) -> int:
        """Get total count of face records."""
        from sqlalchemy import func
        
        query = select(func.count(FaceRecordDB.id))
        if active_only:
            query = query.where(FaceRecordDB.is_active == True)
        
        result = await session.execute(query)
        return result.scalar() or 0
    
    @staticmethod
    async def soft_delete(session: AsyncSession, record_id: str) -> bool:
        """
        Soft delete a face record by setting is_active to False.
        
        Returns:
            True if deleted, False if not found
        """
        try:
            record_uuid = uuid.UUID(record_id)
        except ValueError:
            return False
        
        result = await session.execute(
            update(FaceRecordDB)
            .where(FaceRecordDB.id == record_uuid)
            .where(FaceRecordDB.is_active == True)
            .values(is_active=False)
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.info(f"Soft deleted DB record {record_id}")
            return True
        return False
    
    @staticmethod
    async def hard_delete(session: AsyncSession, record_id: str) -> bool:
        """
        Permanently delete a face record.
        
        Returns:
            True if deleted, False if not found
        """
        try:
            record_uuid = uuid.UUID(record_id)
        except ValueError:
            return False
        
        result = await session.execute(
            delete(FaceRecordDB).where(FaceRecordDB.id == record_uuid)
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.info(f"Hard deleted DB record {record_id}")
            return True
        return False
    
    @staticmethod
    async def get_multiple_by_embedding_indices(
        session: AsyncSession, 
        indices: List[int]
    ) -> dict[int, FaceRecordDB]:
        """
        Get multiple face records by their embedding indices.
        
        Returns:
            Dictionary mapping embedding_index to FaceRecordDB
        """
        if not indices:
            return {}
        
        result = await session.execute(
            select(FaceRecordDB)
            .where(FaceRecordDB.embedding_index.in_(indices))
            .where(FaceRecordDB.is_active == True)
        )
        
        records = result.scalars().all()
        return {record.embedding_index: record for record in records}
    
    @staticmethod
    def db_to_schema(db_record: FaceRecordDB) -> FaceRecord:
        """Convert database model to Pydantic schema."""
        return FaceRecord(
            id=str(db_record.id),
            partner_id=db_record.partner_id,
            partner_name=db_record.partner_name,
            image_name=db_record.image_name,
            embedding_index=db_record.embedding_index,
            is_active=db_record.is_active,
            created_at=db_record.created_at
        )