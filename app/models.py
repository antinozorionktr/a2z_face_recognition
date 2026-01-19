"""
SQLAlchemy ORM Models for Face Recognition Database

Defines the face_records table model matching the schema:
CREATE TABLE python_service.face_records (
    id UUID PRIMARY KEY,
    partner_id INTEGER NOT NULL,
    partner_name TEXT NOT NULL,
    image_name TEXT NOT NULL,
    embedding_index INTEGER NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class FaceRecordDB(Base):
    """
    SQLAlchemy model for face_records table.
    
    Stores partner information linked to face embeddings in FAISS.
    """
    __tablename__ = "face_records"
    __table_args__ = {"schema": "python_service"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    partner_id = Column(Integer, nullable=False, index=True)
    partner_name = Column(Text, nullable=False)
    image_name = Column(Text, nullable=False)
    embedding_index = Column(Integer, nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<FaceRecordDB(id={self.id}, partner_id={self.partner_id}, partner_name='{self.partner_name}')>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "partner_id": self.partner_id,
            "partner_name": self.partner_name,
            "image_name": self.image_name,
            "embedding_index": self.embedding_index,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }