"""
Face Recognition Service using DeepFace

This module handles all face recognition operations including:
- Face detection and extraction
- Embedding generation using ArcFace
- Preprocessing and alignment
"""
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple, List
import logging

from deepface import DeepFace

from app.config import (
    FACE_RECOGNITION_MODEL,
    FACE_DETECTOR_BACKEND,
    MAX_IMAGE_SIZE,
    EMBEDDING_DIM
)

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """
    Service class for face recognition operations.
    
    Uses DeepFace with ArcFace model for generating face embeddings.
    ArcFace is chosen for its high accuracy (99.83% on LFW benchmark)
    and robust performance across different conditions.
    """
    
    def __init__(self):
        """Initialize the face recognition service."""
        self.model_name = FACE_RECOGNITION_MODEL
        self.detector_backend = FACE_DETECTOR_BACKEND
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if not self._model_loaded:
            logger.info(f"Loading {self.model_name} model...")
            # Warm up the model by running a dummy inference
            try:
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                DeepFace.represent(
                    img_path=dummy_img,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False
                )
                self._model_loaded = True
                logger.info(f"{self.model_name} model loaded successfully")
            except Exception as e:
                logger.warning(f"Model warmup warning: {e}")
                self._model_loaded = True
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image bytes into numpy array.
        
        Steps:
        1. Load image from bytes
        2. Convert to RGB
        3. Resize if too large (preserving aspect ratio)
        4. Convert to numpy array
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed numpy array (RGB format)
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB (handles PNG with alpha, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large (preserve aspect ratio)
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                logger.debug(f"Image resized to {image.size}")
            
            # Convert to numpy array
            img_array = np.array(image)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def detect_face(self, img_array: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect and extract face from image.
        
        Uses RetinaFace for detection which provides:
        - High accuracy face detection
        - Facial landmark detection
        - Face alignment
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            Tuple of (face_detected: bool, aligned_face: Optional[np.ndarray])
        """
        try:
            # Extract faces using DeepFace
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True  # Align face using facial landmarks
            )
            
            if faces and len(faces) > 0:
                # Get the first (or largest) detected face
                face_obj = faces[0]
                face_array = face_obj.get("face")
                
                if face_array is not None:
                    # Convert from float [0,1] to uint8 [0,255] if needed
                    if face_array.max() <= 1.0:
                        face_array = (face_array * 255).astype(np.uint8)
                    return True, face_array
                    
            return False, None
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return False, None
    
    def generate_embedding(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from image.
        
        Pipeline:
        1. Face detection
        2. Face extraction & alignment
        3. Embedding generation using ArcFace
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            512-dimensional face embedding vector, or None if no face detected
        """
        self._ensure_model_loaded()
        
        try:
            # Generate embedding using DeepFace
            # This internally handles detection, alignment, and embedding
            embeddings = DeepFace.represent(
                img_path=img_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            if embeddings and len(embeddings) > 0:
                # Get embedding vector
                embedding = embeddings[0].get("embedding")
                
                if embedding is not None:
                    embedding_array = np.array(embedding, dtype=np.float32)
                    
                    # Normalize embedding for cosine similarity
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    return embedding_array
                    
            return None
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def generate_embedding_from_bytes(self, image_bytes: bytes) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Complete pipeline: bytes -> embedding.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (success: bool, embedding: Optional[np.ndarray])
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_bytes)
            
            # Generate embedding
            embedding = self.generate_embedding(img_array)
            
            if embedding is not None:
                return True, embedding
            else:
                return False, None
                
        except ValueError as e:
            logger.error(f"Pipeline failed: {e}")
            return False, None
    
    def verify_faces(
        self, 
        img1_array: np.ndarray, 
        img2_array: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Verify if two images contain the same person.
        
        Args:
            img1_array: First image as numpy array
            img2_array: Second image as numpy array
            
        Returns:
            Tuple of (verified: bool, distance: float)
        """
        try:
            result = DeepFace.verify(
                img1_path=img1_array,
                img2_path=img2_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric="cosine"
            )
            
            return result.get("verified", False), result.get("distance", 1.0)
            
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            return False, 1.0


# Singleton instance
face_service = FaceRecognitionService()