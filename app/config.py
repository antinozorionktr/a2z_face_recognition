"""
Configuration settings for Face Recognition Application
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# DeepFace Configuration
# Using ArcFace - the BEST model for face recognition accuracy
# ArcFace achieves 99.83% accuracy on LFW benchmark
# Requires more GPU/RAM but provides highest accuracy
FACE_RECOGNITION_MODEL = "ArcFace"

# Face Detection Backend
# RetinaFace provides the best detection accuracy
FACE_DETECTOR_BACKEND = "retinaface"

# Distance metric for similarity
# Cosine similarity works best with normalized embeddings
DISTANCE_METRIC = "cosine"

# FAISS Configuration
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

# Embedding dimension for ArcFace
EMBEDDING_DIM = 512

# Recognition Thresholds (STRICTER for better accuracy)
# Lower threshold = stricter matching
# ArcFace with cosine distance thresholds:
# - Very strict: 0.30 (high security)
# - Strict: 0.35
# - Normal: 0.40
# - Relaxed: 0.45
RECOGNITION_THRESHOLD = 0.35  # Strict threshold for high accuracy

# Top-K matches to return
TOP_K_MATCHES = 5

# Image preprocessing
MAX_IMAGE_SIZE = (1024, 1024)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# =============================================================================
# FACE QUALITY VALIDATION SETTINGS
# =============================================================================

# Minimum face size (as fraction of image)
# Rejects very small faces which are often false positives
MIN_FACE_SIZE_RATIO = 0.05  # Face must be at least 5% of image area

# Minimum face detection confidence
MIN_DETECTION_CONFIDENCE = 0.90  # 90% confidence required

# Minimum face area in pixels (width * height)
MIN_FACE_AREA_PIXELS = 10000  # ~100x100 pixels minimum

# Anti-spoofing: Check for real face characteristics
ENABLE_ANTI_SPOOFING = True

# Blur detection threshold (higher = more blurry allowed)
# Laplacian variance below this is considered too blurry
MIN_BLUR_THRESHOLD = 100.0

# Brightness thresholds (0-255)
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# Contrast threshold
MIN_CONTRAST = 30

# =============================================================================
# API Configuration
# =============================================================================
API_TITLE = "Face Recognition API"
API_DESCRIPTION = """
A robust facial identification and recognition API powered by DeepFace and FAISS.

## Features
- **Create Record**: Register a new face with associated metadata
- **List Records**: View all registered faces
- **Match Record**: Find matching faces for an input image
- **Delete Record**: Remove a registered face from the system

## Models Used
- **Face Recognition**: ArcFace (99.83% LFW accuracy - BEST)
- **Face Detection**: RetinaFace (most accurate detector)
- **Similarity Search**: FAISS with cosine distance

## Quality Checks
- Face size validation
- Blur detection
- Brightness/contrast checks
- Detection confidence filtering
"""
API_VERSION = "1.1.0"