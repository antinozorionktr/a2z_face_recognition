# Face Recognition API

A production-ready facial identification and recognition API powered by **DeepFace** and **FAISS**.

## Architecture

```
Input Image
     ↓
Image Preprocessing (resize, RGB conversion)
     ↓
Face Detection (RetinaFace)
     ↓
Face Extraction & Alignment
     ↓
Face Embedding Generation (ArcFace - 512-dim)
     ↓
Query Embedding Vector
     ↓
FAISS Similarity Search (cosine distance)
     ↓
Top-K Nearest Matches
     ↓
Threshold Check
     ↓
Recognized / Unknown
     ↓
Final Output (Name, Confidence Score)
```

## Models Used

| Component | Model | Details |
|-----------|-------|---------|
| Face Recognition | **ArcFace** | 99.83% accuracy on LFW benchmark, 512-dim embeddings |
| Face Detection | **RetinaFace** | State-of-the-art detector with landmark detection |
| Similarity Search | **FAISS IndexFlatIP** | Exact inner product search (cosine similarity) |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API docs at: `http://localhost:8000/docs`

## API Endpoints

### 1. Create Record (POST /records)

Register a new face in the system.

**Request:**
- `image`: Face image file (JPG, PNG, WebP, BMP)
- `name`: Person's name/identifier
- `metadata`: Optional JSON string with additional info

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/records" \
  -F "image=@john_doe.jpg" \
  -F "name=John Doe" \
  -F 'metadata={"department": "Engineering", "employee_id": "EMP001"}'
```

**Response:**
```json
{
  "success": true,
  "message": "Face record created successfully for 'John Doe'",
  "record": {
    "id": "abc123-def456-...",
    "name": "John Doe",
    "metadata": {"department": "Engineering", "employee_id": "EMP001"},
    "created_at": "2024-01-15T10:30:00",
    "embedding_index": 0
  }
}
```

### 2. List Records (GET /records)

Retrieve all registered face records.

**Query Parameters:**
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records to return (default: 100)

**Example:**
```bash
curl "http://localhost:8000/records?skip=0&limit=10"
```

**Response:**
```json
{
  "total_count": 25,
  "records": [
    {
      "id": "abc123-...",
      "name": "John Doe",
      "metadata": {...},
      "created_at": "2024-01-15T10:30:00",
      "embedding_index": 0
    }
  ]
}
```

### 3. Match Record (POST /records/match)

Find matching faces for an input image.

**Request:**
- `image`: Query face image
- `top_k`: Number of top matches (default: 5)
- `threshold`: Recognition threshold (default: 0.45, lower = stricter)

**Example:**
```bash
curl -X POST "http://localhost:8000/records/match" \
  -F "image=@unknown_face.jpg" \
  -F "top_k=5" \
  -F "threshold=0.45"
```

**Response:**
```json
{
  "recognized": true,
  "best_match": {
    "id": "abc123-...",
    "name": "John Doe",
    "confidence": 0.92,
    "distance": 0.08,
    "metadata": {"department": "Engineering"}
  },
  "top_matches": [
    {"id": "abc123-...", "name": "John Doe", "confidence": 0.92, ...},
    {"id": "xyz789-...", "name": "Jane Smith", "confidence": 0.65, ...}
  ],
  "face_detected": true,
  "processing_time_ms": 245.5
}
```

### 4. Delete Record (DELETE /records/{record_id})

Remove a face from the database.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/records/abc123-def456-..."
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully deleted record for 'John Doe'",
  "deleted_id": "abc123-def456-..."
}
```

## Recognition Threshold Guide

The threshold controls how strict the matching is (cosine distance):

| Threshold | Strictness | Use Case |
|-----------|------------|----------|
| 0.35 | Very Strict | High-security access control |
| 0.40 | Strict | Standard access control |
| **0.45** | Normal (default) | General identification |
| 0.50 | Relaxed | Suggestion/recommendation systems |

## Project Structure

```
face_recognition_app/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application & endpoints
│   ├── config.py         # Configuration settings
│   ├── schemas.py        # Pydantic models
│   ├── face_service.py   # DeepFace integration
│   └── vector_store.py   # FAISS vector store
├── data/
│   └── embeddings/       # FAISS index & metadata storage
├── requirements.txt
└── README.md
```

## Configuration

Key settings in `app/config.py`:

```python
# Face Recognition Model
FACE_RECOGNITION_MODEL = "ArcFace"  # Best accuracy

# Face Detection Backend
FACE_DETECTOR_BACKEND = "retinaface"  # Most accurate

# Recognition Threshold
RECOGNITION_THRESHOLD = 0.45  # Cosine distance threshold

# Top-K matches to return
TOP_K_MATCHES = 5
```

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Create a record
with open("john_doe.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/records",
        files={"image": f},
        data={
            "name": "John Doe",
            "metadata": '{"department": "Engineering"}'
        }
    )
    record = response.json()
    print(f"Created: {record}")

# 2. List all records
response = requests.get(f"{BASE_URL}/records")
records = response.json()
print(f"Total records: {records['total_count']}")

# 3. Match a face
with open("unknown.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/records/match",
        files={"image": f},
        data={"top_k": 5, "threshold": 0.45}
    )
    result = response.json()
    
    if result["recognized"]:
        print(f"Recognized: {result['best_match']['name']}")
        print(f"Confidence: {result['best_match']['confidence']:.2%}")
    else:
        print("Unknown person")

# 4. Delete a record
record_id = "abc123-..."
response = requests.delete(f"{BASE_URL}/records/{record_id}")
print(response.json())
```

## Performance Notes

- **First request** may be slow (~10-30s) as models are loaded
- **Subsequent requests** are fast (~100-500ms depending on image size)
- **FAISS search** is extremely fast (sub-millisecond for <100K records)
- Use `POST /maintenance/rebuild-index` periodically to optimize storage

## Error Handling

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Invalid input (bad file format, empty file) |
| 404 | Record not found |
| 422 | No face detected in image |
| 500 | Internal server error |

## License

MIT License
