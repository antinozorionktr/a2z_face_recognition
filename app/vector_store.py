"""
FAISS Vector Store Service

This module manages the FAISS index for efficient similarity search
of face embeddings. It provides:
- Index creation and management
- Adding/removing vectors
- Similarity search with top-K retrieval
- Persistence (save/load index)

Note: Metadata is now stored in PostgreSQL. This module only handles
vector operations.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import threading

from app.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIM,
    TOP_K_MATCHES,
    RECOGNITION_THRESHOLD
)

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for face embeddings.
    
    Uses IndexFlatIP (Inner Product) with normalized vectors
    for cosine similarity search.
    
    Thread-safe implementation with read-write locks.
    
    Note: This class only manages FAISS index. Metadata is stored in PostgreSQL.
    """
    
    def __init__(self):
        """Initialize the vector store."""
        self._lock = threading.RLock()
        self._index: Optional[faiss.IndexFlatIP] = None
        self._next_index = 0
        self._deleted_indices: set = set()
        self._index_mapping: dict = {}  # Maps faiss position to embedding_index
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new one."""
        with self._lock:
            if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
                try:
                    self._load_index()
                    logger.info(f"Loaded existing FAISS index with {self.count} vectors")
                except Exception as e:
                    logger.warning(f"Failed to load index: {e}. Creating new one.")
                    self._create_new_index()
            else:
                self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._next_index = 0
        self._deleted_indices = set()
        self._index_mapping = {}
        logger.info("Created new FAISS index")
    
    def _load_index(self):
        """Load index and metadata from disk."""
        self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, "r") as f:
            data = json.load(f)
        
        self._next_index = data.get("next_index", 0)
        self._deleted_indices = set(data.get("deleted_indices", []))
        self._index_mapping = {int(k): v for k, v in data.get("index_mapping", {}).items()}
    
    def _save_index(self):
        """Persist index and metadata to disk."""
        try:
            faiss.write_index(self._index, str(FAISS_INDEX_PATH))
            
            data = {
                "next_index": self._next_index,
                "deleted_indices": list(self._deleted_indices),
                "index_mapping": {str(k): v for k, v in self._index_mapping.items()}
            }
            
            with open(METADATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    @property
    def count(self) -> int:
        """Get the number of active vectors."""
        with self._lock:
            return self._index.ntotal - len(self._deleted_indices) if self._index else 0
    
    @property
    def next_embedding_index(self) -> int:
        """Get the next embedding index to use."""
        with self._lock:
            return self._next_index
    
    def add(self, embedding: np.ndarray) -> int:
        """
        Add a new face embedding to the index.
        
        Args:
            embedding: 512-dim normalized face embedding
            
        Returns:
            embedding_index: The index assigned to this embedding
        """
        with self._lock:
            # Ensure embedding is the right shape and type
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            # Add to FAISS index
            self._index.add(embedding)
            embedding_index = self._next_index
            
            # Track mapping
            faiss_position = self._index.ntotal - 1
            self._index_mapping[faiss_position] = embedding_index
            
            self._next_index += 1
            
            # Persist changes
            self._save_index()
            
            logger.info(f"Added embedding at index {embedding_index}")
            
            return embedding_index
    
    def delete(self, embedding_index: int) -> bool:
        """
        Mark an embedding as deleted.
        
        Note: FAISS doesn't support true deletion, so we mark the index
        as deleted and filter results during search.
        
        Args:
            embedding_index: Index of the embedding to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            # Find the faiss position for this embedding_index
            faiss_position = None
            for pos, idx in self._index_mapping.items():
                if idx == embedding_index:
                    faiss_position = pos
                    break
            
            if faiss_position is None:
                return False
            
            # Mark as deleted
            self._deleted_indices.add(faiss_position)
            
            # Persist changes
            self._save_index()
            
            logger.info(f"Deleted embedding at index {embedding_index}")
            return True
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = TOP_K_MATCHES,
        threshold: float = RECOGNITION_THRESHOLD
    ) -> Tuple[bool, List[Tuple[int, float, float]]]:
        """
        Search for similar faces.
        
        Args:
            query_embedding: Normalized 512-dim query vector
            top_k: Number of top matches to return
            threshold: Recognition threshold (cosine distance)
            
        Returns:
            Tuple of:
            - recognized: bool (whether best match is above threshold)
            - matches: List of (embedding_index, confidence, distance)
        """
        with self._lock:
            if self.count == 0:
                return False, []
            
            # Prepare query
            query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Search for more than top_k to account for deleted entries
            search_k = min(top_k + len(self._deleted_indices) + 5, self._index.ntotal)
            
            if search_k == 0:
                return False, []
            
            # FAISS search (returns inner product scores, higher = more similar)
            scores, indices = self._index.search(query, search_k)
            
            # Convert results, filtering deleted entries
            matches = []
            for score, faiss_pos in zip(scores[0], indices[0]):
                if faiss_pos < 0 or faiss_pos in self._deleted_indices:
                    continue
                
                # Get the embedding_index for this faiss position
                embedding_index = self._index_mapping.get(faiss_pos)
                if embedding_index is None:
                    continue
                
                # Convert inner product to cosine distance
                # For normalized vectors: cosine_distance = 1 - inner_product
                distance = 1.0 - float(score)
                confidence = float(score)  # Inner product is the confidence
                
                matches.append((
                    embedding_index,
                    round(max(0, min(1, confidence)), 4),
                    round(max(0, distance), 4)
                ))
                
                if len(matches) >= top_k:
                    break
            
            if not matches:
                return False, []
            
            # Sort by confidence (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            best_distance = matches[0][2]
            recognized = best_distance <= threshold
            
            return recognized, matches
    
    def rebuild_index(self) -> int:
        """
        Rebuild the FAISS index to reclaim space from deleted entries.
        
        This is an expensive operation and should be done during maintenance.
        
        Returns:
            Number of active vectors after rebuild
        """
        with self._lock:
            if not self._deleted_indices:
                logger.info("No deleted entries, skipping rebuild")
                return self.count
            
            logger.info(f"Rebuilding index, removing {len(self._deleted_indices)} deleted entries")
            
            # Collect all active embeddings
            active_embeddings = []
            new_index_mapping = {}
            
            for faiss_pos in range(self._index.ntotal):
                if faiss_pos in self._deleted_indices:
                    continue
                
                embedding_index = self._index_mapping.get(faiss_pos)
                if embedding_index is None:
                    continue
                
                # Reconstruct embedding from index
                embedding = self._index.reconstruct(faiss_pos)
                new_faiss_pos = len(active_embeddings)
                
                active_embeddings.append(embedding)
                new_index_mapping[new_faiss_pos] = embedding_index
            
            # Create new index
            new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            if active_embeddings:
                embeddings_array = np.array(active_embeddings, dtype=np.float32)
                new_index.add(embeddings_array)
            
            # Replace old index
            self._index = new_index
            self._index_mapping = new_index_mapping
            self._deleted_indices = set()
            
            self._save_index()
            logger.info(f"Index rebuilt with {self.count} active vectors")
            
            return self.count


# Singleton instance
vector_store = FAISSVectorStore()