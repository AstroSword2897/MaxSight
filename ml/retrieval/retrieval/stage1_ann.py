"""Stage 1: Fast Approximate Nearest Neighbor Search Fast ANN search on fused embeddings for candidate retrieval."""

import numpy as np
import faiss
from typing import Tuple, List, Optional
import time


class Stage1ANN:
    """Stage 1 ANN search for fast candidate retrieval. Uses FAISS index for approximate nearest neighbor search. Target latency: <20ms for HNSW, <50ms for IVF-PQ."""
    
    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        index_path: Optional[str] = None
    ):
        self.index = index
        
        if index is None and index_path:
            self.index = faiss.read_index(index_path)
    
    def search(
        self,
        query: np.ndarray,  # [embed_dim] or [B, embed_dim].
        k: int = 200,
        ef_search: Optional[int] = None  # For HNSW.
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-K candidates."""
        if self.index is None:
            raise ValueError("Index not initialized. Provide index or index_path.")
        
        # Ensure query is 2D.
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        
        # Set ef_search for HNSW if provided.
        if ef_search is not None and hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
        
        # Search.
        start_time = time.time()
        distances, indices = self.index.search(query, k)
        elapsed = (time.time() - start_time) * 1000  # Ms.
        
        return distances, indices
    
    def batch_search(
        self,
        queries: np.ndarray,  # [B, embed_dim].
        k: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search for multiple queries."""
        return self.search(queries, k)







