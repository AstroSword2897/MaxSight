"""
Neural Index Builder for Multi-Vector Retrieval

Builds FAISS indices with learned quantization.
"""

import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List
import torch


class NeuralIndexBuilder:
    """
    Builds FAISS indices with neural quantization.
    
    Supports:
    - HNSW: Fast approximate search
    - IVF-PQ: Product quantization for compression
    - GPU support with CPU fallback
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        index_type: str = "HNSW",  # "HNSW", "IVF-PQ", "Flat"
        use_gpu: bool = True
    ):
        self.embed_dim = embed_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        self.index = None
    
    def build_index(
        self,
        embeddings: np.ndarray,  # [N, embed_dim]
        index_path: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index.
        
        Args:
            embeddings: Embeddings to index [N, embed_dim]
            index_path: Optional path to save index
        
        Returns:
            FAISS index
        """
        N, D = embeddings.shape
        assert D == self.embed_dim, f"Expected dimension {self.embed_dim}, got {D}"
        
        # Convert to float32
        embeddings = embeddings.astype('float32')
        
        # Create index based on type
        if self.index_type == "HNSW":
            # HNSW: Hierarchical Navigable Small World
            M = 32  # Number of connections
            self.index = faiss.IndexHNSWFlat(D, M)
            self.index.hnsw.efConstruction = 200
        
        elif self.index_type == "IVF-PQ":
            # IVF-PQ: Inverted File with Product Quantization
            nlist = 16384  # Number of clusters
            m = 64  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(D)
            self.index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)
            
            # Train on subset
            n_train = min(100000, N)
            train_embeddings = embeddings[:n_train]
            self.index.train(train_embeddings)
        
        elif self.index_type == "Flat":
            # Flat: Exact search
            self.index = faiss.IndexFlatL2(D)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                print("GPU not available, using CPU")
                self.use_gpu = False
        
        # Add embeddings
        self.index.add(embeddings)
        
        # Save if path provided
        if index_path:
            if self.use_gpu:
                # Move back to CPU for saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
        
        return self.index
    
    def load_index(self, index_path: str) -> faiss.Index:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                self.use_gpu = False
        
        return self.index
    
    def search(
        self,
        query: np.ndarray,  # [K, embed_dim] or [embed_dim]
        k: int = 10
    ) -> tuple:
        """
        Search index.
        
        Args:
            query: Query embeddings
            k: Number of neighbors
        
        Returns:
            distances, indices
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        return distances, indices


