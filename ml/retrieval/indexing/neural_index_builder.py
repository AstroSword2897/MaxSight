"""Neural Index Builder for Multi-Vector Retrieval Builds FAISS indices with learned quantization."""

import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List
import torch


class NeuralIndexBuilder:
    """Builds FAISS indices with neural quantization. Supports: - HNSW: Fast approximate search - IVF-PQ: Product quantization for compression - GPU support with CPU fallback."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        dimension: Optional[int] = None,  # Alias for embed_dim.
        index_type: str = "HNSW",  # "HNSW", "IVF-PQ", "Flat"
        metric: str = "L2",  # "L2" or "cosine"
        use_gpu: bool = True
    ):
        # Support both embed_dim and dimension for compatibility.
        self.embed_dim = dimension if dimension is not None else embed_dim
        self.index_type = index_type
        self.metric = metric
        
        # Check GPU availability (handle CPU-only FAISS)
        try:
            self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        except AttributeError:
            # CPU-only FAISS doesn't have get_num_gpus.
            self.use_gpu = False
        
        self.index = None
    
    def build_index(
        self,
        embeddings: np.ndarray,  # [N, embed_dim].
        index_path: Optional[str] = None
    ) -> faiss.Index:
        """Build FAISS index."""
        N, D = embeddings.shape
        assert D == self.embed_dim, f"Expected dimension {self.embed_dim}, got {D}"
        
        # Convert to float32.
        embeddings = embeddings.astype('float32')
        
        # Create index based on type (case-insensitive)
        index_type_upper = self.index_type.upper()
        
        if index_type_upper == "HNSW":
            # HNSW: Hierarchical Navigable Small World.
            M = 32  # Number of connections.
            if self.metric.lower() == "cosine":
                # For cosine similarity, normalize embeddings.
                faiss.normalize_L2(embeddings)
                self.index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(D, M)
            self.index.hnsw.efConstruction = 200
        
        elif index_type_upper == "IVF-PQ" or index_type_upper == "IVFPQ":
            # IVF-PQ: Inverted File with Product Quantization.
            nlist = min(16384, N // 10)  # Number of clusters (adjust for small datasets)
            m = 64  # Number of subquantizers.
            quantizer = faiss.IndexFlatL2(D)
            self.index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)
            
            # Train on subset.
            n_train = min(100000, N)
            if n_train < nlist:
                n_train = nlist  # Need at least nlist samples.
            train_embeddings = embeddings[:n_train]
            self.index.train(train_embeddings)
        
        elif index_type_upper == "FLAT":
            # Flat: Exact search.
            if self.metric.lower() == "cosine":
                faiss.normalize_L2(embeddings)
                self.index = faiss.IndexFlatIP(D)  # Inner product for cosine.
            else:
                self.index = faiss.IndexFlatL2(D)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Supported: HNSW, IVF-PQ, Flat")
        
        # Move to GPU if available.
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                print("GPU not available, using CPU")
                self.use_gpu = False
        
        # Add embeddings.
        self.index.add(embeddings)
        
        # Save if path provided.
        if index_path:
            if self.use_gpu:
                # Move back to CPU for saving.
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
        
        return self.index
    
    def load_index(self, index_path: str) -> faiss.Index:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if available.
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                self.use_gpu = False
        
        return self.index
    
    def search(
        self,
        query: np.ndarray,  # [K, embed_dim] or [embed_dim].
        k: int = 10
    ) -> tuple:
        """Search index. Args: query: Query embeddings k: Number of neighbors Returns: distances, indices."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D.
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        
        # Search.
        distances, indices = self.index.search(query, k)
        
        return distances, indices







