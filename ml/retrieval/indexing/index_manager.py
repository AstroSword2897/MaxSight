"""Index Manager for Multi-Vector Retrieval Manages FAISS index loading, updates, and versioning."""

import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json
from datetime import datetime


class IndexManager:
    """Manages FAISS index lifecycle. Features: - Index loading and saving - Incremental updates - Index versioning - Sharding for large corpora."""
    
    def __init__(
        self,
        index_dir: str = "indices/",
        index_name: str = "fusion_index"
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        
        self.index = None
        self.metadata = {}
    
    def load_index(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ) -> faiss.Index:
        """Load index and metadata. Args: index_path: Path to index file metadata_path: Path to metadata file Returns: FAISS index."""
        if index_path is None:
            index_path = self.index_dir / f"{self.index_name}.faiss"
        
        if metadata_path is None:
            metadata_path = self.index_dir / f"{self.index_name}_metadata.json"
        
        # Load index.
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata.
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        return self.index
    
    def save_index(
        self,
        index: faiss.Index,
        metadata: Optional[Dict] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """Save index and metadata."""
        if index_path is None:
            index_path = self.index_dir / f"{self.index_name}.faiss"
        
        if metadata_path is None:
            metadata_path = self.index_dir / f"{self.index_name}_metadata.json"
        
        # Move to CPU if on GPU.
        if faiss.get_num_gpus() > 0:
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, str(index_path))
            except Exception:
                faiss.write_index(index, str(index_path))
        else:
            faiss.write_index(index, str(index_path))
        
        # Save metadata.
        if metadata is None:
            metadata = {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'num_vectors': index.ntotal,
                'dimension': index.d
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None
    ):
        """Add vectors to existing index. Args: vectors: Vectors to add [N, D] ids: Optional IDs for vectors."""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        
        # Update metadata.
        self.metadata['num_vectors'] = self.index.ntotal
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def remove_vectors(self, ids: List[int]):
        """Remove vectors by ID. Note: FAISS doesn't support direct removal. This would require rebuilding the index."""
        # FAISS doesn't support removal directly. Would need to rebuild index without removed vectors.
        raise NotImplementedError("FAISS doesn't support vector removal. Rebuild index instead.")








