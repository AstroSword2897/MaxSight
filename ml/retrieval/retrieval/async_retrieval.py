"""Async/Non-Blocking Retrieval for MaxSight 3.0 Retrieval system that runs asynchronously to avoid blocking inference. Uses threading/queue for non-blocking execution."""

import torch
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class RetrievalRequest:
    """Request for async retrieval."""
    query_embeddings: Dict[str, np.ndarray]
    callback: Optional[Callable] = None
    request_id: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class RetrievalResult:
    """Result from async retrieval."""
    request_id: Optional[str]
    results: Optional[Dict[str, Any]]
    latency_ms: float
    success: bool
    error: Optional[str] = None


class AsyncRetrievalWorker:
    """Worker thread for async retrieval. Processes retrieval requests in background without blocking inference."""
    
    def __init__(
        self,
        stage1_ann,
        stage2_reranker=None,
        knowledge_augment=None,
        max_queue_size: int = 10,
        timeout_ms: float = 100.0
    ):
        self.stage1_ann = stage1_ann
        self.stage2_reranker = stage2_reranker
        self.knowledge_augment = knowledge_augment
        self.max_queue_size = max_queue_size
        self.timeout_ms = timeout_ms
        
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_cache = {}  # Cache recent results.
        self.cache_max_size = 100
        self.cache_ttl = 5.0  # Seconds.
        
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Statistics.
        self.stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'requests_timeout': 0,
            'avg_latency_ms': 0.0
        }
    
    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
    
    def _worker_loop(self):
        """Main worker loop - processes requests from queue."""
        while self.running:
            try:
                # Get request with timeout.
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process request.
                result = self._process_request(request)
                
                # Call callback if provided.
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        print(f"Retrieval callback error: {e}")
                
                # Cache result.
                if request.request_id:
                    self._cache_result(request.request_id, result)
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Retrieval worker error: {e}")
                self.stats['requests_failed'] += 1
    
    def _process_request(self, request: RetrievalRequest) -> RetrievalResult:
        """Process a single retrieval request."""
        start_time = time.time()
        
        try:
            # Stage 1: ANN search.
            if self.stage1_ann is None:
                return RetrievalResult(
                    request_id=request.request_id,
                    results=None,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error="Stage1ANN not initialized"
                )
            
            # Extract query embedding (use global if available)
            query_emb = None
            if 'global' in request.query_embeddings:
                query_emb = request.query_embeddings['global']
            elif len(request.query_embeddings) > 0:
                query_emb = list(request.query_embeddings.values())[0]
            
            if query_emb is None:
                return RetrievalResult(
                    request_id=request.request_id,
                    results=None,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error="No query embedding provided"
                )
            
            # Convert to numpy if needed.
            if isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.detach().cpu().numpy()
            
            # Ensure 2D.
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            
            # Search (with timeout check)
            distances, indices = self.stage1_ann.search(query_emb, k=10)
            
            # Stage 2: Reranking (if available)
            reranked_results = None
            if self.stage2_reranker is not None:
                # Reranking would go here.
                pass
            
            # Knowledge augmentation (if available)
            kg_scores = None
            if self.knowledge_augment is not None:
                # Knowledge augmentation would go here.
                pass
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update stats.
            self.stats['requests_processed'] += 1
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * (self.stats['requests_processed'] - 1) + latency_ms) /
                self.stats['requests_processed']
            )
            
            return RetrievalResult(
                request_id=request.request_id,
                results={
                    'distances': distances,
                    'indices': indices,
                    'reranked': reranked_results,
                    'kg_scores': kg_scores
                },
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            self.stats['requests_failed'] += 1
            return RetrievalResult(
                request_id=request.request_id,
                results=None,
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
    
    def _cache_result(self, request_id: str, result: RetrievalResult):
        """Cache a retrieval result."""
        with self.lock:
            # Remove old entries if cache is full.
            if len(self.result_cache) >= self.cache_max_size:
                # Remove oldest entry.
                oldest_key = min(self.result_cache.keys(), key=lambda k: self.result_cache[k]['timestamp'])
                del self.result_cache[oldest_key]
            
            self.result_cache[request_id] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def get_cached_result(self, request_id: str) -> Optional[RetrievalResult]:
        """Get cached result if available and not expired."""
        with self.lock:
            if request_id in self.result_cache:
                cached = self.result_cache[request_id]
                age = time.time() - cached['timestamp']
                if age < self.cache_ttl:
                    return cached['result']
                else:
                    # Expired, remove.
                    del self.result_cache[request_id]
        return None
    
    def submit_request(
        self,
        query_embeddings: Dict[str, np.ndarray],
        callback: Optional[Callable] = None,
        request_id: Optional[str] = None,
        blocking: bool = False,
        timeout_ms: Optional[float] = None
    ) -> Optional[RetrievalResult]:
        """Submit a retrieval request."""
        # Check cache first.
        if request_id:
            cached = self.get_cached_result(request_id)
            if cached:
                return cached
        
        # Create request.
        request = RetrievalRequest(
            query_embeddings=query_embeddings,
            callback=callback,
            request_id=request_id,
            timestamp=time.time()
        )
        
        # Submit to queue (non-blocking)
        try:
            self.request_queue.put_nowait(request)
        except queue.Full:
            # Queue full - return None (non-blocking)
            self.stats['requests_timeout'] += 1
            return None
        
        # If blocking, wait for result.
        if blocking:
            timeout = timeout_ms / 1000.0 if timeout_ms else self.timeout_ms / 1000.0
            start = time.time()
            while time.time() - start < timeout:
                if request_id:
                    result = self.get_cached_result(request_id)
                    if result:
                        return result
                time.sleep(0.01)  # Small sleep to avoid busy-waiting.
            
            # Timeout.
            self.stats['requests_timeout'] += 1
            return None
        
        return None  # Non-blocking, return immediately.


class AsyncRetrievalSystem:
    """Async retrieval system wrapper. Provides non-blocking retrieval that doesn't delay inference."""
    
    def __init__(
        self,
        stage1_ann,
        stage2_reranker=None,
        knowledge_augment=None,
        enable_async: bool = True,
        max_queue_size: int = 10,
        timeout_ms: float = 100.0
    ):
        self.enable_async = enable_async
        
        if enable_async:
            self.worker = AsyncRetrievalWorker(
                stage1_ann=stage1_ann,
                stage2_reranker=stage2_reranker,
                knowledge_augment=knowledge_augment,
                max_queue_size=max_queue_size,
                timeout_ms=timeout_ms
            )
            self.worker.start()
        else:
            # Synchronous mode (for testing/debugging)
            self.worker = None
            self.stage1_ann = stage1_ann
            self.stage2_reranker = stage2_reranker
            self.knowledge_augment = knowledge_augment
    
    def retrieve(
        self,
        query_embeddings: Dict[str, np.ndarray],
        request_id: Optional[str] = None,
        blocking: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieve similar items (non-blocking by default)."""
        if not self.enable_async:
            # Synchronous mode.
            if self.stage1_ann is None:
                return None
            
            query_emb = query_embeddings.get('global')
            if query_emb is None and len(query_embeddings) > 0:
                query_emb = list(query_embeddings.values())[0]
            
            if query_emb is None:
                return None
            
            if isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.detach().cpu().numpy()
            
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            
            try:
                distances, indices = self.stage1_ann.search(query_emb, k=10)
                return {'distances': distances, 'indices': indices}
            except Exception:
                return None
        
        # Async mode.
        result = self.worker.submit_request(
            query_embeddings=query_embeddings,
            request_id=request_id,
            blocking=blocking
        )
        
        if result and result.success:
            return result.results
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if self.worker:
            return self.worker.stats.copy()
        return {}
    
    def shutdown(self):
        """Shutdown async worker."""
        if self.worker:
            self.worker.stop()







