from typing import List, Any, Dict, Tuple
import time
import hashlib
from functools import lru_cache
from src.app.config import Config

class BatchProcessor:
    """Optimized batch processing utilities for ML inference"""
    
    @staticmethod
    def create_optimal_batches(items: List[Any], max_batch_size: int = None) -> List[List[Any]]:
        """
        Split items into optimal batch sizes for processing
        
        Args:
            items: List of items to batch
            max_batch_size: Maximum size per batch
            
        Returns:
            List of batches
        """
        if not items:
            return []
            
        max_batch_size = max_batch_size or Config.BATCH_PROCESSING['OPTIMAL_BATCH_SIZE']
        
        batches = []
        for i in range(0, len(items), max_batch_size):
            batch = items[i:i + max_batch_size]
            batches.append(batch)
            
        return batches
    
    @staticmethod
    def should_use_batching(num_items: int) -> bool:
        """
        Determine if batching would be beneficial
        
        Args:
            num_items: Number of items to process
            
        Returns:
            True if batching should be used
        """
        return (Config.BATCH_PROCESSING['ENABLE_AUTO_BATCHING'] and 
                num_items >= Config.BATCH_PROCESSING['MIN_BATCH_SIZE'])

class PerformanceMonitor:
    """Simple performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return TimingContext(operation_name, self.metrics)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get recorded performance metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all recorded metrics"""
        self.metrics.clear()

class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, metrics_dict: Dict):
        self.operation_name = operation_name
        self.metrics_dict = metrics_dict
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_dict[self.operation_name] = duration

class TextCache:
    """Simple LRU cache for text predictions"""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.PERFORMANCE['MAX_CACHE_SIZE']
        self._cache = {}
        self._access_times = {}
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate a hash key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Any:
        """Get cached prediction for text"""
        key = self._generate_cache_key(text)
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, text: str, prediction: Any):
        """Cache prediction for text"""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._generate_cache_key(text)
        self._cache[key] = prediction
        self._access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Remove least recently used item"""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), 
                        key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Clear all cached items"""
        self._cache.clear()
        self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

class RequestOptimizer:
    """Utilities for optimizing API requests"""
    
    @staticmethod
    def deduplicate_texts(texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        Remove duplicate texts and return mapping for reconstruction
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (unique_texts, index_mapping)
        """
        seen = {}
        unique_texts = []
        index_mapping = []
        
        for i, text in enumerate(texts):
            if text not in seen:
                seen[text] = len(unique_texts)
                unique_texts.append(text)
            index_mapping.append(seen[text])
        
        return unique_texts, index_mapping
    
    @staticmethod
    def reconstruct_results(unique_results: List[Any], 
                          index_mapping: List[int]) -> List[Any]:
        """
        Reconstruct full results from deduplicated processing
        
        Args:
            unique_results: Results for unique texts
            index_mapping: Mapping from original to unique indices
            
        Returns:
            Full results list matching original input order
        """
        return [unique_results[idx] for idx in index_mapping]