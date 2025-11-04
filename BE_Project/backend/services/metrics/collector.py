"""
Metrics collection and monitoring service.
"""

import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager
from threading import Lock
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector with sliding window.
        
        Args:
            window_size: Number of samples to keep for each metric
        """
        self.window_size = window_size
        self._metrics: Dict[str, deque] = {}
        self._lock = Lock()
        
    @contextmanager
    def measure_query_time(self, query_type: str):
        """
        Context manager to measure query execution time.
        
        Args:
            query_type: Type of query being measured
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(f"{query_type}_time", duration)
            
    @contextmanager
    def measure_phase(self, phase_name: str):
        """
        Context manager to measure execution phase timing.
        
        Args:
            phase_name: Name of the execution phase
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(f"phase_{phase_name}", duration)
    
    def record_timing(self, metric_name: str, value: float):
        """
        Record a timing metric in the sliding window.
        
        Args:
            metric_name: Name of the metric
            value: Timing value in seconds
        """
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = deque(maxlen=self.window_size)
            self._metrics[metric_name].append(value)
            
    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a metric from its sliding window.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary with min, max, avg, median, p95 statistics
        """
        with self._lock:
            if metric_name not in self._metrics:
                return None
                
            values = list(self._metrics[metric_name])
            if not values:
                return None
                
            return {
                'min': min(values),
                'max': max(values),
                'avg': statistics.mean(values),
                'median': statistics.median(values),
                'p95': statistics.quantiles(values, n=20)[-1]
                if len(values) >= 20 else max(values)
            }
            
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {
            name: stats for name, stats in 
            ((name, self.get_stats(name)) 
             for name in self._metrics.keys())
            if stats is not None
        }