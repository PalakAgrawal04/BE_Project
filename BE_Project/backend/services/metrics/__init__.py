"""
Metrics package for IntelliQuery backend.

This exposes the MetricsCollector class so other modules can import
`from backend.services.metrics import MetricsCollector`.
"""

from .collector import MetricsCollector

__all__ = ["MetricsCollector"]
