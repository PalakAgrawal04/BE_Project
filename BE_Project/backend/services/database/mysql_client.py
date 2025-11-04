"""
MySQL client with connection pooling and safe query execution.
"""

import logging
from typing import Dict, List, Any, Optional
from mysql.connector.pooling import MySQLConnectionPool
from contextlib import contextmanager

from .connection_pool import get_mysql_pool
from ..metrics import MetricsCollector

logger = logging.getLogger(__name__)

class MySQLClient:
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.pool = get_mysql_pool()
        self.metrics = metrics_collector

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup."""
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def execute_read_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query safely with parameter binding.
        
        Args:
            query: SQL query string with %s placeholders
            params: Dictionary of parameters to bind
            
        Returns:
            List of dictionaries containing query results
            
        Raises:
            ValueError: If query is not SELECT-only
            Exception: For other database errors
        """
        # Basic SQL injection prevention
        query = query.strip()
        if not query.lower().startswith('select'):
            raise ValueError("Only SELECT queries are allowed")

        with self.get_connection() as conn:
            try:
                if self.metrics:
                    with self.metrics.measure_query_time(query):
                        cursor = conn.cursor(dictionary=True)
                        cursor.execute(query, params or {})
                        results = cursor.fetchall()
                else:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(query, params or {})
                    results = cursor.fetchall()
                    
                return results

            except Exception as e:
                logger.error(f"Database error executing query: {str(e)}")
                raise

    def log_query(self, query_text: str, query_type: str, 
                  execution_time: float, status: str,
                  error_message: Optional[str] = None) -> None:
        """
        Log query execution details to the analytics table.
        """
        insert_query = """
        INSERT INTO query_logs 
        (query_text, query_type, execution_time_ms, status, error_message)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(insert_query, 
                             (query_text, query_type, execution_time,
                              status, error_message))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to log query: {str(e)}")
                # Don't raise - logging failure shouldn't affect main flow