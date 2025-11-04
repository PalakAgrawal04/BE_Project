"""
Connection pooling management for database clients.
"""

import os
from typing import Optional
from mysql.connector.pooling import MySQLConnectionPool
from qdrant_client import QdrantClient

_mysql_pool: Optional[MySQLConnectionPool] = None
_vector_client: Optional[QdrantClient] = None

def get_mysql_pool() -> MySQLConnectionPool:
    """
    Get or create MySQL connection pool.
    """
    global _mysql_pool
    
    if _mysql_pool is None:
        pool_config = {
            'pool_name': 'intelliquery_pool',
            'pool_size': int(os.getenv('MYSQL_POOL_SIZE', 5)),
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'intelliquery'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'database': os.getenv('MYSQL_DATABASE', 'intelliquery'),
            # Read-only user settings
            'allow_local_infile': False,
            'sql_mode': 'NO_ENGINE_SUBSTITUTION,NO_AUTO_CREATE_USER'
        }
        
        _mysql_pool = MySQLConnectionPool(**pool_config)
        
    return _mysql_pool

def get_vector_pool() -> QdrantClient:
    """
    Get or create Qdrant client instance.
    """
    global _vector_client
    
    if _vector_client is None:
        _vector_client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        
    return _vector_client