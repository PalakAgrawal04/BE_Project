"""
Query execution service with SQL validation and results merging.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sqlparse

from ..database import MySQLClient, VectorClient
from ..metrics import MetricsCollector
from ..intent_agent import IntentAgentOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query execution results."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    query_type: str = ''
    timestamp: str = ''

class QueryExecutor:
    def __init__(self,
                 mysql_client: MySQLClient,
                 vector_client: VectorClient,
                 intent_agent: IntentAgentOrchestrator,
                 metrics: MetricsCollector):
        """
        Initialize query executor with necessary clients.
        
        Args:
            mysql_client: Client for SQL database operations
            vector_client: Client for vector similarity search
            intent_agent: NLP intent extraction orchestrator
            metrics: Metrics collection service
        """
        self.mysql = mysql_client
        self.vector = vector_client
        self.intent_agent = intent_agent
        self.metrics = metrics

    def validate_sql(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety and read-only operations.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse and validate SQL
            parsed = sqlparse.parse(query)[0]
            
            # Check if it's a SELECT statement
            if not parsed.get_type().upper() == 'SELECT':
                return False, "Only SELECT queries are allowed"
            
            # Check for dangerous keywords
            dangerous_keywords = ['DELETE', 'DROP', 'UPDATE', 'INSERT', 
                               'TRUNCATE', 'ALTER', 'EXEC', 'EXECUTE']
            query_upper = query.upper()
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return False, f"Dangerous keyword found: {keyword}"
            
            return True, None
            
        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

    def execute_query(self, 
                     natural_query: str,
                     include_similar_docs: bool = True) -> QueryResult:
        """
        Execute a natural language query through the full pipeline.
        
        1. Extract intent and entities
        2. Generate and validate SQL
        3. Execute query safely
        4. Find similar documents if requested
        5. Merge and return results
        
        Args:
            natural_query: Natural language query from user
            include_similar_docs: Whether to include vector search results
            
        Returns:
            QueryResult with combined data and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get intent and generated SQL
            with self.metrics.measure_phase('intent_extraction'):
                intent_result = self.intent_agent.run_intent_agent(natural_query)

            # Handle validation results first
            if not intent_result.get('is_valid', True):
                return QueryResult(
                    success=False,
                    data={
                        'validation': intent_result.get('validation', {}),
                        'suggested_rewrite': intent_result.get('suggested_rewrite'),
                        'issues': intent_result.get('issues', [])
                    },
                    error="Query validation failed",
                    query_type='invalid',
                    timestamp=datetime.now().isoformat()
                )
                
            generated_sql = intent_result.get('generated_sql')
            if not generated_sql:
                return QueryResult(
                    success=False,
                    error="Failed to generate SQL from query",
                    query_type='invalid',
                    timestamp=datetime.now().isoformat()
                )
            
            # Validate SQL
            is_valid, error = self.validate_sql(generated_sql)
            if not is_valid:
                return QueryResult(
                    success=False,
                    error=error,
                    query_type='invalid',
                    timestamp=datetime.now().isoformat()
                )
            
            # Execute SQL query
            with self.metrics.measure_phase('sql_execution'):
                sql_results = self.mysql.execute_read_query(generated_sql)
            
            # Get similar documents if requested
            doc_results = []
            if include_similar_docs:
                with self.metrics.measure_phase('vector_search'):
                    query_embedding = intent_result.get('query_embedding')
                    if query_embedding:
                        doc_results = self.vector.search_documents(
                            query_vector=query_embedding,
                            limit=5
                        )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log successful query
            self.mysql.log_query(
                query_text=natural_query,
                query_type='mixed' if include_similar_docs else 'sql',
                execution_time=execution_time,
                status='success'
            )
            
            # Combine results
            return QueryResult(
                success=True,
                data={
                    'sql_results': sql_results,
                    'similar_documents': doc_results,
                    'intent_analysis': intent_result.get('intent_analysis', {}),
                    'generated_sql': generated_sql
                },
                execution_time=execution_time,
                query_type='mixed' if include_similar_docs else 'sql',
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query execution failed: {error_msg}")
            
            # Log failed query
            execution_time = (datetime.now() - start_time).total_seconds()
            self.mysql.log_query(
                query_text=natural_query,
                query_type='error',
                execution_time=execution_time,
                status='error',
                error_message=error_msg
            )
            
            return QueryResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                query_type='error',
                timestamp=datetime.now().isoformat()
            )