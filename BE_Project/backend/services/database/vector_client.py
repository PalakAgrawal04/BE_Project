"""
Vector database client for document similarity search using Qdrant.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

from ..metrics import MetricsCollector

logger = logging.getLogger(__name__)

class VectorClient:
    def __init__(self, 
                 collection_name: str = "documents",
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize Qdrant client for vector similarity search.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            metrics_collector: Optional metrics collector for timing
        """
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.metrics = metrics_collector

    def search_documents(self, 
                        query_vector: np.ndarray,
                        limit: int = 5,
                        score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query vector.
        
        Args:
            query_vector: Embedded query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of matching documents with metadata and scores
        """
        try:
            if self.metrics:
                with self.metrics.measure_query_time("vector_search"):
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector.tolist(),
                        limit=limit,
                        score_threshold=score_threshold
                    )
            else:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=limit,
                    score_threshold=score_threshold
                )

            # Format results
            results = []
            for hit in search_result:
                doc = {
                    'document_id': hit.id,
                    'score': hit.score,
                    'metadata': hit.payload
                }
                results.append(doc)
                
            return results

        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            raise

    def add_document(self, 
                    doc_id: str,
                    vector: np.ndarray,
                    metadata: Dict[str, Any]) -> None:
        """
        Add a new document vector to the collection.
        
        Args:
            doc_id: Unique document identifier
            vector: Document embedding vector
            metadata: Document metadata (title, content, etc.)
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=vector.tolist(),
                        payload=metadata
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise