"""
Embedding retrieval module for Intent Agent.

This module handles embedding generation using Sentence-BERT and
similarity retrieval using FAISS for finding relevant context.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """
    Handles embedding generation and similarity retrieval using Sentence-BERT and FAISS.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        """
        Initialize the embedding retriever.
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
            index_path (str, optional): Path to the FAISS index file
        """
        self.model_name = model_name
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.index = None
        self.embeddings_data = []
        
        self._load_model()
        self._load_or_create_index()
    
    def _load_model(self) -> None:
        """Load the Sentence-BERT model."""
        try:
            self.logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        if self.index_path and os.path.exists(self.index_path):
            try:
                self.logger.info(f"Loading FAISS index from: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                # Load associated data
                data_path = self.index_path.replace('.faiss', '_data.json')
                if os.path.exists(data_path):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        self.embeddings_data = json.load(f)
                
                self.logger.info(f"Index loaded with {self.index.ntotal} vectors")
            except Exception as e:
                self.logger.warning(f"Failed to load index: {str(e)}. Creating new index.")
                self._create_new_index()
        else:
            self.logger.info("No existing index found. Creating new index.")
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        try:
            # Get embedding dimension from model
            sample_embedding = self.model.encode(["sample text"])
            dimension = sample_embedding.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(dimension)
            self.embeddings_data = []
            
            self.logger.info(f"Created new FAISS index with dimension: {dimension}")
        except Exception as e:
            self.logger.error(f"Failed to create index: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        try:
            # Generate embedding
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def add_to_index(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add texts to the FAISS index.
        
        Args:
            texts (List[str]): List of texts to add
            metadata (List[Dict], optional): Associated metadata for each text
        """
        if not texts:
            return
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            for i, text in enumerate(texts):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                self.embeddings_data.append({
                    "text": text,
                    "index": self.index.ntotal - len(texts) + i,
                    "metadata": meta
                })
            
            self.logger.info(f"Added {len(texts)} texts to index. Total: {self.index.ntotal}")
            
        except Exception as e:
            self.logger.error(f"Failed to add texts to index: {str(e)}")
            raise
    
    def retrieve_similar_queries(self, embedding: List[float], top_k: int = 3) -> List[str]:
        """
        Retrieve top-k similar queries from the FAISS index.
        
        Args:
            embedding (List[float]): Query embedding
            top_k (int): Number of similar queries to retrieve
            
        Returns:
            List[str]: List of similar query texts
        """
        if not embedding or len(embedding) == 0:
            return []
        
        if self.index.ntotal == 0:
            self.logger.warning("Index is empty. Cannot retrieve similar queries.")
            return []
        
        try:
            # Convert to numpy array and reshape
            query_vector = np.array([embedding], dtype=np.float32)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Retrieve similar texts
            similar_queries = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.embeddings_data):
                    similar_queries.append(self.embeddings_data[idx]["text"])
            
            self.logger.info(f"Retrieved {len(similar_queries)} similar queries")
            return similar_queries
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar queries: {str(e)}")
            return []
    
    def retrieve_similar_with_scores(self, embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve similar queries with similarity scores.
        
        Args:
            embedding (List[float]): Query embedding
            top_k (int): Number of similar queries to retrieve
            
        Returns:
            List[Dict]: List of dictionaries with 'text', 'score', and 'metadata'
        """
        if not embedding or len(embedding) == 0:
            return []
        
        if self.index.ntotal == 0:
            self.logger.warning("Index is empty. Cannot retrieve similar queries.")
            return []
        
        try:
            # Convert to numpy array and reshape
            query_vector = np.array([embedding], dtype=np.float32)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Retrieve similar texts with scores
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.embeddings_data):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    results.append({
                        "text": self.embeddings_data[idx]["text"],
                        "score": similarity_score,
                        "distance": distance,
                        "metadata": self.embeddings_data[idx].get("metadata", {})
                    })
            
            self.logger.info(f"Retrieved {len(results)} similar queries with scores")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar queries with scores: {str(e)}")
            return []
    
    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save the FAISS index and associated data to disk.
        
        Args:
            index_path (str, optional): Path to save the index
        """
        save_path = index_path or self.index_path
        if not save_path:
            self.logger.warning("No index path provided. Cannot save index.")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, save_path)
            
            # Save associated data
            data_path = save_path.replace('.faiss', '_data.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Index saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def initialize_with_sample_data(self) -> None:
        """
        Initialize the index with sample query data for testing.
        """
        sample_queries = [
            "Show me sales data for last month",
            "What are the top products by revenue?",
            "Compare sales between Q1 and Q2",
            "Find customers with pending orders",
            "Get support ticket statistics",
            "Analyze customer feedback trends",
            "Update product pricing",
            "Generate monthly sales report",
            "Predict next quarter revenue",
            "Summarize employee performance"
        ]
        
        sample_metadata = [
            {"intent": "read", "workspace": "sales", "complexity": "simple"},
            {"intent": "read", "workspace": "sales", "complexity": "simple"},
            {"intent": "compare", "workspace": "sales", "complexity": "simple"},
            {"intent": "read", "workspace": "sales", "complexity": "simple"},
            {"intent": "read", "workspace": "support", "complexity": "simple"},
            {"intent": "analyze", "workspace": "support", "complexity": "complex"},
            {"intent": "update", "workspace": "sales", "complexity": "simple"},
            {"intent": "summarize", "workspace": "sales", "complexity": "simple"},
            {"intent": "predict", "workspace": "sales", "complexity": "complex"},
            {"intent": "summarize", "workspace": "hr", "complexity": "simple"}
        ]
        
        self.add_to_index(sample_queries, sample_metadata)
        self.logger.info("Initialized index with sample data")


# Global instance for convenience
_retriever_instance = None


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for the given text using the global retriever instance.
    
    Args:
        text (str): Input text
        
    Returns:
        List[float]: Embedding vector
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EmbeddingRetriever()
    return _retriever_instance.get_embedding(text)


def retrieve_similar_queries(embedding: List[float], top_k: int = 3) -> List[str]:
    """
    Retrieve similar queries using the global retriever instance.
    
    Args:
        embedding (List[float]): Query embedding
        top_k (int): Number of similar queries to retrieve
        
    Returns:
        List[str]: List of similar query texts
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EmbeddingRetriever()
    return _retriever_instance.retrieve_similar_queries(embedding, top_k)


if __name__ == "__main__":
    # Test the embedding retriever
    retriever = EmbeddingRetriever()
    
    # Initialize with sample data
    retriever.initialize_with_sample_data()
    
    # Test embedding generation
    test_query = "Show me sales in Mumbai for last month"
    embedding = retriever.get_embedding(test_query)
    print(f"Generated embedding with dimension: {len(embedding)}")
    
    # Test similarity retrieval
    similar_queries = retriever.retrieve_similar_queries(embedding, top_k=3)
    print(f"Similar queries to '{test_query}':")
    for i, query in enumerate(similar_queries, 1):
        print(f"{i}. {query}")
    
    # Test with scores
    similar_with_scores = retriever.retrieve_similar_with_scores(embedding, top_k=3)
    print(f"\nSimilar queries with scores:")
    for result in similar_with_scores:
        print(f"Score: {result['score']:.3f} - {result['text']}")
