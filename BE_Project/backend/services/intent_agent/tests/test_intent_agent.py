"""
Test suite for Intent Agent module.

This module contains comprehensive tests for all components of the Intent Agent,
including unit tests, integration tests, and mock implementations.
"""

import unittest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from preprocessor import TextPreprocessor
from entity_extractor import EntityExtractor
from classifier import IntentClassifier
from schema_validator import SchemaValidator
from agent_orchestrator import IntentAgentOrchestrator


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "Show me sales in Mumbai for last month"
        result = self.preprocessor.preprocess_text(text)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, text.lower())  # Should be lowercased
    
    def test_preprocess_text_cleaning(self):
        """Test text cleaning functionality."""
        text = "  Multiple    spaces   and   special   chars!!!  "
        result = self.preprocessor.preprocess_text(text)
        
        self.assertIsInstance(result, str)
        self.assertNotIn("  ", result)  # No double spaces
        self.assertNotIn("!!!", result)  # Special chars removed
    
    def test_preprocess_text_empty(self):
        """Test preprocessing with empty input."""
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_text("")
        
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_text(None)
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "show me sales data for mumbai last month"
        keywords = self.preprocessor.extract_keywords(text)
        
        self.assertIsInstance(keywords, list)
        self.assertIn("sales", keywords)
        self.assertIn("mumbai", keywords)
        self.assertNotIn("me", keywords)  # Stop word should be filtered
    
    def test_detect_query_type(self):
        """Test query type detection."""
        # Question
        question = "What is the sales data?"
        self.assertEqual(self.preprocessor.detect_query_type(question), "question")
        
        # Command
        command = "Show me sales data"
        self.assertEqual(self.preprocessor.detect_query_type(command), "command")
        
        # Statement
        statement = "Sales data is available"
        self.assertEqual(self.preprocessor.detect_query_type(statement), "statement")


class TestEntityExtractor(unittest.TestCase):
    """Test cases for EntityExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock spaCy model to avoid dependency issues
        with patch('entity_extractor.spacy.load') as mock_spacy:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "Mumbai"
            mock_ent.label_ = "GPE"
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            self.extractor = EntityExtractor()
    
    def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        text = "Show me sales in Mumbai for last month"
        entities = self.extractor.extract_entities(text)
        
        self.assertIsInstance(entities, dict)
        self.assertIn("dates", entities)
        self.assertIn("locations", entities)
        self.assertIn("quantities", entities)
    
    def test_extract_entities_empty(self):
        """Test entity extraction with empty input."""
        result = self.extractor.extract_entities("")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["dates"], [])
        self.assertEqual(result["locations"], [])
    
    def test_extract_temporal_expressions(self):
        """Test temporal expression extraction."""
        text = "Show me data for last month and next quarter"
        temporal = self.extractor.extract_temporal_expressions(text)
        
        self.assertIsInstance(temporal, list)
        # Should find temporal expressions
    
    def test_extract_numeric_expressions(self):
        """Test numeric expression extraction."""
        text = "Show me top 10 products with 95% confidence"
        numeric = self.extractor.extract_numeric_expressions(text)
        
        self.assertIsInstance(numeric, list)
        # Should find numeric expressions


class TestIntentClassifier(unittest.TestCase):
    """Test cases for IntentClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()
    
    def test_predict_intent_type_read(self):
        """Test read intent prediction."""
        text = "Show me sales data"
        intent = self.classifier.predict_intent_type(text)
        
        self.assertEqual(intent, "read")
    
    def test_predict_intent_type_compare(self):
        """Test compare intent prediction."""
        text = "Compare sales between 2023 and 2024"
        intent = self.classifier.predict_intent_type(text)
        
        self.assertEqual(intent, "compare")
    
    def test_predict_intent_type_update(self):
        """Test update intent prediction."""
        text = "Update customer status to active"
        intent = self.classifier.predict_intent_type(text)
        
        self.assertEqual(intent, "update")
    
    def test_predict_workspace(self):
        """Test workspace prediction."""
        text = "Show me sales data for last month"
        workspaces = self.classifier.predict_workspace(text)
        
        self.assertIsInstance(workspaces, list)
        if workspaces:
            self.assertIn("id", workspaces[0])
            self.assertIn("score", workspaces[0])
    
    def test_classify_query_complexity(self):
        """Test query complexity classification."""
        simple_query = "Show me sales data"
        complex_query = "Analyze the correlation between customer satisfaction and sales trends"
        
        self.assertEqual(self.classifier.classify_query_complexity(simple_query), "simple")
        self.assertEqual(self.classifier.classify_query_complexity(complex_query), "complex")
    
    def test_detect_time_sensitivity(self):
        """Test time sensitivity detection."""
        immediate_query = "Show me current sales data now"
        historical_query = "Show me sales data for last year"
        
        self.assertEqual(self.classifier.detect_time_sensitivity(immediate_query), "immediate")
        self.assertEqual(self.classifier.detect_time_sensitivity(historical_query), "historical")


class TestSchemaValidator(unittest.TestCase):
    """Test cases for SchemaValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SchemaValidator()
    
    def test_validate_intent_schema_valid(self):
        """Test validation with valid intent data."""
        valid_intent = {
            "intent_type": "read",
            "workspaces": ["sales"],
            "entities": {
                "dates": ["last month"],
                "locations": ["Mumbai"],
                "quantities": [],
                "products": [],
                "organizations": [],
                "people": [],
                "custom": {}
            },
            "confidence": 0.95,
            "rationale": "Clear request for sales data"
        }
        
        result = self.validator.validate_intent_schema(valid_intent)
        self.assertTrue(result)
    
    def test_validate_intent_schema_invalid(self):
        """Test validation with invalid intent data."""
        invalid_intent = {
            "intent_type": "invalid_type",
            "workspaces": [],
            "confidence": 1.5
        }
        
        result = self.validator.validate_intent_schema(invalid_intent)
        self.assertFalse(result)
    
    def test_validate_and_repair(self):
        """Test validation and repair functionality."""
        invalid_intent = {
            "intent_type": "invalid_type",
            "workspaces": [],
            "confidence": 1.5
        }
        
        repaired = self.validator.validate_and_repair(invalid_intent)
        
        self.assertIsInstance(repaired, dict)
        self.assertEqual(repaired["intent_type"], "read")  # Should be repaired
        self.assertEqual(repaired["confidence"], 1.0)  # Should be capped at 1.0
    
    def test_validate_entities(self):
        """Test entity validation."""
        entities = {
            "dates": ["last month", "last month", ""],  # Duplicate and empty
            "locations": ["Mumbai"],
            "quantities": [],
            "products": [],
            "organizations": [],
            "people": [],
            "custom": {}
        }
        
        validated = self.validator.validate_entities(entities)
        
        self.assertIsInstance(validated, dict)
        self.assertEqual(len(validated["dates"]), 1)  # Duplicates should be removed
    
    def test_is_valid_intent_type(self):
        """Test intent type validation."""
        self.assertTrue(self.validator.is_valid_intent_type("read"))
        self.assertTrue(self.validator.is_valid_intent_type("compare"))
        self.assertFalse(self.validator.is_valid_intent_type("invalid"))


class TestIntentAgentOrchestrator(unittest.TestCase):
    """Test cases for IntentAgentOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock all external dependencies
        with patch('agent_orchestrator.EmbeddingRetriever') as mock_embedding, \
             patch('agent_orchestrator.EntityExtractor') as mock_entity, \
             patch('agent_orchestrator.LLMMapper') as mock_llm:
            
            # Set up mocks
            mock_embedding_instance = Mock()
            mock_embedding_instance.get_embedding.return_value = [0.1] * 384
            mock_embedding_instance.retrieve_similar_queries.return_value = ["similar query"]
            mock_embedding_instance.index.ntotal = 10
            mock_embedding.return_value = mock_embedding_instance
            
            mock_entity_instance = Mock()
            mock_entity_instance.extract_entities.return_value = {
                'dates': [], 'locations': [], 'quantities': [], 'products': [],
                'organizations': [], 'people': [], 'custom': {}
            }
            mock_entity.return_value = mock_entity_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.map_intent_with_llm.return_value = {
                'intent_type': 'read',
                'workspaces': ['sales'],
                'entities': {'dates': [], 'locations': [], 'quantities': [], 'products': [], 'organizations': [], 'people': [], 'custom': {}},
                'confidence': 0.95,
                'rationale': 'Test rationale'
            }
            mock_llm.return_value = mock_llm_instance
            
            self.orchestrator = IntentAgentOrchestrator()
    
    def test_run_intent_agent_success(self):
        """Test successful intent agent execution."""
        query = "Show me sales data for last month"
        result = self.orchestrator.run_intent_agent(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn("intent_analysis", result)
        self.assertIn("metadata", result)
        
        intent_analysis = result["intent_analysis"]
        self.assertIn("intent_type", intent_analysis)
        self.assertIn("workspaces", intent_analysis)
        self.assertIn("entities", intent_analysis)
        self.assertIn("confidence", intent_analysis)
    
    def test_run_intent_agent_empty_input(self):
        """Test intent agent with empty input."""
        result = self.orchestrator.run_intent_agent("")
        
        self.assertIsInstance(result, dict)
        self.assertIn("metadata", result)
        self.assertTrue(result["metadata"].get("error", False))
    
    def test_run_intent_agent_invalid_input(self):
        """Test intent agent with invalid input."""
        result = self.orchestrator.run_intent_agent(None)
        
        self.assertIsInstance(result, dict)
        self.assertIn("metadata", result)
        self.assertTrue(result["metadata"].get("error", False))
    
    def test_get_pipeline_status(self):
        """Test pipeline status retrieval."""
        status = self.orchestrator.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("preprocessing", status)
        self.assertIn("embedding_retrieval", status)
        self.assertIn("entity_extraction", status)
        self.assertIn("classification", status)
        self.assertIn("llm_mapping", status)
        self.assertIn("schema_validation", status)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Use real components but with mocked external dependencies
        with patch('embedding_retriever.SentenceTransformer') as mock_st, \
             patch('entity_extractor.spacy.load') as mock_spacy, \
             patch('llm_mapper.genai') as mock_genai:
            
            # Mock SentenceTransformer
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1] * 384]
            mock_st.return_value = mock_model
            
            # Mock spaCy
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "Mumbai"
            mock_ent.label_ = "GPE"
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            # Mock Gemini
            mock_genai.configure.return_value = None
            mock_model_instance = Mock()
            mock_response = Mock()
            mock_response.text = '{"intent_type": "read", "workspaces": ["sales"], "entities": {}, "confidence": 0.95}'
            mock_model_instance.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model_instance
            
            self.orchestrator = IntentAgentOrchestrator()
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        query = "Show me sales data for Mumbai last month"
        
        result = self.orchestrator.run_intent_agent(query)
        
        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn("intent_analysis", result)
        self.assertIn("metadata", result)
        
        # Verify intent analysis structure
        intent_analysis = result["intent_analysis"]
        required_fields = ["intent_type", "workspaces", "entities", "confidence"]
        for field in required_fields:
            self.assertIn(field, intent_analysis)
        
        # Verify metadata structure
        metadata = result["metadata"]
        self.assertIn("processing_time_seconds", metadata)
        self.assertIn("timestamp", metadata)
        self.assertIn("pipeline_components", metadata)
    
    def test_pipeline_with_different_queries(self):
        """Test pipeline with various query types."""
        test_queries = [
            "Show me sales data",
            "Compare sales between 2023 and 2024",
            "Update customer status",
            "Analyze customer trends",
            "Predict next quarter sales"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                result = self.orchestrator.run_intent_agent(query)
                
                self.assertIsInstance(result, dict)
                self.assertIn("intent_analysis", result)
                
                # Should not have errors
                self.assertFalse(result.get("metadata", {}).get("error", False))


class MockDataGenerator:
    """Helper class to generate mock data for testing."""
    
    @staticmethod
    def generate_mock_intent_data() -> Dict[str, Any]:
        """Generate mock intent data."""
        return {
            "intent_type": "read",
            "workspaces": ["sales"],
            "entities": {
                "dates": ["last month"],
                "locations": ["Mumbai"],
                "quantities": [],
                "products": [],
                "organizations": [],
                "people": [],
                "custom": {}
            },
            "confidence": 0.95,
            "rationale": "Mock intent data for testing",
            "query_type": "simple",
            "time_sensitivity": "historical"
        }
    
    @staticmethod
    def generate_mock_workspace_catalog() -> List[Dict[str, Any]]:
        """Generate mock workspace catalog."""
        return [
            {
                "id": "sales",
                "name": "Sales Analytics",
                "description": "Sales and transaction data",
                "keywords": ["sales", "revenue", "profit"]
            },
            {
                "id": "support",
                "name": "Customer Support",
                "description": "Customer feedback and complaints",
                "keywords": ["support", "feedback", "complaint"]
            }
        ]
    
    @staticmethod
    def generate_mock_entities() -> Dict[str, Any]:
        """Generate mock entities."""
        return {
            "dates": ["last month", "Q3 2024"],
            "locations": ["Mumbai", "Delhi"],
            "quantities": ["top 10", "95%"],
            "products": ["premium plan", "basic service"],
            "organizations": ["Acme Corp", "Tech Solutions"],
            "people": ["John Doe", "Jane Smith"],
            "custom": {"status": "active", "priority": "high"}
        }


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
