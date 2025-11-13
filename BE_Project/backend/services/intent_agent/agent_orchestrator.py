"""
Agent orchestrator module for Intent Agent.

This module orchestrates the complete intent analysis pipeline by combining
all components: preprocessing, embedding retrieval, entity extraction,
classification, LLM mapping, and schema validation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import all component modules
from .preprocessor import TextPreprocessor
from .embedding_retriever import EmbeddingRetriever
from .entity_extractor import EntityExtractor
from .classifier import IntentClassifier
from .llm_mapper import LLMMapper
from .schema_validator import SchemaValidator
from .context_validator import ContextValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentAgentOrchestrator:
    """
    Orchestrates the complete intent analysis pipeline.
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 workspace_catalog_path: Optional[str] = None,
                 faiss_index_path: Optional[str] = None):
        """
        Initialize the intent agent orchestrator.
        
        Args:
            gemini_api_key (str, optional): Google API key for Gemini
            workspace_catalog_path (str, optional): Path to workspace catalog
            faiss_index_path (str, optional): Path to FAISS index
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.embedding_retriever = EmbeddingRetriever(index_path=faiss_index_path)
        self.entity_extractor = EntityExtractor()
        self.classifier = IntentClassifier(workspace_catalog_path=workspace_catalog_path)
        self.schema_validator = SchemaValidator()
        self.context_validator = ContextValidator()
        
        # Initialize LLM mapper (may fail if API key not provided)
        try:
            self.llm_mapper = LLMMapper(api_key=gemini_api_key)
            self.llm_available = True
            self.logger.info("LLM mapper initialized successfully")
        except Exception as e:
            self.logger.warning(f"LLM mapper initialization failed: {str(e)}")
            self.llm_mapper = None
            self.llm_available = False
        
        # Pipeline configuration
        self.enable_llm_mapping = self.llm_available
        self.enable_similarity_retrieval = True
        self.enable_entity_extraction = True
        self.enable_classification = True
        self.enable_context_validation = True
        
        self.logger.info("Intent Agent Orchestrator initialized")
    
    def run_intent_agent(self, user_query: str) -> Dict[str, Any]:
        """
        End-to-end pipeline to extract and return structured intent information.
        
        Args:
            user_query (str): Raw user query text
            
        Returns:
            Dict[str, Any]: Structured intent data with metadata
        """
        if not user_query or not isinstance(user_query, str):
            return self._create_error_response("Invalid input: user_query must be a non-empty string")
        
        start_time = datetime.now()
        self.logger.info(f"Starting intent analysis for: {user_query[:100]}...")
        
        try:
            # Step 0: Validate context (check if query makes sense)
            if self.enable_context_validation:
                validation_result = self._validate_context(user_query)
                if not validation_result['is_valid']:
                    return self._create_context_error_response(
                        validation_result, user_query, start_time
                    )
            
            # Step 1: Preprocess the query
            processed_query = self._preprocess_query(user_query)
            
            # Step 2: Generate embeddings and retrieve similar queries
            similar_queries = self._retrieve_similar_queries(processed_query)
            
            # Step 3: Extract entities
            entities = self._extract_entities(processed_query)
            
            # Step 4: Classify intent and workspaces
            predicted_intent = self._classify_intent(processed_query)
            predicted_workspaces = self._classify_workspaces(processed_query)
            
            # Step 5: Build context for LLM
            context = self._build_context(
                processed_query, entities, similar_queries, 
                predicted_intent, predicted_workspaces
            )
            
            # Step 6: Use LLM for final mapping (if available)
            if self.enable_llm_mapping:
                intent_data = self._llm_mapping(processed_query, context)
            else:
                intent_data = self._rule_based_mapping(processed_query, context)
            
            # Step 7: Validate and repair schema
            validated_intent = self._validate_schema(intent_data)
            
            # Step 8: Add metadata and finalize response
            final_response = self._finalize_response(
                validated_intent, user_query, processed_query, start_time
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Intent analysis completed in {processing_time:.2f} seconds")
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {str(e)}")
            return self._create_error_response(f"Intent analysis failed: {str(e)}")
    
    def _validate_context(self, user_query: str) -> Dict[str, Any]:
        """Validate the context of the user query."""
        try:
            validation_result = self.context_validator.validate_context(user_query)
            if validation_result['is_valid']:
                self.logger.info("Context validation passed")
            else:
                self.logger.warning(f"Context validation failed: {validation_result['message']}")
            return validation_result
        except Exception as e:
            self.logger.warning(f"Context validation failed: {str(e)}, proceeding with query")
            # If validation fails due to error, proceed with query
            return {'is_valid': True, 'message': 'Validation error, proceeding'}
    
    def _preprocess_query(self, user_query: str) -> str:
        """Preprocess the user query."""
        try:
            processed = self.preprocessor.preprocess_text(user_query)
            self.logger.info(f"Query preprocessed: {len(user_query)} -> {len(processed)} chars")
            return processed
        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {str(e)}, using original query")
            return user_query
    
    def _retrieve_similar_queries(self, processed_query: str) -> List[str]:
        """Retrieve similar queries using embeddings."""
        if not self.enable_similarity_retrieval:
            return []
        
        try:
            embedding = self.embedding_retriever.get_embedding(processed_query)
            similar_queries = self.embedding_retriever.retrieve_similar_queries(embedding, top_k=3)
            self.logger.info(f"Retrieved {len(similar_queries)} similar queries")
            return similar_queries
        except Exception as e:
            self.logger.warning(f"Similar query retrieval failed: {str(e)}")
            return []
    
    def _extract_entities(self, processed_query: str) -> Dict[str, Any]:
        """Extract entities from the processed query."""
        if not self.enable_entity_extraction:
            return {
                'dates': [], 'locations': [], 'quantities': [], 'products': [],
                'organizations': [], 'people': [], 'custom': {}
            }
        
        try:
            entities = self.entity_extractor.extract_entities(processed_query)
            self.logger.info(f"Extracted entities: {self._summarize_entities(entities)}")
            return entities
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {str(e)}")
            return {
                'dates': [], 'locations': [], 'quantities': [], 'products': [],
                'organizations': [], 'people': [], 'custom': {}
            }
    
    def _classify_intent(self, processed_query: str) -> str:
        """Classify the intent type."""
        if not self.enable_classification:
            return "read"
        
        try:
            intent = self.classifier.predict_intent_type(processed_query)
            self.logger.info(f"Predicted intent type: {intent}")
            return intent
        except Exception as e:
            self.logger.warning(f"Intent classification failed: {str(e)}")
            return "read"
    
    def _classify_workspaces(self, processed_query: str) -> List[Dict[str, Any]]:
        """Classify relevant workspaces."""
        if not self.enable_classification:
            return [{'id': 'sales', 'score': 1.0, 'confidence': 0.5}]
        
        try:
            workspaces = self.classifier.predict_workspace(processed_query)
            self.logger.info(f"Predicted {len(workspaces)} workspaces")
            return workspaces
        except Exception as e:
            self.logger.warning(f"Workspace classification failed: {str(e)}")
            return [{'id': 'sales', 'score': 1.0, 'confidence': 0.5}]
    
    def _build_context(self, 
                      processed_query: str,
                      entities: Dict[str, Any],
                      similar_queries: List[str],
                      predicted_intent: str,
                      predicted_workspaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context dictionary for LLM mapping."""
        return {
            'processed_query': processed_query,
            'entities': entities,
            'similar_queries': similar_queries,
            'predicted_intent': predicted_intent,
            'predicted_workspaces': predicted_workspaces,
            'query_complexity': self.classifier.classify_query_complexity(processed_query),
            'time_sensitivity': self.classifier.detect_time_sensitivity(processed_query)
        }
    
    def _llm_mapping(self, processed_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM for intent mapping. Attempt lazy LLM init if not available."""
        try:
            # Lazy initialize the LLM mapper if an API key is present at runtime
            if not self.llm_available:
                try:
                    api_key = os.getenv("GEMINI_API_KEY")
                    if api_key:
                        self.llm_mapper = LLMMapper(api_key=api_key)
                        self.llm_available = True
                        self.logger.info("LLM mapper lazily initialized from environment variable")
                except Exception as e:
                    # Keep going; we will fall back to rule-based mapping
                    self.logger.warning(f"Lazy LLM initialization failed: {e}")

            if self.llm_available and self.llm_mapper:
                intent_data = self.llm_mapper.map_intent_with_llm(processed_query, context)
                self.logger.info("LLM mapping completed successfully")
                return intent_data
            else:
                self.logger.info("LLM not available, using rule-based mapping")
                return self._rule_based_mapping(processed_query, context)

        except Exception as e:
            self.logger.warning(f"LLM mapping failed: {str(e)}, falling back to rule-based mapping")
            return self._rule_based_mapping(processed_query, context)
    
    def _rule_based_mapping(self, processed_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based intent mapping when LLM is not available."""
        self.logger.info("Using rule-based mapping")
        
        # Extract data from context
        entities = context.get('entities', {})
        predicted_intent = context.get('predicted_intent', 'read')
        predicted_workspaces = context.get('predicted_workspaces', [])
        
        # Build workspace list
        workspaces = [w['id'] for w in predicted_workspaces[:2]] if predicted_workspaces else ['sales']
        
        # Calculate confidence based on available data
        confidence = 0.5  # Base confidence
        if entities and any(entities.values()):
            confidence += 0.2
        if predicted_workspaces and predicted_workspaces[0].get('confidence', 0) > 0.7:
            confidence += 0.2
        
        return {
            'intent_type': predicted_intent,
            'workspaces': workspaces,
            'entities': entities,
            'confidence': min(confidence, 1.0),
            'rationale': f"Rule-based mapping: intent={predicted_intent}, workspaces={workspaces}",
            'query_type': context.get('query_complexity', 'simple'),
            'time_sensitivity': context.get('time_sensitivity', 'historical')
        }
    
    def _validate_schema(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and repair the intent schema."""
        try:
            validated_data = self.schema_validator.validate_and_repair(intent_data)
            self.logger.info("Schema validation completed successfully")
            return validated_data
        except Exception as e:
            self.logger.warning(f"Schema validation failed: {str(e)}")
            return intent_data
    
    def _finalize_response(self, 
                          validated_intent: Dict[str, Any],
                          original_query: str,
                          processed_query: str,
                          start_time: datetime) -> Dict[str, Any]:
        """Add metadata and finalize the response."""
        processing_time = (datetime.now() - start_time).total_seconds()
        # Include top-level keys expected by downstream components (QueryExecutor)
        response = {
            'intent_analysis': validated_intent,
            # Generated SQL and embedding may be populated by LLM mapper in future
            'generated_sql': validated_intent.get('generated_sql') if isinstance(validated_intent, dict) else None,
            'query_embedding': validated_intent.get('query_embedding') if isinstance(validated_intent, dict) else None,
            # Validation metadata for higher-level orchestration
            'is_valid': True,
            'validation': None,
            'metadata': {
                'original_query': original_query,
                'processed_query': processed_query,
                'processing_time_seconds': round(processing_time, 3),
                'timestamp': datetime.now().isoformat(),
                'pipeline_components': {
                    'preprocessing': True,
                    'context_validation': self.enable_context_validation,
                    'embedding_retrieval': self.enable_similarity_retrieval,
                    'entity_extraction': self.enable_entity_extraction,
                    'classification': self.enable_classification,
                    'llm_mapping': self.enable_llm_mapping,
                    'schema_validation': True
                },
                'version': '1.0.0'
            }
        }

        # If no generated_sql was returned by the intent mapping step, attempt a safe
        # fallback using entity-based SQL generation or workspace catalog
        if not response.get('generated_sql'):
            try:
                # Build SQL from entities if we have them
                generated_sql = self._generate_sql_from_intent(validated_intent, original_query)
                if generated_sql:
                    response['generated_sql'] = generated_sql
                    self.logger.info(f"Generated SQL from intent entities: {generated_sql}")
                else:
                    # Fall back to workspace sample_sql
                    response['generated_sql'] = self._get_workspace_sample_sql(validated_intent)
            except Exception as e:
                self.logger.warning(f"SQL generation fallback failed: {e}, will use workspace sample_sql")
                response['generated_sql'] = self._get_workspace_sample_sql(validated_intent)

        return response
    
    def _generate_sql_from_intent(self, validated_intent: Dict[str, Any], original_query: str) -> Optional[str]:
        """Generate basic SQL from intent entities and workspace."""
        try:
            if not isinstance(validated_intent, dict):
                return None
            
            intent_type = validated_intent.get('intent_type', 'read')
            workspaces = validated_intent.get('workspaces', [])
            entities = validated_intent.get('entities', {})
            
            if not workspaces or intent_type not in ['read', 'summarize', 'compare', 'analyze']:
                return None
            
            workspace = workspaces[0] if isinstance(workspaces, list) else workspaces
            
            # Map workspace to table name
            table_map = {
                'sales': 'sales_transactions',
                'support': 'support_tickets',
                'marketing': 'campaigns',
                'hr': 'employees',
                'finance': 'expenses',
                'operations': 'inventory'
            }
            
            table_name = table_map.get(workspace, f"{workspace}_data")
            
            # Build WHERE clause from entities
            where_clauses = []
            
            # Add date filters
            if entities.get('dates') and isinstance(entities['dates'], list):
                for date_entity in entities['dates'][:1]:  # Use first date entity
                    date_str = str(date_entity).lower()
                    # Try to extract month/year
                    if 'april' in date_str or '04' in date_str or '4' in date_str:
                        where_clauses.append("MONTH(date) = 4")
                    if '2025' in date_str:
                        where_clauses.append("YEAR(date) = 2025")
                    elif '2024' in date_str:
                        where_clauses.append("YEAR(date) = 2024")
            
            # Add location filters
            if entities.get('locations') and isinstance(entities['locations'], list):
                for location in entities['locations'][:1]:
                    where_clauses.append(f"location LIKE '%{location}%'")
            
            # Build the SQL query
            if intent_type == 'summarize':
                sql = f"SELECT COUNT(*) as total_records FROM {table_name}"
            else:
                sql = f"SELECT * FROM {table_name}"
            
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            
            sql += " LIMIT 100;"
            
            return sql
            
        except Exception as e:
            self.logger.debug(f"SQL generation from intent failed: {e}")
            return None
    
    def _get_workspace_sample_sql(self, validated_intent: Dict[str, Any]) -> Optional[str]:
        """Get sample SQL from workspace catalog."""
        try:
            ws_list = []
            if isinstance(validated_intent, dict) and validated_intent.get('workspaces'):
                ws_list = validated_intent.get('workspaces')

            picked_sql = None
            if ws_list and isinstance(ws_list, list):
                # Take first workspace id
                ws_id = ws_list[0]
                # Find in classifier catalog
                for ws in self.classifier.workspace_catalog:
                    if ws.get('id') == ws_id or ws.get('id') == str(ws_id):
                        # Prefer explicit sample_sql
                        picked_sql = ws.get('sample_sql')
                        if not picked_sql and ws.get('tables'):
                            picked_sql = f"SELECT * FROM {ws['tables'][0]} LIMIT 100;"
                        break

            return picked_sql
        except Exception:
            return None

    
    def _create_context_error_response(self, 
                                      validation_result: Dict[str, Any],
                                      user_query: str,
                                      start_time: datetime) -> Dict[str, Any]:
        """Create error response for context validation failures."""
        processing_time = (datetime.now() - start_time).total_seconds()
        # Provide consistent top-level fields so the caller can inspect validation status
        return {
            'intent_analysis': {
                'intent_type': 'read',
                'workspaces': ['sales'],
                'entities': {
                    'dates': [], 'locations': [], 'quantities': [], 'products': [],
                    'organizations': [], 'people': [], 'custom': {}
                },
                'confidence': validation_result.get('confidence', 0.0),
                'rationale': f"Context validation failed: {validation_result.get('message', 'Unknown error')}",
                'query_type': 'invalid',
                'time_sensitivity': 'historical',
                'validation_issues': validation_result.get('issues', [])
            },
            'generated_sql': None,
            'query_embedding': None,
            'is_valid': False,
            'validation': validation_result,
            'metadata': {
                'error': True,
                'error_type': 'context_validation_failed',
                'error_message': validation_result.get('message', 'Query contains logical inconsistencies'),
                'validation_issues': validation_result.get('issues', []),
                'original_query': user_query,
                'processing_time_seconds': round(processing_time, 3),
                'timestamp': datetime.now().isoformat(),
                'pipeline_components': {
                    'preprocessing': False,
                    'context_validation': True,
                    'embedding_retrieval': False,
                    'entity_extraction': False,
                    'classification': False,
                    'llm_mapping': False,
                    'schema_validation': False
                },
                'version': '1.0.0'
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'intent_analysis': {
                'intent_type': 'read',
                'workspaces': ['sales'],
                'entities': {
                    'dates': [], 'locations': [], 'quantities': [], 'products': [],
                    'organizations': [], 'people': [], 'custom': {}
                },
                'confidence': 0.0,
                'rationale': f"Error: {error_message}",
                'query_type': 'simple',
                'time_sensitivity': 'historical'
            },
            'generated_sql': None,
            'query_embedding': None,
            'is_valid': False,
            'validation': {'error': error_message},
            'metadata': {
                'error': True,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
    
    def _summarize_entities(self, entities: Dict[str, Any]) -> str:
        """Create a summary of extracted entities."""
        summary_parts = []
        for key, value in entities.items():
            if isinstance(value, list) and value:
                summary_parts.append(f"{key}: {len(value)}")
            elif isinstance(value, dict) and value:
                summary_parts.append(f"{key}: {len(value)}")
        return ", ".join(summary_parts) if summary_parts else "none"
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the status of all pipeline components."""
        return {
            'preprocessing': True,
            'context_validation': self.enable_context_validation,
            'embedding_retrieval': self.enable_similarity_retrieval,
            'entity_extraction': self.enable_entity_extraction,
            'classification': self.enable_classification,
            'llm_mapping': self.enable_llm_mapping,
            'schema_validation': True,
            'faiss_index_size': self.embedding_retriever.index.ntotal if hasattr(self.embedding_retriever, 'index') else 0,
            'workspace_catalog_size': len(self.classifier.workspace_catalog)
        }
    
    def initialize_sample_data(self) -> None:
        """Initialize the system with sample data for testing."""
        try:
            self.embedding_retriever.initialize_with_sample_data()
            self.logger.info("Sample data initialization completed")
        except Exception as e:
            self.logger.warning(f"Sample data initialization failed: {str(e)}")


# Global instance for convenience
_orchestrator_instance = None


def run_intent_agent(user_query: str) -> Dict[str, Any]:
    """
    Run the complete intent agent pipeline.
    
    Args:
        user_query (str): User query text
        
    Returns:
        Dict[str, Any]: Structured intent analysis result
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = IntentAgentOrchestrator()
    return _orchestrator_instance.run_intent_agent(user_query)


if __name__ == "__main__":
    # Test the orchestrator
    print("Initializing Intent Agent Orchestrator...")
    
    # Initialize with sample data
    orchestrator = IntentAgentOrchestrator()
    orchestrator.initialize_sample_data()
    
    # Test queries
    test_queries = [
        "Show me sales in Mumbai for last month",
        "Compare complaint volumes between 2024 and 2025",
        "What's the average response time for support tickets in Q3 2024?",
        "Update customer status to active for all customers with pending orders",
        "Analyze the trend in customer satisfaction over the past year",
        "Predict sales for next quarter based on current trends"
    ]
    
    print("\nTesting Intent Agent Pipeline:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            result = orchestrator.run_intent_agent(query)
            
            # Display key results
            intent_analysis = result['intent_analysis']
            print(f"Intent Type: {intent_analysis['intent_type']}")
            print(f"Workspaces: {intent_analysis['workspaces']}")
            print(f"Confidence: {intent_analysis['confidence']:.2f}")
            print(f"Rationale: {intent_analysis['rationale']}")
            
            # Show entities if any
            entities = intent_analysis['entities']
            entity_summary = []
            for key, values in entities.items():
                if values:
                    entity_summary.append(f"{key}: {values}")
            if entity_summary:
                print(f"Entities: {', '.join(entity_summary)}")
            
            # Show metadata
            metadata = result['metadata']
            print(f"Processing Time: {metadata['processing_time_seconds']}s")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 40)
    
    # Show pipeline status
    print(f"\nPipeline Status:")
    status = orchestrator.get_pipeline_status()
    for component, status_val in status.items():
        print(f"  {component}: {status_val}")
    
    print("\nIntent Agent testing completed!")
