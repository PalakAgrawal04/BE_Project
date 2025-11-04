"""
LLM mapping module for Intent Agent.

This module uses Google's Gemini API to synthesize final structured intent
from text and retrieved context, providing intelligent intent mapping.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google-generativeai not installed. LLM mapping will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMMapper:
    """
    Handles LLM-powered intent mapping using Google's Gemini API.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-pro",
                 temperature: float = 0.0,
                 system_prompt_path: Optional[str] = None,
                 few_shot_examples_path: Optional[str] = None):
        """
        Initialize the LLM mapper.
        
        Args:
            api_key (str, optional): Google API key
            model_name (str): Gemini model name to use
            temperature (float): Temperature for generation (0.0 for deterministic)
            system_prompt_path (str, optional): Path to system prompt file
            few_shot_examples_path (str, optional): Path to few-shot examples file
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.temperature = temperature
        
        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini
        self._configure_gemini()
        
        # Load prompts and examples
        self.system_prompt = ""
        self.few_shot_examples = []
        self._load_prompts(system_prompt_path, few_shot_examples_path)
    
    def _configure_gemini(self) -> None:
        """Configure the Gemini API client."""
        if not genai:
            raise ImportError("google-generativeai package is required for LLM mapping")
        
        try:
            genai.configure(api_key=self.api_key)
            self.logger.info("Gemini API configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise
    
    def _load_prompts(self, system_prompt_path: Optional[str], few_shot_examples_path: Optional[str]) -> None:
        """Load system prompt and few-shot examples from files."""
        # Load system prompt
        if not system_prompt_path:
            system_prompt_path = "backend/services/intent_agent/prompts/intent_system_prompt.txt"
        
        try:
            if Path(system_prompt_path).exists():
                with open(system_prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompt = f.read()
                self.logger.info(f"Loaded system prompt from {system_prompt_path}")
            else:
                self.logger.warning(f"System prompt file not found: {system_prompt_path}")
                self.system_prompt = self._get_default_system_prompt()
        except Exception as e:
            self.logger.error(f"Failed to load system prompt: {str(e)}")
            self.system_prompt = self._get_default_system_prompt()
        
        # Load few-shot examples
        if not few_shot_examples_path:
            few_shot_examples_path = "backend/services/intent_agent/prompts/few_shot_examples.json"
        
        try:
            if Path(few_shot_examples_path).exists():
                with open(few_shot_examples_path, 'r', encoding='utf-8') as f:
                    self.few_shot_examples = json.load(f)
                self.logger.info(f"Loaded {len(self.few_shot_examples)} few-shot examples")
            else:
                self.logger.warning(f"Few-shot examples file not found: {few_shot_examples_path}")
                self.few_shot_examples = []
        except Exception as e:
            self.logger.error(f"Failed to load few-shot examples: {str(e)}")
            self.few_shot_examples = []
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if file loading fails."""
        return """You are the Intent Agent for IntelliQuery.
Given a user query, your goal is to extract:
1. Intent type (read, compare, update, summarize, analyze, predict)
2. Workspaces/domains relevant to the query
3. Entities (time, location, quantity, etc.)
4. Confidence score (0–1)
Return output in strict JSON per schema."""
    
    def map_intent_with_llm(self, user_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini to synthesize final structured intent from text and context.
        
        Args:
            user_text (str): User query text
            context (Dict): Context including entities, similar queries, etc.
            
        Returns:
            Dict[str, Any]: Structured intent data
        """
        if not user_text or not isinstance(user_text, str):
            raise ValueError("User text must be a non-empty string")
        
        self.logger.info(f"Mapping intent with LLM for: {user_text[:50]}...")
        
        try:
            # Build the prompt
            prompt = self._build_prompt(user_text, context)
            
            # Generate response using Gemini
            response = self._generate_response(prompt)
            
            # Parse and validate response
            intent_data = self._parse_response(response)
            
            self.logger.info(f"LLM mapping completed successfully")
            return intent_data
            
        except Exception as e:
            self.logger.error(f"LLM mapping failed: {str(e)}")
            # Return fallback intent data
            return self._create_fallback_intent(user_text, context)
    
    def _build_prompt(self, user_text: str, context: Dict[str, Any]) -> str:
        """
        Build the complete prompt for Gemini.
        
        Args:
            user_text (str): User query text
            context (Dict): Context data
            
        Returns:
            str: Complete prompt
        """
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(self.system_prompt)
        
        # Few-shot examples
        if self.few_shot_examples:
            prompt_parts.append("\n## Examples:")
            for example in self.few_shot_examples[:3]:  # Use first 3 examples
                prompt_parts.append(f"Query: {example['user_query']}")
                prompt_parts.append(f"Context: {example.get('context', [])}")
                prompt_parts.append(f"Intent: {json.dumps(example['intent'], indent=2)}")
                prompt_parts.append("")
        
        # Current query and context
        prompt_parts.append("## Current Query:")
        prompt_parts.append(f"User Query: {user_text}")
        
        if context:
            prompt_parts.append("\n## Context:")
            if context.get('entities'):
                prompt_parts.append(f"Extracted Entities: {json.dumps(context['entities'], indent=2)}")
            
            if context.get('similar_queries'):
                prompt_parts.append(f"Similar Queries: {context['similar_queries']}")
            
            if context.get('predicted_intent'):
                prompt_parts.append(f"Predicted Intent: {context['predicted_intent']}")
            
            if context.get('predicted_workspaces'):
                prompt_parts.append(f"Predicted Workspaces: {[w['id'] for w in context['predicted_workspaces'][:3]]}")
        
        prompt_parts.append("\n## Your Response:")
        prompt_parts.append("Provide the structured intent analysis in JSON format:")
        
        return "\n".join(prompt_parts)
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using Gemini API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated response
        """
        try:
            # Create model instance
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Generate content
            response = model.generate_content(prompt)
            
            if response.text:
                self.logger.info("Generated response from Gemini")
                return response.text
            else:
                raise ValueError("Empty response from Gemini")
                
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini response and extract JSON.
        
        Args:
            response (str): Raw response from Gemini
            
        Returns:
            Dict[str, Any]: Parsed intent data
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                intent_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['intent_type', 'workspaces', 'entities', 'confidence']
                for field in required_fields:
                    if field not in intent_data:
                        self.logger.warning(f"Missing required field: {field}")
                        intent_data[field] = self._get_default_value(field)
                
                return intent_data
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse response: {str(e)}")
            raise
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'intent_type': 'read',
            'workspaces': ['sales'],
            'entities': {'dates': [], 'locations': [], 'quantities': [], 'products': [], 'organizations': [], 'people': [], 'custom': {}},
            'confidence': 0.5,
            'rationale': 'Generated with fallback values',
            'query_type': 'simple',
            'time_sensitivity': 'historical'
        }
        return defaults.get(field, None)
    
    def _create_fallback_intent(self, user_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback intent data when LLM fails.
        
        Args:
            user_text (str): User query text
            context (Dict): Context data
            
        Returns:
            Dict[str, Any]: Fallback intent data
        """
        self.logger.warning("Creating fallback intent data")
        
        # Use context data if available
        intent_type = context.get('predicted_intent', 'read')
        workspaces = [w['id'] for w in context.get('predicted_workspaces', [])[:2]]
        if not workspaces:
            workspaces = ['sales']
        
        entities = context.get('entities', {})
        if not entities:
            entities = {'dates': [], 'locations': [], 'quantities': [], 'products': [], 'organizations': [], 'people': [], 'custom': {}}
        
        return {
            'intent_type': intent_type,
            'workspaces': workspaces,
            'entities': entities,
            'confidence': 0.3,  # Low confidence for fallback
            'rationale': 'Fallback intent generated due to LLM failure',
            'query_type': 'simple',
            'time_sensitivity': 'historical'
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_prompt = "Say 'Hello, world!' in JSON format: {\"message\": \"Hello, world!\"}"
            response = self._generate_response(test_prompt)
            
            # Try to parse as JSON
            json.loads(response)
            self.logger.info("Gemini API connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Gemini API connection test failed: {str(e)}")
            return False


# Global instance for convenience
_mapper_instance = None


def map_intent_with_llm(user_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map intent using the global LLM mapper instance.
    
    Args:
        user_text (str): User query text
        context (Dict): Context data
        
    Returns:
        Dict[str, Any]: Structured intent data
    """
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = LLMMapper()
    return _mapper_instance.map_intent_with_llm(user_text, context)


if __name__ == "__main__":
    # Test the LLM mapper
    try:
        mapper = LLMMapper()
        
        # Test connection
        if mapper.test_connection():
            print("✓ Gemini API connection successful")
        else:
            print("✗ Gemini API connection failed")
            exit(1)
        
        # Test intent mapping
        test_query = "Show me sales in Mumbai for last month"
        test_context = {
            'entities': {
                'dates': ['last month'],
                'locations': ['Mumbai'],
                'quantities': [],
                'products': [],
                'organizations': [],
                'people': [],
                'custom': {}
            },
            'similar_queries': ['sales data for Mumbai', 'monthly sales reports'],
            'predicted_intent': 'read',
            'predicted_workspaces': [{'id': 'sales', 'score': 8.5}]
        }
        
        result = mapper.map_intent_with_llm(test_query, test_context)
        print(f"\nIntent mapping result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error testing LLM mapper: {e}")
        print("Make sure GEMINI_API_KEY environment variable is set")
