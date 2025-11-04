"""
Schema validation module for Intent Agent.

This module provides Pydantic-based schema validation for intent data
and handles validation errors with fallback repair mechanisms.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentEntities(BaseModel):
    """
    Pydantic model for intent entities.
    """
    dates: List[str] = Field(default_factory=list, description="Date-related entities")
    locations: List[str] = Field(default_factory=list, description="Location entities")
    quantities: List[str] = Field(default_factory=list, description="Numeric quantities and measures")
    products: List[str] = Field(default_factory=list, description="Product or service names")
    organizations: List[str] = Field(default_factory=list, description="Company or organization names")
    people: List[str] = Field(default_factory=list, description="Person names")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom entity types")
    
    @validator('dates', 'locations', 'quantities', 'products', 'organizations', 'people', pre=True)
    def normalize_entity_lists(cls, v):
        """Normalize entity lists to remove duplicates and empty values."""
        if isinstance(v, list):
            # Remove duplicates while preserving order
            seen = set()
            normalized = []
            for item in v:
                if isinstance(item, str) and item.strip() and item.strip().lower() not in seen:
                    seen.add(item.strip().lower())
                    normalized.append(item.strip())
            return normalized
        return []
    
    @validator('custom', pre=True)
    def normalize_custom_entities(cls, v):
        """Normalize custom entities dictionary."""
        if isinstance(v, dict):
            return {k: v for k, v in v.items() if v is not None and v != ""}
        return {}


class IntentSchema(BaseModel):
    """
    Pydantic model for the complete intent schema.
    """
    intent_type: str = Field(..., description="The primary intent of the user query")
    workspaces: List[str] = Field(..., min_items=1, description="List of relevant workspace domains")
    entities: IntentEntities = Field(..., description="Extracted entities from the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the intent analysis")
    rationale: Optional[str] = Field(None, description="Explanation of the intent analysis reasoning")
    query_type: str = Field(default="simple", description="Complexity classification of the query")
    time_sensitivity: str = Field(default="historical", description="Time sensitivity of the query")
    
    @validator('intent_type')
    def validate_intent_type(cls, v):
        """Validate intent type is one of the allowed values."""
        allowed_intents = ['read', 'compare', 'update', 'summarize', 'analyze', 'predict']
        if v.lower() not in allowed_intents:
            logger.warning(f"Invalid intent type: {v}, defaulting to 'read'")
            return 'read'
        return v.lower()
    
    @validator('workspaces')
    def validate_workspaces(cls, v):
        """Validate workspaces list."""
        if not v or not isinstance(v, list):
            return ['sales']  # Default workspace
        
        # Normalize workspace IDs
        normalized = []
        for workspace in v:
            if isinstance(workspace, str) and workspace.strip():
                normalized.append(workspace.strip().lower())
        
        return normalized if normalized else ['sales']
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not isinstance(v, (int, float)):
            logger.warning(f"Invalid confidence type: {type(v)}, defaulting to 0.5")
            return 0.5
        
        # Ensure confidence is within bounds
        if v < 0.0:
            return 0.0
        elif v > 1.0:
            return 1.0
        return float(v)
    
    @validator('query_type')
    def validate_query_type(cls, v):
        """Validate query type."""
        allowed_types = ['simple', 'complex', 'multi_intent']
        if v.lower() not in allowed_types:
            return 'simple'
        return v.lower()
    
    @validator('time_sensitivity')
    def validate_time_sensitivity(cls, v):
        """Validate time sensitivity."""
        allowed_sensitivities = ['immediate', 'near_term', 'historical', 'future']
        if v.lower() not in allowed_sensitivities:
            return 'historical'
        return v.lower()
    
    @validator('rationale', pre=True)
    def validate_rationale(cls, v):
        """Validate rationale field."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v.strip()
        return str(v).strip()


class SchemaValidator:
    """
    Handles schema validation for intent data using Pydantic models.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema_path (str, optional): Path to intent schema JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.schema_path = schema_path
        self.schema_def = None
        
        # Load schema definition
        self._load_schema()
    
    def _load_schema(self) -> None:
        """Load schema definition from JSON file."""
        if not self.schema_path:
            self.schema_path = "backend/services/intent_agent/config/intent_schema.json"
        
        try:
            if Path(self.schema_path).exists():
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    self.schema_def = json.load(f)
                self.logger.info(f"Loaded schema definition from {self.schema_path}")
            else:
                self.logger.warning(f"Schema file not found: {self.schema_path}")
                self.schema_def = self._get_default_schema()
        except Exception as e:
            self.logger.error(f"Failed to load schema: {str(e)}")
            self.schema_def = self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default schema definition."""
        return {
            "type": "object",
            "properties": {
                "intent_type": {"type": "string"},
                "workspaces": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "object"},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"}
            },
            "required": ["intent_type", "workspaces", "entities", "confidence"]
        }
    
    def validate_intent_schema(self, intent_data: Dict[str, Any]) -> bool:
        """
        Validate JSON structure for the Intent Agent output.
        
        Args:
            intent_data (Dict): Intent data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validate using Pydantic model
            validated_data = IntentSchema(**intent_data)
            self.logger.info("Intent schema validation successful")
            return True
            
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {str(e)}")
            return False
    
    def validate_and_repair(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate intent data and repair if necessary.
        
        Args:
            intent_data (Dict): Intent data to validate and repair
            
        Returns:
            Dict[str, Any]: Validated and potentially repaired intent data
        """
        try:
            # Try to create validated instance
            validated_data = IntentSchema(**intent_data)
            self.logger.info("Intent data validated successfully")
            return validated_data.dict()
            
        except ValidationError as e:
            self.logger.warning(f"Validation failed, attempting repair: {str(e)}")
            return self._repair_intent_data(intent_data, e)
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {str(e)}")
            return self._create_fallback_intent()
    
    def _repair_intent_data(self, intent_data: Dict[str, Any], validation_error: ValidationError) -> Dict[str, Any]:
        """
        Repair intent data based on validation errors.
        
        Args:
            intent_data (Dict): Original intent data
            validation_error (ValidationError): Validation error details
            
        Returns:
            Dict[str, Any]: Repaired intent data
        """
        self.logger.info("Attempting to repair intent data")
        
        repaired_data = intent_data.copy()
        
        # Handle missing required fields
        if 'intent_type' not in repaired_data:
            repaired_data['intent_type'] = 'read'
        
        if 'workspaces' not in repaired_data:
            repaired_data['workspaces'] = ['sales']
        
        if 'entities' not in repaired_data:
            repaired_data['entities'] = {
                'dates': [], 'locations': [], 'quantities': [], 'products': [],
                'organizations': [], 'people': [], 'custom': {}
            }
        
        if 'confidence' not in repaired_data:
            repaired_data['confidence'] = 0.5
        
        # Handle field-specific repairs
        for error in validation_error.errors():
            field_path = '.'.join(str(loc) for loc in error['loc'])
            error_type = error['type']
            
            if field_path == 'intent_type':
                repaired_data['intent_type'] = 'read'
            elif field_path == 'workspaces':
                repaired_data['workspaces'] = ['sales']
            elif field_path == 'confidence':
                if isinstance(repaired_data.get('confidence'), (int, float)):
                    repaired_data['confidence'] = max(0.0, min(1.0, repaired_data['confidence']))
                else:
                    repaired_data['confidence'] = 0.5
            elif field_path.startswith('entities.'):
                # Handle entity field repairs
                entity_field = field_path.split('.')[1]
                if entity_field in ['dates', 'locations', 'quantities', 'products', 'organizations', 'people']:
                    if entity_field not in repaired_data['entities']:
                        repaired_data['entities'][entity_field] = []
                    elif not isinstance(repaired_data['entities'][entity_field], list):
                        repaired_data['entities'][entity_field] = []
                elif entity_field == 'custom':
                    if entity_field not in repaired_data['entities']:
                        repaired_data['entities'][entity_field] = {}
                    elif not isinstance(repaired_data['entities'][entity_field], dict):
                        repaired_data['entities'][entity_field] = {}
        
        # Try validation again
        try:
            validated_data = IntentSchema(**repaired_data)
            self.logger.info("Intent data repaired successfully")
            return validated_data.dict()
        except ValidationError:
            self.logger.error("Repair failed, using fallback intent")
            return self._create_fallback_intent()
    
    def _create_fallback_intent(self) -> Dict[str, Any]:
        """
        Create fallback intent data when validation and repair fail.
        
        Returns:
            Dict[str, Any]: Fallback intent data
        """
        self.logger.warning("Creating fallback intent data")
        
        return {
            'intent_type': 'read',
            'workspaces': ['sales'],
            'entities': {
                'dates': [],
                'locations': [],
                'quantities': [],
                'products': [],
                'organizations': [],
                'people': [],
                'custom': {}
            },
            'confidence': 0.1,  # Very low confidence for fallback
            'rationale': 'Fallback intent generated due to validation failure',
            'query_type': 'simple',
            'time_sensitivity': 'historical'
        }
    
    def get_schema_definition(self) -> Dict[str, Any]:
        """
        Get the schema definition.
        
        Returns:
            Dict[str, Any]: Schema definition
        """
        return self.schema_def
    
    def validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize entities structure.
        
        Args:
            entities (Dict): Entities to validate
            
        Returns:
            Dict[str, Any]: Validated entities
        """
        try:
            validated_entities = IntentEntities(**entities)
            return validated_entities.dict()
        except ValidationError as e:
            self.logger.warning(f"Entity validation failed: {str(e)}")
            # Return default entities structure
            return {
                'dates': [],
                'locations': [],
                'quantities': [],
                'products': [],
                'organizations': [],
                'people': [],
                'custom': {}
            }
    
    def is_valid_intent_type(self, intent_type: str) -> bool:
        """
        Check if intent type is valid.
        
        Args:
            intent_type (str): Intent type to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        valid_intents = ['read', 'compare', 'update', 'summarize', 'analyze', 'predict']
        return intent_type.lower() in valid_intents
    
    def is_valid_workspace(self, workspace: str) -> bool:
        """
        Check if workspace is valid (basic validation).
        
        Args:
            workspace (str): Workspace to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return isinstance(workspace, str) and len(workspace.strip()) > 0


# Global instance for convenience
_validator_instance = None


def validate_intent_schema(intent_data: Dict[str, Any]) -> bool:
    """
    Validate intent schema using the global validator instance.
    
    Args:
        intent_data (Dict): Intent data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SchemaValidator()
    return _validator_instance.validate_intent_schema(intent_data)


def validate_and_repair(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and repair intent data using the global validator instance.
    
    Args:
        intent_data (Dict): Intent data to validate and repair
        
    Returns:
        Dict[str, Any]: Validated and repaired intent data
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SchemaValidator()
    return _validator_instance.validate_and_repair(intent_data)


if __name__ == "__main__":
    # Test the schema validator
    validator = SchemaValidator()
    
    # Test valid intent data
    valid_intent = {
        "intent_type": "read",
        "workspaces": ["sales", "marketing"],
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
        "rationale": "Clear request for sales data",
        "query_type": "simple",
        "time_sensitivity": "historical"
    }
    
    print("Testing valid intent data:")
    print(f"Validation result: {validator.validate_intent_schema(valid_intent)}")
    
    # Test invalid intent data
    invalid_intent = {
        "intent_type": "invalid_type",
        "workspaces": [],
        "entities": "invalid_entities",
        "confidence": 1.5
    }
    
    print("\nTesting invalid intent data:")
    print(f"Validation result: {validator.validate_intent_schema(invalid_intent)}")
    
    # Test repair functionality
    print("\nTesting repair functionality:")
    repaired = validator.validate_and_repair(invalid_intent)
    print(f"Repaired data: {json.dumps(repaired, indent=2)}")
    
    # Test entity validation
    print("\nTesting entity validation:")
    test_entities = {
        "dates": ["last month", "last month", ""],  # Duplicate and empty
        "locations": ["Mumbai", "mumbai"],  # Duplicate (case insensitive)
        "custom": {"status": "active", "empty": ""}  # Empty value
    }
    
    validated_entities = validator.validate_entities(test_entities)
    print(f"Validated entities: {json.dumps(validated_entities, indent=2)}")
