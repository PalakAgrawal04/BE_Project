"""
Entity extraction module for Intent Agent.

This module uses spaCy NER and custom patterns to extract structured
entities from user queries, including dates, locations, quantities, etc.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
import dateparser
import spacy
from spacy import displacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Handles entity extraction using spaCy NER and custom patterns.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name (str): Name of the spaCy model to load
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy model
        self.nlp = None
        self._load_spacy_model()
        
        # Custom patterns for additional entity types
        self._compile_patterns()
    
    def _load_spacy_model(self) -> None:
        """Load the spaCy model."""
        try:
            self.logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            self.logger.info("spaCy model loaded successfully")
        except OSError:
            self.logger.error(f"spaCy model '{self.model_name}' not found. Please install it with: python -m spacy download {self.model_name}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {str(e)}")
            raise
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for custom entity extraction."""
        # Date patterns
        self.date_patterns = [
            r'\b(last|next|this)\s+(month|year|quarter|week|day)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b(q[1-4])\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b',
            r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
        ]
        
        # Quantity patterns
        self.quantity_patterns = [
            r'\b(top|bottom|first|last)\s+\d+\b',
            r'\b\d+\s*(percent|%|pct)\b',
            r'\b\d+\s*(million|billion|thousand|k|m|b)\b',
            r'\b\d+\.?\d*\s*(dollars?|\$|usd|eur|gbp)\b',
            r'\b(high|low|average|total|sum|count|max|min)\b'
        ]
        
        # Product patterns (common business terms)
        self.product_patterns = [
            r'\b\w*\s*(product|item|service|plan|package|subscription)\b',
            r'\b(premium|basic|standard|enterprise|pro|free)\s+\w*\b'
        ]
        
        # Compile patterns
        self.compiled_date_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]
        self.compiled_quantity_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.quantity_patterns]
        self.compiled_product_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.product_patterns]
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract and normalize entities from user query.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted entities
        """
        if not text or not isinstance(text, str):
            return self._empty_entities()
        
        self.logger.info(f"Extracting entities from: {text[:50]}...")
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Initialize result structure
            entities = {
                "dates": [],
                "locations": [],
                "quantities": [],
                "products": [],
                "organizations": [],
                "people": [],
                "custom": {}
            }
            
            # Extract spaCy entities
            entities.update(self._extract_spacy_entities(doc))
            
            # Extract custom pattern entities
            custom_entities = self._extract_custom_entities(text)
            for key, value in custom_entities.items():
                if key in entities:
                    entities[key].extend(value)
                else:
                    entities["custom"][key] = value
            
            # Normalize and deduplicate entities
            entities = self._normalize_entities(entities)
            
            self.logger.info(f"Extracted entities: {self._summarize_entities(entities)}")
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error during entity extraction: {str(e)}")
            return self._empty_entities()
    
    def _extract_spacy_entities(self, doc) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        entities = {
            "dates": [],
            "locations": [],
            "quantities": [],
            "products": [],
            "organizations": [],
            "people": []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            
            if ent.label_ == "DATE":
                # Parse and normalize date
                normalized_date = self._normalize_date(entity_text)
                if normalized_date:
                    entities["dates"].append(normalized_date)
                    
            elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                entities["locations"].append(entity_text)
                
            elif ent.label_ == "CARDINAL":  # Cardinal numbers
                entities["quantities"].append(entity_text)
                
            elif ent.label_ == "ORG":  # Organization
                entities["organizations"].append(entity_text)
                
            elif ent.label_ == "PERSON":  # Person
                entities["people"].append(entity_text)
                
            elif ent.label_ == "MONEY":  # Money
                entities["quantities"].append(entity_text)
                
            elif ent.label_ == "PERCENT":  # Percentage
                entities["quantities"].append(entity_text)
        
        return entities
    
    def _extract_custom_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using custom regex patterns."""
        entities = {
            "dates": [],
            "quantities": [],
            "products": []
        }
        
        # Extract dates
        for pattern in self.compiled_date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = " ".join(match)
                else:
                    match_text = match
                
                normalized_date = self._normalize_date(match_text)
                if normalized_date:
                    entities["dates"].append(normalized_date)
        
        # Extract quantities
        for pattern in self.compiled_quantity_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = " ".join(match)
                else:
                    match_text = match
                entities["quantities"].append(match_text)
        
        # Extract products
        for pattern in self.compiled_product_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = " ".join(match)
                else:
                    match_text = match
                entities["products"].append(match_text)
        
        return entities
    
    def _normalize_date(self, date_text: str) -> Optional[str]:
        """
        Normalize date expressions using dateparser.
        
        Args:
            date_text (str): Raw date text
            
        Returns:
            Optional[str]: Normalized date string or None if parsing fails
        """
        try:
            # Parse the date
            parsed_date = dateparser.parse(date_text)
            if parsed_date:
                # Format as ISO date string
                return parsed_date.strftime("%Y-%m-%d")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to normalize date '{date_text}': {str(e)}")
            return None
    
    def _normalize_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and deduplicate entities."""
        normalized = {}
        
        for key, value in entities.items():
            if isinstance(value, list):
                # Remove duplicates while preserving order
                seen = set()
                unique_items = []
                for item in value:
                    item_lower = item.lower().strip()
                    if item_lower not in seen and item_lower:
                        seen.add(item_lower)
                        unique_items.append(item.strip())
                normalized[key] = unique_items
            else:
                normalized[key] = value
        
        return normalized
    
    def _summarize_entities(self, entities: Dict[str, Any]) -> str:
        """Create a summary of extracted entities."""
        summary_parts = []
        for key, value in entities.items():
            if isinstance(value, list) and value:
                summary_parts.append(f"{key}: {len(value)}")
            elif isinstance(value, dict) and value:
                summary_parts.append(f"{key}: {len(value)}")
        
        return ", ".join(summary_parts) if summary_parts else "none"
    
    def _empty_entities(self) -> Dict[str, Any]:
        """Return empty entities structure."""
        return {
            "dates": [],
            "locations": [],
            "quantities": [],
            "products": [],
            "organizations": [],
            "people": [],
            "custom": {}
        }
    
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract temporal expressions with their types.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: List of temporal expressions with metadata
        """
        temporal_expressions = []
        
        # Relative time patterns
        relative_patterns = [
            (r'\b(last|previous)\s+(month|year|quarter|week|day)\b', 'past'),
            (r'\b(next|upcoming)\s+(month|year|quarter|week|day)\b', 'future'),
            (r'\bthis\s+(month|year|quarter|week|day)\b', 'current'),
            (r'\btoday\b', 'current'),
            (r'\byesterday\b', 'past'),
            (r'\btomorrow\b', 'future')
        ]
        
        for pattern, time_type in relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_expressions.append({
                    "text": match.group(),
                    "type": time_type,
                    "start": match.start(),
                    "end": match.end()
                })
        
        return temporal_expressions
    
    def extract_numeric_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric expressions with their context.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: List of numeric expressions with metadata
        """
        numeric_expressions = []
        
        # Number patterns
        number_patterns = [
            r'\b\d+\b',  # Simple integers
            r'\b\d+\.\d+\b',  # Decimals
            r'\b\d+%',  # Percentages
            r'\b\$?\d+(?:,\d{3})*(?:\.\d{2})?\b'  # Currency
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numeric_expressions.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": self._get_numeric_context(text, match.start(), match.end())
                })
        
        return numeric_expressions
    
    def _get_numeric_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Get context around a numeric expression."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()


# Global instance for convenience
_extractor_instance = None


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract entities from text using the global extractor instance.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, Any]: Extracted entities
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance.extract_entities(text)


if __name__ == "__main__":
    # Test the entity extractor
    extractor = EntityExtractor()
    
    test_queries = [
        "Show me sales in Mumbai for last month",
        "Compare complaint volumes between 2024 and 2025",
        "What's the average response time for support tickets in Q3 2024?",
        "Update customer status to active for all customers with pending orders",
        "Show me the top 10 products by revenue in December 2024",
        "Predict sales for next quarter based on current trends"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        entities = extractor.extract_entities(query)
        
        for entity_type, values in entities.items():
            if values:
                print(f"  {entity_type}: {values}")
        
        # Test temporal expressions
        temporal = extractor.extract_temporal_expressions(query)
        if temporal:
            print(f"  temporal: {temporal}")
        
        # Test numeric expressions
        numeric = extractor.extract_numeric_expressions(query)
        if numeric:
            print(f"  numeric: {numeric}")
        
        print("-" * 50)
