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
        # Date patterns - IMPORTANT: Include month names without years
        self.date_patterns = [
            r'\b(last|next|this)\s+(month|year|quarter|week|day)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',  # Month with year
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Month name only
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}\b',  # Short month with year
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # Short month only
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
            
            # Extract advanced date patterns (ranges, comparisons, quarters)
            advanced_date_entities = self._extract_advanced_date_patterns(text)
            for key, value in advanced_date_entities.items():
                if key in entities:
                    if isinstance(entities[key], list) and isinstance(value, list):
                        entities[key].extend(value)
                    elif isinstance(entities[key], dict) and isinstance(value, dict):
                        entities[key].update(value)
                    else:
                        entities["custom"][key] = value
                else:
                    entities["custom"][key] = value
            
            # Extract aggregation keywords
            aggregation_info = self._extract_aggregation_keywords(text)
            if aggregation_info:
                entities["custom"]["aggregation"] = aggregation_info
            
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
        
        # Extract dates using patterns
        for pattern in self.compiled_date_patterns:
            matches = pattern.finditer(text, re.IGNORECASE)
            for match in matches:
                # Get the full match text
                match_text = match.group(0)
                # If there are groups, prefer them (for patterns like "last month")
                if match.groups():
                    groups = [g for g in match.groups() if g]
                    if groups:
                        match_text = " ".join(groups)
                
                normalized_date = self._normalize_date(match_text)
                if normalized_date and normalized_date not in entities["dates"]:
                    entities["dates"].append(normalized_date)
        
        # Also explicitly check for month names in the text (case-insensitive)
        month_names = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        
        text_lower = text.lower()
        for month in month_names:
            # Look for month name as a whole word
            pattern = r'\b' + re.escape(month) + r'\b'
            if re.search(pattern, text_lower):
                # Check if it's already in dates list
                normalized = self._normalize_date(month)
                if normalized and normalized not in entities["dates"]:
                    entities["dates"].append(normalized)
        
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
        Normalize date expressions, preserving month-only information.
        
        Args:
            date_text (str): Raw date text
            
        Returns:
            Optional[str]: Normalized date string preserving original format when appropriate
        """
        if not date_text:
            return None
            
        date_lower = date_text.lower().strip()
        
        # Month name mapping
        month_map = {
            'january': 'January', 'jan': 'January',
            'february': 'February', 'feb': 'February',
            'march': 'March', 'mar': 'March',
            'april': 'April', 'apr': 'April',
            'may': 'May',
            'june': 'June', 'jun': 'June',
            'july': 'July', 'jul': 'July',
            'august': 'August', 'aug': 'August',
            'september': 'September', 'sep': 'September', 'sept': 'September',
            'october': 'October', 'oct': 'October',
            'november': 'November', 'nov': 'November',
            'december': 'December', 'dec': 'December'
        }
        
        # Check if it's just a month name (no year)
        for month_key, month_name in month_map.items():
            if date_lower == month_key or date_lower == month_key + ' ':
                # Return the month name as-is (don't convert to ISO date)
                return month_name
        
        # Check if it's month + year
        import re
        month_year_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b', date_lower)
        if month_year_match:
            month_part = month_year_match.group(1)
            year_part = month_year_match.group(2)
            # Return as "Month YYYY" format
            full_month = month_map.get(month_part, month_part.capitalize())
            return f"{full_month} {year_part}"
        
        # For other date formats, try dateparser
        try:
            parsed_date = dateparser.parse(date_text)
            if parsed_date:
                # Format as ISO date string
                return parsed_date.strftime("%Y-%m-%d")
            return date_text  # Return original if parsing fails
        except Exception as e:
            self.logger.warning(f"Failed to normalize date '{date_text}': {str(e)}")
            return date_text  # Return original text
    
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
    
    def _extract_advanced_date_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract advanced date patterns: month ranges, date comparisons, quarters.
        
        Returns:
            Dict containing date_range, comparison, compare_date, quarter
        """
        result = {}
        text_lower = text.lower()
        
        # Month name to number mapping
        month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # 1. Extract month ranges: "between X and Y" or "from X to Y"
        month_range_pattern = r'\b(between|from)\s+(\w+)\s+(and|to)\s+(\w+)\b'
        range_match = re.search(month_range_pattern, text_lower)
        if range_match:
            month1_str = range_match.group(2).lower()
            month2_str = range_match.group(4).lower()
            
            month1_num = month_map.get(month1_str)
            month2_num = month_map.get(month2_str)
            
            if month1_num and month2_num:
                result["date_range"] = {
                    "start_month": min(month1_num, month2_num),
                    "end_month": max(month1_num, month2_num)
                }
        
        # 2. Extract date comparisons: "after", "before", "greater than", "less than"
        comparison_keywords = {
            'after': '>',
            'before': '<',
            'greater than': '>',
            'less than': '<',
            'more than': '>',
            'earlier than': '<',
            'later than': '>'
        }
        
        for keyword, operator in comparison_keywords.items():
            pattern = rf'\b{re.escape(keyword)}\s+([\w\s]+?)(?:\s|$)'
            match = re.search(pattern, text_lower)
            if match:
                date_text = match.group(1).strip()
                # Try to parse the date
                try:
                    parsed_date = dateparser.parse(date_text)
                    if parsed_date:
                        result["comparison"] = operator
                        result["compare_date"] = parsed_date.strftime("%Y-%m-%d")
                        break  # Use first match
                except:
                    pass
        
        # Also check for explicit > or < operators with dates
        operator_pattern = r'([<>])\s*([\w\s]+)'
        op_match = re.search(operator_pattern, text_lower)
        if op_match and not result.get("comparison"):
            operator = op_match.group(1)
            date_text = op_match.group(2).strip()
            try:
                parsed_date = dateparser.parse(date_text)
                if parsed_date:
                    result["comparison"] = operator
                    result["compare_date"] = parsed_date.strftime("%Y-%m-%d")
            except:
                pass
        
        # 3. Extract quarters: Q1, Q2, Q3, Q4
        quarter_pattern = r'\bq([1-4])\s+(?:of\s+)?(\d{4})\b'
        quarter_match = re.search(quarter_pattern, text_lower)
        if quarter_match:
            quarter_num = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            
            # Map quarter to months
            quarter_months = {
                1: [1, 2, 3],
                2: [4, 5, 6],
                3: [7, 8, 9],
                4: [10, 11, 12]
            }
            
            result["quarter"] = {
                "months": quarter_months[quarter_num],
                "year": year
            }
        
        return result
    
    def _extract_aggregation_keywords(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract aggregation keywords: SUM, COUNT, AVG, etc.
        
        Returns:
            Dict with aggregation type and column name, or None
        """
        text_lower = text.lower()
        
        # Aggregation keywords
        aggregation_patterns = {
            'sum': r'\b(sum|total|sum of|total of)\b',
            'count': r'\b(count|number of|how many)\b',
            'avg': r'\b(average|avg|mean)\b',
            'max': r'\b(max|maximum|highest)\b',
            'min': r'\b(min|minimum|lowest)\b'
        }
        
        for agg_type, pattern in aggregation_patterns.items():
            if re.search(pattern, text_lower):
                # Try to detect column name
                column_name = "amount"  # Default
                
                # Check for common column names in context
                if 'sales' in text_lower or 'revenue' in text_lower or 'total sales' in text_lower:
                    column_name = "amount"
                elif 'order' in text_lower:
                    column_name = "total_amount"
                elif 'customer' in text_lower:
                    column_name = "customer_count"
                
                return {
                    "type": agg_type.upper(),
                    "column": column_name
                }
        
        return None


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
