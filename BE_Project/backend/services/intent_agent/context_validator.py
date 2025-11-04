"""
Context validation module for Intent Agent.

This module validates user queries for logical consistency and sensibleness
before proceeding with intent extraction. It detects redundant, contradictory,
or nonsensical prompts.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextValidator:
    """
    Validates user queries for logical consistency and sensibleness.
    """
    
    def __init__(self):
        """Initialize the context validator."""
        self.logger = logging.getLogger(__name__)
        
        # Month names (full and abbreviated)
        self.months = {
            'january', 'jan', 'february', 'feb', 'march', 'mar',
            'april', 'apr', 'may', 'june', 'jun', 'july', 'jul',
            'august', 'aug', 'september', 'sep', 'sept', 'october', 'oct',
            'november', 'nov', 'december', 'dec'
        }
        
        # Temporal keywords
        self.temporal_keywords = {
            'last', 'next', 'this', 'previous', 'upcoming', 'current',
            'today', 'yesterday', 'tomorrow', 'month', 'year', 'quarter',
            'week', 'day', 'hour', 'minute'
        }
        
        # Common contradictory patterns
        self.contradictory_patterns = [
            (r'\b(\w+)\s+(sales|data|information|results|reports)\s+in\s+\1\b', 
             'redundant_temporal'),
            (r'\b(\w+)\s+(show|get|find|display)\s+\1\b',
             'redundant_action'),
            (r'\b(both|all|every)\s+(and|or|with)\s+(none|nothing)\b',
             'contradictory_scope'),
            (r'\b(show|get|find)\s+(me|us)\s+(show|get|find)\s+(me|us)\b',
             'redundant_request')
        ]
        
        # Compile patterns
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for validation."""
        self.compiled_contradictory_patterns = [
            (re.compile(pattern, re.IGNORECASE), issue_type)
            for pattern, issue_type in self.contradictory_patterns
        ]
    
    def validate_context(self, user_query: str) -> Dict[str, Any]:
        """
        Validate the context of a user query for logical consistency.
        
        Args:
            user_query (str): User query text
            
        Returns:
            Dict[str, Any]: Validation result with 'is_valid', 'issues', and 'message'
        """
        if not user_query or not isinstance(user_query, str):
            return {
                'is_valid': False,
                'issues': ['invalid_input'],
                'message': 'Invalid input: query must be a non-empty string',
                'confidence': 1.0
            }
        
        self.logger.info(f"Validating context for: {user_query[:100]}...")
        
        issues = []
        
        # Check 1: Redundant temporal references
        temporal_issues = self._check_redundant_temporal(user_query)
        issues.extend(temporal_issues)
        
        # Check 2: Duplicate words/phrases
        duplicate_issues = self._check_duplicate_phrases(user_query)
        issues.extend(duplicate_issues)
        
        # Check 3: Contradictory patterns
        contradictory_issues = self._check_contradictory_patterns(user_query)
        issues.extend(contradictory_issues)
        
        # Check 4: Month repetition in context
        month_issues = self._check_month_repetition(user_query)
        issues.extend(month_issues)
        
        # Check 5: Nonsensical combinations
        nonsense_issues = self._check_nonsensical_combinations(user_query)
        issues.extend(nonsense_issues)
        
        # Determine validity
        is_valid = len(issues) == 0
        
        # Build message
        if is_valid:
            message = 'Query context is valid'
        else:
            issue_descriptions = {
                'redundant_temporal': 'redundant or contradictory temporal references',
                'redundant_action': 'redundant action words',
                'duplicate_phrases': 'duplicate phrases detected',
                'month_repetition': 'month name appears redundantly',
                'contradictory_scope': 'contradictory scope indicators',
                'redundant_request': 'redundant request phrases'
            }
            descriptions = [issue_descriptions.get(issue, issue) for issue in issues]
            message = f"Query contains {', '.join(descriptions)}. Please clarify your request."
        
        # Calculate confidence (lower confidence if issues found)
        confidence = max(0.0, 1.0 - (len(issues) * 0.3))
        
        result = {
            'is_valid': is_valid,
            'issues': issues,
            'message': message,
            'confidence': confidence,
            'original_query': user_query
        }
        
        if not is_valid:
            self.logger.warning(f"Context validation failed: {message}")
        
        return result
    
    def _check_redundant_temporal(self, query: str) -> List[str]:
        """Check for redundant temporal references."""
        issues = []
        query_lower = query.lower()
        
        # Pattern: "march sales in march", "last month sales last month"
        temporal_words = list(self.months) + list(self.temporal_keywords)
        
        for temporal_word in temporal_words:
            # Count occurrences
            pattern = r'\b' + re.escape(temporal_word) + r'\b'
            matches = re.findall(pattern, query_lower)
            
            if len(matches) >= 2:
                # Check if they appear in redundant contexts
                # Look for patterns like "WORD ... in WORD" or "WORD ... WORD"
                redundant_patterns = [
                    rf'\b{re.escape(temporal_word)}\s+\w+\s+(in|for|during|of)\s+{re.escape(temporal_word)}\b',
                    rf'\b{re.escape(temporal_word)}\s+{re.escape(temporal_word)}\b'
                ]
                
                for pattern in redundant_patterns:
                    if re.search(pattern, query_lower):
                        issues.append('redundant_temporal')
                        break
        
        return issues
    
    def _check_duplicate_phrases(self, query: str) -> List[str]:
        """Check for duplicate phrases that indicate redundancy."""
        issues = []
        query_lower = query.lower()
        
        # Tokenize and check for repeated sequences
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Check for repeated 2-3 word phrases
        for phrase_length in [2, 3]:
            phrases = []
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i+phrase_length])
                phrases.append(phrase)
            
            # Count phrase occurrences
            phrase_counts = Counter(phrases)
            
            # Check if any phrase appears multiple times in suspicious contexts
            for phrase, count in phrase_counts.items():
                if count >= 2 and len(phrase.split()) >= 2:
                    # Check if it's not just a common word like "the the"
                    if not all(word in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for'] 
                              for word in phrase.split()):
                        issues.append('duplicate_phrases')
                        break
        
        return issues
    
    def _check_contradictory_patterns(self, query: str) -> List[str]:
        """Check for contradictory patterns."""
        issues = []
        query_lower = query.lower()
        
        for pattern, issue_type in self.compiled_contradictory_patterns:
            if pattern.search(query_lower):
                issues.append(issue_type)
        
        return issues
    
    def _check_month_repetition(self, query: str) -> List[str]:
        """Check specifically for month name repetition like 'march sales in march'."""
        issues = []
        query_lower = query.lower()
        
        # Check each month
        for month in self.months:
            # Pattern: "month ... in month" or "month ... month"
            patterns = [
                rf'\b{re.escape(month)}\s+[^.]*\s+(in|for|during|of)\s+{re.escape(month)}\b',
                rf'\b{re.escape(month)}\s+[^.]*\s+{re.escape(month)}\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    issues.append('month_repetition')
                    return issues  # Return immediately on first match
        
        return issues
    
    def _check_nonsensical_combinations(self, query: str) -> List[str]:
        """Check for nonsensical word combinations."""
        issues = []
        query_lower = query.lower()
        
        # Check for patterns that don't make sense
        nonsensical_patterns = [
            (r'\b(show|get|find)\s+(show|get|find)\b', 'redundant_request'),
            (r'\b(the|a|an)\s+(the|a|an)\b', 'redundant_articles'),
            (r'\b(yesterday|today|tomorrow)\s+(yesterday|today|tomorrow)\b', 'contradictory_time')
        ]
        
        for pattern, issue_type in nonsensical_patterns:
            if re.search(pattern, query_lower):
                issues.append(issue_type)
        
        return issues
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of validation results.
        
        Args:
            validation_result (Dict): Result from validate_context()
            
        Returns:
            str: Human-readable summary
        """
        if validation_result['is_valid']:
            return "✅ Query validation passed: The query makes logical sense."
        
        issues = validation_result.get('issues', [])
        message = validation_result.get('message', 'Unknown validation error')
        
        summary = f"⚠️ Query validation failed: {message}\n"
        summary += f"   Issues detected: {len(issues)}\n"
        summary += f"   Confidence: {validation_result.get('confidence', 0.0):.2f}\n"
        summary += f"   Please rephrase your query to remove redundancies or contradictions."
        
        return summary


# Global instance for convenience
_validator_instance = None


def validate_query_context(user_query: str) -> Dict[str, Any]:
    """
    Validate the context of a user query.
    
    Args:
        user_query (str): User query text
        
    Returns:
        Dict[str, Any]: Validation result
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ContextValidator()
    return _validator_instance.validate_context(user_query)


if __name__ == "__main__":
    # Test the context validator
    validator = ContextValidator()
    
    test_queries = [
        "Show me sales in Mumbai for last month",  # Valid
        "march sales in march",  # Invalid - redundant month
        "last month sales last month",  # Invalid - redundant temporal
        "Show me show me sales",  # Invalid - redundant action
        "Compare complaint volumes between 2024 and 2025",  # Valid
        "What's the average response time for support tickets?",  # Valid
        "march data in march 2024",  # Invalid - redundant month
        "get me get sales data",  # Invalid - redundant request
        "sales in march for march",  # Invalid - redundant month
        "today today sales",  # Invalid - contradictory time
    ]
    
    print("Testing Context Validator:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)
        
        result = validator.validate_context(query)
        
        print(f"Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Message: {result['message']}")
        
        if result['issues']:
            print(f"Issues: {', '.join(result['issues'])}")
        
        summary = validator.get_validation_summary(result)
        print(f"Summary: {summary}")
    
    print("\n" + "=" * 60)
    print("Context validation testing completed!")

