"""
Intent and workspace classification module for Intent Agent.

This module provides hybrid classification using keyword-based logic
and vector similarity for intent type and workspace prediction.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Handles intent type and workspace classification using hybrid approaches.
    """
    
    def __init__(self, workspace_catalog_path: Optional[str] = None):
        """
        Initialize the intent classifier.
        
        Args:
            workspace_catalog_path (str, optional): Path to workspace catalog JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.workspace_catalog = []
        self.workspace_catalog_path = workspace_catalog_path
        
        # Load workspace catalog
        self._load_workspace_catalog()
        
        # Initialize intent type patterns
        self._initialize_intent_patterns()
    
    def _load_workspace_catalog(self) -> None:
        """Load workspace catalog from JSON file."""
        if not self.workspace_catalog_path:
            # Use default path
            self.workspace_catalog_path = "backend/services/intent_agent/config/workspace_catalog.json"
        
        try:
            if Path(self.workspace_catalog_path).exists():
                with open(self.workspace_catalog_path, 'r', encoding='utf-8') as f:
                    self.workspace_catalog = json.load(f)
                self.logger.info(f"Loaded workspace catalog with {len(self.workspace_catalog)} workspaces")
            else:
                self.logger.warning(f"Workspace catalog not found at {self.workspace_catalog_path}")
                self._create_default_workspace_catalog()
        except Exception as e:
            self.logger.error(f"Failed to load workspace catalog: {str(e)}")
            self._create_default_workspace_catalog()
    
    def _create_default_workspace_catalog(self) -> None:
        """Create a default workspace catalog if file loading fails."""
        self.workspace_catalog = [
            {
                "id": "sales",
                "name": "Sales Analytics",
                "description": "Sales and transaction data, revenue analysis, customer purchases",
                "keywords": ["sales", "revenue", "profit", "purchase", "transaction", "customer", "order", "product"]
            },
            {
                "id": "support",
                "name": "Customer Support",
                "description": "Customer feedback, complaints, support tickets, satisfaction surveys",
                "keywords": ["support", "feedback", "complaint", "ticket", "issue", "help", "customer service", "satisfaction"]
            },
            {
                "id": "marketing",
                "name": "Marketing Analytics",
                "description": "Campaign performance, lead generation, conversion rates, marketing metrics",
                "keywords": ["marketing", "campaign", "lead", "conversion", "advertisement", "promotion", "email", "social"]
            },
            {
                "id": "hr",
                "name": "Human Resources",
                "description": "Employee data, performance reviews, attendance, recruitment",
                "keywords": ["employee", "hr", "performance", "attendance", "recruitment", "salary", "department", "manager"]
            },
            {
                "id": "finance",
                "name": "Financial Data",
                "description": "Financial reports, budgets, expenses, accounting data",
                "keywords": ["finance", "budget", "expense", "accounting", "cost", "revenue", "financial", "money"]
            },
            {
                "id": "operations",
                "name": "Operations",
                "description": "Operational metrics, inventory, logistics, supply chain",
                "keywords": ["operations", "inventory", "logistics", "supply", "warehouse", "shipping", "delivery", "stock"]
            }
        ]
        self.logger.info("Created default workspace catalog")
    
    def _initialize_intent_patterns(self) -> None:
        """Initialize patterns for intent type classification."""
        self.intent_patterns = {
            "read": [
                r'\b(show|display|get|find|search|list|give|provide|fetch|retrieve)\b',
                r'\bwhat\s+(is|are|was|were)\b',
                r'\bhow\s+(many|much|often)\b',
                r'\bwhere\s+(is|are)\b',
                r'\bwhen\s+(is|are|was|were)\b'
            ],
            "compare": [
                r'\b(compare|comparison|vs|versus|against|difference|between)\b',
                r'\bvs\.?\b',
                r'\bversus\b',
                r'\bcompared\s+to\b',
                r'\brelative\s+to\b'
            ],
            "update": [
                r'\b(update|change|modify|set|edit|alter|revise)\b',
                r'\b(create|add|insert|new)\b',
                r'\b(delete|remove|drop|eliminate)\b',
                r'\b(assign|transfer|move)\b'
            ],
            "summarize": [
                r'\b(summarize|summary|total|sum|aggregate|consolidate)\b',
                r'\b(average|mean|median|count|maximum|minimum|max|min)\b',
                r'\b(group\s+by|grouped|grouping)\b',
                r'\b(overview|recap|roundup)\b'
            ],
            "analyze": [
                r'\b(analyze|analysis|trend|pattern|insight|examine)\b',
                r'\b(correlation|relationship|association)\b',
                r'\b(breakdown|break\s+down|drill\s+down)\b',
                r'\b(deep\s+dive|investigate|explore)\b'
            ],
            "predict": [
                r'\b(predict|forecast|estimate|project|anticipate)\b',
                r'\b(future|upcoming|next|ahead)\b',
                r'\b(prediction|forecasting|projection)\b',
                r'\b(likely|probability|chance)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_intent_patterns = {}
        for intent_type, patterns in self.intent_patterns.items():
            self.compiled_intent_patterns[intent_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def predict_intent_type(self, text: str) -> str:
        """
        Predict intent type from text using keyword-based classification.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted intent type
        """
        if not text or not isinstance(text, str):
            return "read"  # Default intent
        
        self.logger.info(f"Classifying intent type for: {text[:50]}...")
        
        text_lower = text.lower()
        intent_scores = {}
        
        # Score each intent type based on pattern matches
        for intent_type, patterns in self.compiled_intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                score += len(matches)
                
                # Bonus for exact word matches
                if pattern.search(text_lower):
                    score += 1
            
            intent_scores[intent_type] = score
        
        # Find the intent with highest score
        if intent_scores:
            predicted_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[predicted_intent] > 0:
                self.logger.info(f"Predicted intent: {predicted_intent} (score: {intent_scores[predicted_intent]})")
                return predicted_intent
        
        # Default to "read" if no patterns match
        self.logger.info("No clear intent pattern found, defaulting to 'read'")
        return "read"
    
    def predict_workspace(self, text: str, workspace_catalog: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Predict relevant workspaces from text using keyword matching and similarity.
        
        Args:
            text (str): Input text
            workspace_catalog (List[Dict], optional): Custom workspace catalog
            
        Returns:
            List[Dict]: List of workspaces with scores
        """
        if not text or not isinstance(text, str):
            return []
        
        catalog = workspace_catalog or self.workspace_catalog
        if not catalog:
            return []
        
        self.logger.info(f"Classifying workspace for: {text[:50]}...")
        
        text_lower = text.lower()
        workspace_scores = []
        
        for workspace in catalog:
            score = self._calculate_workspace_score(text_lower, workspace)
            if score > 0:
                workspace_scores.append({
                    "id": workspace["id"],
                    "name": workspace.get("name", workspace["id"]),
                    "description": workspace.get("description", ""),
                    "score": score,
                    "confidence": min(score / 10.0, 1.0)  # Normalize to 0-1
                })
        
        # Sort by score (descending)
        workspace_scores.sort(key=lambda x: x["score"], reverse=True)
        
        self.logger.info(f"Found {len(workspace_scores)} relevant workspaces")
        return workspace_scores
    
    def _calculate_workspace_score(self, text_lower: str, workspace: Dict) -> float:
        """
        Calculate relevance score for a workspace based on keyword matching.
        
        Args:
            text_lower (str): Lowercase input text
            workspace (Dict): Workspace configuration
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        
        # Check keywords
        keywords = workspace.get("keywords", [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            if keyword_lower in text_lower:
                score += 2.0
                
                # Bonus for word boundary matches
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                    score += 1.0
            
            # Partial match
            elif keyword_lower in text_lower.replace(' ', ''):
                score += 0.5
        
        # Check description similarity
        description = workspace.get("description", "").lower()
        if description:
            similarity = self._text_similarity(text_lower, description)
            score += similarity * 3.0
        
        # Check workspace ID
        workspace_id = workspace.get("id", "").lower()
        if workspace_id and workspace_id in text_lower:
            score += 1.5
        
        # Check name similarity
        workspace_name = workspace.get("name", "").lower()
        if workspace_name:
            similarity = self._text_similarity(text_lower, workspace_name)
            score += similarity * 2.0
        
        return score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def classify_query_complexity(self, text: str) -> str:
        """
        Classify query complexity based on patterns and structure.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Complexity level ('simple', 'complex', 'multi_intent')
        """
        if not text or not isinstance(text, str):
            return "simple"
        
        text_lower = text.lower()
        complexity_score = 0
        
        # Simple query indicators
        simple_patterns = [
            r'\b(show|get|find|list)\s+\w+\b',
            r'\bwhat\s+(is|are)\b',
            r'\bhow\s+(many|much)\b'
        ]
        
        # Complex query indicators
        complex_patterns = [
            r'\b(compare|analyze|breakdown|drill\s+down)\b',
            r'\b(trend|pattern|correlation|relationship)\b',
            r'\b(group\s+by|aggregate|summarize)\b',
            r'\b(join|union|intersection)\b'
        ]
        
        # Multi-intent indicators
        multi_intent_patterns = [
            r'\b(and|also|additionally|furthermore)\b',
            r'\b(then|next|after|before)\b',
            r'\b(if|when|where|while)\b'
        ]
        
        # Count pattern matches
        for pattern in simple_patterns:
            if re.search(pattern, text_lower):
                complexity_score -= 1
        
        for pattern in complex_patterns:
            if re.search(pattern, text_lower):
                complexity_score += 2
        
        for pattern in multi_intent_patterns:
            if re.search(pattern, text_lower):
                complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 3:
            return "multi_intent"
        elif complexity_score >= 1:
            return "complex"
        else:
            return "simple"
    
    def detect_time_sensitivity(self, text: str) -> str:
        """
        Detect time sensitivity of the query.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Time sensitivity level
        """
        if not text or not isinstance(text, str):
            return "historical"
        
        text_lower = text.lower()
        
        # Immediate indicators
        immediate_patterns = [
            r'\b(now|immediately|urgent|asap|right\s+now)\b',
            r'\b(current|today|this\s+hour|this\s+minute)\b'
        ]
        
        # Near-term indicators
        near_term_patterns = [
            r'\b(this\s+week|this\s+month|recent|latest)\b',
            r'\b(last\s+few\s+days|last\s+few\s+hours)\b'
        ]
        
        # Future indicators
        future_patterns = [
            r'\b(next|upcoming|future|tomorrow|next\s+week|next\s+month)\b',
            r'\b(predict|forecast|project)\b'
        ]
        
        # Check patterns
        for pattern in immediate_patterns:
            if re.search(pattern, text_lower):
                return "immediate"
        
        for pattern in near_term_patterns:
            if re.search(pattern, text_lower):
                return "near_term"
        
        for pattern in future_patterns:
            if re.search(pattern, text_lower):
                return "future"
        
        return "historical"


# Global instance for convenience
_classifier_instance = None


def predict_intent_type(text: str) -> str:
    """
    Predict intent type using the global classifier instance.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Predicted intent type
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.predict_intent_type(text)


def predict_workspace(text: str, workspace_catalog: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
    """
    Predict workspaces using the global classifier instance.
    
    Args:
        text (str): Input text
        workspace_catalog (List[Dict], optional): Custom workspace catalog
        
    Returns:
        List[Dict]: List of workspaces with scores
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.predict_workspace(text, workspace_catalog)


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()
    
    test_queries = [
        "Show me sales in Mumbai for last month",
        "Compare complaint volumes between 2024 and 2025",
        "What's the average response time for support tickets in Q3 2024?",
        "Update customer status to active for all customers with pending orders",
        "Analyze the trend in customer satisfaction over the past year",
        "Predict sales for next quarter based on current trends"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Test intent classification
        intent = classifier.predict_intent_type(query)
        print(f"Intent Type: {intent}")
        
        # Test workspace classification
        workspaces = classifier.predict_workspace(query)
        print(f"Workspaces: {[(w['id'], f"{w['score']:.2f}") for w in workspaces[:3]]}")
        
        # Test complexity classification
        complexity = classifier.classify_query_complexity(query)
        print(f"Complexity: {complexity}")
        
        # Test time sensitivity
        time_sensitivity = classifier.detect_time_sensitivity(query)
        print(f"Time Sensitivity: {time_sensitivity}")
        
        print("-" * 50)
