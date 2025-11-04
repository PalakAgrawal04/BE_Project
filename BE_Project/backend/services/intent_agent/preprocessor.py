"""
Text preprocessing module for Intent Agent.

This module handles text normalization, language detection, and cleaning
of user queries before they are processed by the intent analysis pipeline.
"""

import re
import logging
from typing import Optional
from langdetect import detect, LangDetectException
from langdetect.lang_detect_exception import LangDetectException as LDE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Handles text preprocessing operations including normalization,
    language detection, and cleaning.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.logger = logging.getLogger(__name__)
        
    def preprocess_text(self, raw_text: str) -> str:
        """
        Normalize and clean the user query for downstream processing.
        
        Args:
            raw_text (str): Raw user input text
            
        Returns:
            str: Cleaned and normalized text
            
        Raises:
            ValueError: If input text is empty or invalid
        """
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError("Input text must be a non-empty string")
            
        self.logger.info(f"Preprocessing text: {raw_text[:50]}...")
        
        try:
            # Step 1: Basic cleaning
            cleaned_text = self._basic_cleaning(raw_text)
            
            # Step 2: Language detection and handling
            processed_text = self._handle_language(cleaned_text)
            
            # Step 3: Final normalization
            final_text = self._final_normalization(processed_text)
            
            self.logger.info(f"Preprocessing complete. Original length: {len(raw_text)}, Final length: {len(final_text)}")
            
            return final_text
            
        except Exception as e:
            self.logger.error(f"Error during text preprocessing: {str(e)}")
            # Fallback to basic cleaning if other steps fail
            return self._basic_cleaning(raw_text)
    
    def _basic_cleaning(self, text: str) -> str:
        """
        Perform basic text cleaning operations.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]]+', '', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([\.\,\!\?\:\;])\1+', r'\1', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text.strip()
    
    def _handle_language(self, text: str) -> str:
        """
        Detect language and handle non-English text if needed.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Processed text
        """
        try:
            # Detect language
            detected_lang = detect(text)
            self.logger.info(f"Detected language: {detected_lang}")
            
            # For now, we'll keep the original text regardless of language
            # In a production system, you might want to translate non-English text
            if detected_lang != 'en':
                self.logger.warning(f"Non-English text detected: {detected_lang}. Consider translation.")
                
            return text
            
        except (LangDetectException, LDE) as e:
            self.logger.warning(f"Language detection failed: {str(e)}. Using original text.")
            return text
        except Exception as e:
            self.logger.error(f"Unexpected error in language handling: {str(e)}")
            return text
    
    def _final_normalization(self, text: str) -> str:
        """
        Apply final normalization steps.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Final normalized text
        """
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([\.\,\!\?\:\;])', r'\1', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([\.\,\!\?\:\;])([^\s])', r'\1 \2', text)
        
        # Final whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 3) -> list[str]:
        """
        Extract potential keywords from the text.
        
        Args:
            text (str): Input text
            min_length (int): Minimum length for keywords
            
        Returns:
            list[str]: List of extracted keywords
        """
        # Remove common stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def detect_query_type(self, text: str) -> str:
        """
        Detect the basic type of query based on patterns.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Query type ('question', 'command', 'statement')
        """
        text_lower = text.lower().strip()
        
        # Question patterns
        question_patterns = [
            r'^(what|how|when|where|why|who|which|can|could|would|should|may|might)',
            r'\?$'
        ]
        
        # Command patterns
        command_patterns = [
            r'^(show|display|get|find|search|list|give|provide)',
            r'^(update|change|modify|set|create|add|delete|remove)',
            r'^(compare|analyze|calculate|compute|generate)'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return 'question'
        
        for pattern in command_patterns:
            if re.search(pattern, text_lower):
                return 'command'
        
        return 'statement'


def preprocess_text(raw_text: str) -> str:
    """
    Convenience function for text preprocessing.
    
    Args:
        raw_text (str): Raw user input text
        
    Returns:
        str: Cleaned and normalized text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(raw_text)


if __name__ == "__main__":
    # Test the preprocessor
    test_texts = [
        "Show me sales in Mumbai for last month",
        "What's the average response time for support tickets?",
        "Compare complaint volumes between 2024 and 2025",
        "Update customer status to active",
        "  Multiple    spaces   and   special   chars!!!  ",
        ""
    ]
    
    preprocessor = TextPreprocessor()
    
    for text in test_texts:
        try:
            result = preprocessor.preprocess_text(text)
            print(f"Original: '{text}'")
            print(f"Processed: '{result}'")
            print(f"Keywords: {preprocessor.extract_keywords(result)}")
            print(f"Query Type: {preprocessor.detect_query_type(result)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
