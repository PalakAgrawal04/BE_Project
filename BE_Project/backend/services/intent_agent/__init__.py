"""
Intent Agent Module for IntelliQuery

This module provides comprehensive intent understanding capabilities including:
- Text preprocessing and normalization
- Entity extraction using spaCy
- Embedding generation and similarity retrieval
- Intent classification and workspace mapping
- LLM-powered intent synthesis using Gemini
- Schema validation and error handling

Main entry point: run_intent_agent()
"""

from .agent_orchestrator import run_intent_agent

__version__ = "1.0.0"
__all__ = ["run_intent_agent"]
