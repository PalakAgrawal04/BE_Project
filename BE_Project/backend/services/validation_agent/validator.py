# validator.py
"""
Orchestration for the Validation Agent.
This module runs linguistic and logical checks sequentially and builds a combined JSON
result for downstream use (Intent Agent).
"""

import logging
from typing import Dict, Any, Optional
from .llm_checker import check_linguistic_quality, check_logical_validity
from .schema_validator import validate_linguistic, validate_logical, build_combined

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_validation_agent(input_text: str, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the validation pipeline:
      1. Linguistic check (grammar/coherence)
      2. If coherent, logical validity check
      3. Combine results and return structured JSON

    Returns:
      {
        "is_coherent": bool,
        "is_valid": bool,
        "issues": [...],
        "suggested_rewrite": "...",
        "final_decision": "valid" | "needs_clarification" | "invalid"
      }
    """
    if not input_text or not input_text.strip():
        logger.debug("Empty input received in validation agent.")
        return {
            "is_coherent": False,
            "is_valid": False,
            "issues": ["empty_input"],
            "suggested_rewrite": None,
            "final_decision": "invalid"
        }

    # 1. Linguistic check
    try:
        linguistic_raw = check_linguistic_quality(input_text)
        linguistic = validate_linguistic(linguistic_raw)
    except Exception as e:
        logger.exception("Linguistic check or validation failed.")
        # Fallback: consider it incoherent
        return {
            "is_coherent": False,
            "is_valid": False,
            "issues": [f"linguistic_check_failed: {str(e)}"],
            "suggested_rewrite": None,
            "final_decision": "invalid"
        }

    # If not coherent, return immediately with suggested rewrite
    if not linguistic.is_coherent:
        return {
            "is_coherent": False,
            "is_valid": False,
            "issues": linguistic.issues,
            "suggested_rewrite": linguistic.suggested_rewrite,
            "final_decision": "needs_clarification"
        }

    # 2. Logical check (include optional context to help)
    try:
        logical_raw = check_logical_validity(input_text, additional_context=additional_context or {})
        logical = validate_logical(logical_raw)
    except Exception as e:
        logger.exception("Logical check or validation failed.")
        return {
            "is_coherent": True,
            "is_valid": False,
            "issues": [f"logical_check_failed: {str(e)}"],
            "suggested_rewrite": None,
            "final_decision": "invalid"
        }

    # 3. Combine and return
    combined = build_combined(linguistic, logical)
    return combined.dict()
