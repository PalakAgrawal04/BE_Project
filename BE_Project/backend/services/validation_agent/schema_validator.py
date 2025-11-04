# schema_validator.py
"""
Pydantic models and validation helpers for the Validation Agent outputs.
"""

from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any


class LinguisticValidation(BaseModel):
    is_coherent: bool
    issues: List[str]
    suggested_rewrite: Optional[str] = None


class LogicalValidation(BaseModel):
    is_valid: bool
    reason: Optional[str] = None
    issues: List[str] = []
    suggested_rewrite: Optional[str] = None
    final_decision: str  # 'valid' | 'needs_clarification' | 'invalid'


class CombinedValidation(BaseModel):
    is_coherent: bool
    is_valid: bool
    issues: List[str]
    suggested_rewrite: Optional[str] = None
    final_decision: str


def validate_linguistic(data: Dict[str, Any]) -> LinguisticValidation:
    """Validate the linguistic JSON returned by the LLM."""
    try:
        return LinguisticValidation(**data)
    except ValidationError as e:
        raise ValueError(f"Linguistic validation schema error: {e}")


def validate_logical(data: Dict[str, Any]) -> LogicalValidation:
    """Validate the logical JSON returned by the LLM."""
    try:
        return LogicalValidation(**data)
    except ValidationError as e:
        raise ValueError(f"Logical validation schema error: {e}")


def build_combined(linguistic: LinguisticValidation, logical: LogicalValidation) -> CombinedValidation:
    """Combine linguistic and logical outputs into a single CombinedValidation object."""
    # Merge issues (deduplicate)
    merged_issues = list(dict.fromkeys((linguistic.issues or []) + (logical.issues or [])))
    suggested = logical.suggested_rewrite or linguistic.suggested_rewrite
    return CombinedValidation(
        is_coherent=linguistic.is_coherent,
        is_valid=logical.is_valid,
        issues=merged_issues,
        suggested_rewrite=suggested,
        final_decision=logical.final_decision
    )
