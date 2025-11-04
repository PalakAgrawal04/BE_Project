# test_validation_agent.py
"""
Unit tests for validation agent. These tests mock the LLM wrapper functions
to avoid real API calls during CI/dev.
"""

import pytest
from unittest.mock import patch
from backend.services.validation_agent.validator import run_validation_agent

# Example tests mocking llm_checker responses
@patch("backend.services.validation_agent.llm_checker.check_linguistic_quality")
@patch("backend.services.validation_agent.llm_checker.check_logical_validity")
def test_needs_clarification(mock_logical, mock_linguistic):
    mock_linguistic.return_value = {
        "is_coherent": True,
        "issues": ["ambiguous time reference"],
        "suggested_rewrite": "Compare March and April sales performance."
    }
    mock_logical.return_value = {
        "is_valid": False,
        "reason": "Ambiguous timeframe",
        "issues": ["ambiguous time reference"],
        "suggested_rewrite": "Compare March and April sales performance.",
        "final_decision": "needs_clarification"
    }

    out = run_validation_agent("March sales in April")
    assert out["is_coherent"] is True
    assert out["is_valid"] is False
    assert out["final_decision"] == "needs_clarification"
    assert "ambiguous time reference" in out["issues"]

@patch("backend.services.validation_agent.llm_checker.check_linguistic_quality")
@patch("backend.services.validation_agent.llm_checker.check_logical_validity")
def test_valid_query(mock_logical, mock_linguistic):
    mock_linguistic.return_value = {
        "is_coherent": True,
        "issues": [],
        "suggested_rewrite": None
    }
    mock_logical.return_value = {
        "is_valid": True,
        "reason": "Explicit action and timeframe",
        "issues": [],
        "suggested_rewrite": None,
        "final_decision": "valid"
    }

    out = run_validation_agent("Show me total revenue for last quarter")
    assert out["is_coherent"] is True
    assert out["is_valid"] is True
    assert out["final_decision"] == "valid"
