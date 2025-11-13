# llm_checker.py
"""
Gemini LLM wrapper for linguistic and logical checks.
This module contains safe helper functions to call Gemini and parse JSON outputs.
Adjust the exact Gemini client call if your google-generativeai version uses different method names.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .logging_config import setup_logging

# Set up logging
setup_logging()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Import the client. Depending on your installed package version the import path varies.
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # We'll guard usage below

PROMPTS_PATH = os.getenv(
    "VALIDATION_PROMPTS_PATH",
    "backend/services/validation_agent/config/validation_prompts.json"
)


def _load_prompts() -> Dict[str, Any]:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from the model text output robustly.
    """
    logger = logging.getLogger(__name__)
    logger.info("Attempting to parse model response as JSON")
    logger.debug(f"Raw text: {text}")

    text = text.strip()

    # Try to parse entire text first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse entire text as JSON: {e}")

    # Try to extract JSON from code blocks
    code_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"`([\s\S]*?)`"
    ]

    for pattern in code_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                candidate = match.strip()
                logger.debug(f"Trying JSON from code block: {candidate}")
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse code block as JSON: {e}")

    # Fallback: parse plain JSON objects within text
    text_no_code = re.sub(r"```[\s\S]*?```", "", text)
    text_no_code = re.sub(r"`[^`]*`", "", text_no_code)

    depth = 0
    start = -1
    json_candidates = []

    for i, char in enumerate(text_no_code):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                json_candidates.append(text_no_code[start:i + 1])

    for candidate in json_candidates:
        try:
            logger.debug(f"Trying JSON candidate: {candidate}")
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed candidate: {e}")

    logger.error(f"No valid JSON found in response: {text}")
    raise ValueError("No JSON object found in model response.")


def _gemini_generate(
    prompt: str,
    model: str = "models/gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40
) -> str:
    """
    Call the Gemini API using google-generativeai >= 0.7 style.
    Returns the raw text output.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting Gemini generation with model: {model}")

    if genai is None:
        raise RuntimeError("google.generativeai package not installed or failed to import.")
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    genai.configure(api_key=GEMINI_API_KEY)

    safety_settings = {
        "HARASSMENT": "block_none",
        "HATE_SPEECH": "block_none",
        "SEXUALLY_EXPLICIT": "block_none",
        # "DANGEROUS_CONTENT": "block_none"
    }

    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            contents=prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k
            },
            safety_settings=safety_settings
        )

        if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
            raise RuntimeError(f"Content blocked: {response.prompt_feedback.block_reason}")

        if not response.candidates:
            raise RuntimeError("No candidates in response")

        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text
        else:
            raise RuntimeError("No valid content in response")

        logger.debug(f"Generated text: {text}")
        return text.strip()

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}", exc_info=True)
        raise RuntimeError(f"Gemini call failed: {e}")


def check_linguistic_quality(text: str) -> Dict[str, Any]:
    """
    Return a dict containing is_coherent, issues, suggested_rewrite.
    Matches LinguisticValidation Pydantic model.
    """
    prompt = f"""INSTRUCTIONS: Return ONLY a JSON object. No other text.

INPUT: {json.dumps(text)}

JSON_SCHEMA: {{
    "is_coherent": boolean,  // Is the query well-formed and clear?
    "issues": string[],      // List of any linguistic issues found
    "suggested_rewrite": string | null  // Improved version if issues found
}}

EXAMPLE: {{
    "is_coherent": true,
    "issues": [],
    "suggested_rewrite": null
}}

OUTPUT_FORMAT: Strict JSON only. No markdown. No explanations."""

    raw = _gemini_generate(prompt, temperature=0.0)
    return _parse_json_from_text(raw)


def check_logical_validity(text: str, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a dict containing is_valid, reason, issues, suggested_rewrite, final_decision.
    Matches LogicalValidation Pydantic model.
    """
    context_section = ""
    if additional_context:
        context_section = f"\nCONTEXT: {json.dumps(additional_context)}"

    prompt = f"""INSTRUCTIONS: Return ONLY a JSON object. No other text.

INPUT: {json.dumps(text)}{context_section}

JSON_SCHEMA: {{
    "is_valid": boolean,  // Is the query valid?
    "reason": string | null,  // Why valid/invalid?
    "issues": string[],  // List of problems found
    "suggested_rewrite": string | null,  // Fixed version if needed
    "final_decision": string  // One of: 'valid', 'needs_clarification', 'invalid'
}}

EXAMPLE: {{
    "is_valid": true,
    "reason": "Query is well-formed and complete",
    "issues": [],
    "suggested_rewrite": null,
    "final_decision": "valid"
}}

OUTPUT_FORMAT: Strict JSON only. No markdown. No explanations."""

    raw = _gemini_generate(prompt, temperature=0.0)
    return _parse_json_from_text(raw)


def list_available_models():
    """
    List all available Gemini models and their supported methods.
    """
    if genai is None:
        raise RuntimeError("google.generativeai package not installed or failed to import.")
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    genai.configure(api_key=GEMINI_API_KEY)

    try:
        models = genai.list_models()
        for model in models:
            print(f"Model: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Generation Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 80)
        return models
    except Exception as e:
        raise RuntimeError(f"Failed to list models: {e}")
