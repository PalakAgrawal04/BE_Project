# llm_checker.py
"""
Gemini LLM wrapper for linguistic and logical checks.
This module contains safe helper functions to call Gemini and parse JSON outputs.
Adjust the exact Gemini client call if your google-generativeai version uses different method names.
"""

import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Import the client. Depending on your installed package version the import path varies.
# Example: 'import google.generativeai as genai'
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # We'll guard usage below

PROMPTS_PATH = os.getenv("VALIDATION_PROMPTS_PATH",
                         "backend/services/validation_agent/config/validation_prompts.json")


def _load_prompts() -> Dict[str, Any]:
    import json
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from the model text output robustly.
    Removes code fences and finds {...}.
    """
    # remove markdown/code fences
    text = re.sub(r"```[\s\S]*?```", "", text)
    # find first curly-brace JSON
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError("No JSON object found in model response.")
    json_text = text[first_brace:last_brace+1]
    return json.loads(json_text)


def _gemini_generate(prompt: str, model: str = "gemini-1.5-flash", temperature: float = 0.0) -> str:
    """
    Call the Gemini API. Adjust call to your client version if needed.
    This function returns the raw textual output.
    """
    if genai is None:
        raise RuntimeError("google.generativeai package not installed or failed to import.")

    # Configure once
    genai.configure(api_key=GEMINI_API_KEY)

    # Example call: Some versions use genai.generate_text, some use genai.models.generate
    # Try both patterns
    try:
        # Preferred if available
        response = genai.generate_text(model=model, prompt=prompt, temperature=temperature)
        if hasattr(response, "text"):
            return response.text
        # If the return is raw string:
        return str(response)
    except Exception:
        pass

    try:
        # Alternative API surface (older/newer versions differ)
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt)  # returns an object; adapt as needed
        # Try to fetch a textual result
        if hasattr(resp, "candidates"):
            # find first candidate text
            return resp.candidates[0].content[0].text
        # Otherwise fallback to stringify
        return str(resp)
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}")


def check_linguistic_quality(text: str) -> Dict[str, Any]:
    """
    Return a dict containing is_coherent, issues, suggested_rewrite.
    Uses the linguistic prompt and a deterministic Gemini call (temperature=0).
    """
    prompts = _load_prompts()
    prompt_template = prompts["linguistic_prompt"]
    # Build prompt with examples appended to help grounding
    # Append a small few-shot with input to reduce hallucination
    examples = prompts.get("few_shot_examples", [])
    example_block = ""
    for ex in examples[:2]:
        example_block += f"\n\nEXAMPLE INPUT: \"{ex['input']}\"\nEXPECTED: {json.dumps(ex['linguistic_response'])}"
    full_prompt = f"{prompt_template}\n\nINPUT: \"{text}\"{example_block}\n\nReturn JSON only."

    raw = _gemini_generate(full_prompt, model="gemini-1.5-flash", temperature=0.0)
    parsed = _parse_json_from_text(raw)
    return parsed


def check_logical_validity(text: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Return a dict containing is_valid, reason, issues, suggested_rewrite, final_decision.
    additional_context can include workspace_catalog or retrieved examples to improve accuracy.
    """
    prompts = _load_prompts()
    prompt_template = prompts["logical_prompt"]
    examples = prompts.get("few_shot_examples", [])
    example_block = ""
    for ex in examples[:2]:
        example_block += f"\n\nEXAMPLE INPUT: \"{ex['input']}\"\nEXPECTED: {json.dumps(ex['logical_response'])}"
    context_block = ""
    if additional_context:
        context_block = "\n\nCONTEXT:\n" + json.dumps(additional_context)

    full_prompt = f"{prompt_template}\n\nINPUT: \"{text}\"{context_block}{example_block}\n\nReturn JSON only."

    raw = _gemini_generate(full_prompt, model="gemini-1.5-pro", temperature=0.0)
    parsed = _parse_json_from_text(raw)
    return parsed
