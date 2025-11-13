"""
Test script to list available Gemini models
"""

from llm_checker import list_available_models

if __name__ == "__main__":
    try:
        print("Fetching available Gemini models...")
        models = list_available_models()
        print("\nTotal models found:", len(models))
    except Exception as e:
        print(f"Error: {e}")