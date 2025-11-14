#!/usr/bin/env python3
"""
Quick test script for IntelliQuery Intent Agent.

Simple script to quickly test SQL generation with a single query.
Usage: python quick_test.py "your query here"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get the script directory and try to find backend
script_dir = Path(__file__).parent.absolute()

# Try multiple possible backend locations
possible_backend_paths = [
    script_dir / "BE_Project" / "backend",  # If script is in BE_Project/
    script_dir / "backend",  # If script is already in BE_Project/BE_Project/
    script_dir.parent / "BE_Project" / "backend",  # If script is nested deeper
    script_dir.parent.parent / "BE_Project" / "backend",  # Another level up
]

backend_dir = None
for path in possible_backend_paths:
    if path.exists() and (path / "services").exists():
        backend_dir = path
        break

# If still not found, try to find it by searching
if not backend_dir:
    # Search from script directory up to 3 levels
    search_dir = script_dir
    for _ in range(3):
        potential = search_dir / "backend"
        if potential.exists() and (potential / "services").exists():
            backend_dir = potential
            break
        search_dir = search_dir.parent

# Verify backend directory exists
if not backend_dir:
    print(f"❌ Error: Could not find backend directory")
    print(f"   Searched from: {script_dir}")
    print(f"   Tried paths:")
    for path in possible_backend_paths:
        print(f"     - {path} (exists: {path.exists()})")
    sys.exit(1)

# Add backend directory to Python path (insert at beginning for priority)
backend_path = str(backend_dir)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Load environment variables (try both locations)
env_paths = [
    script_dir / ".env",
    backend_dir / ".env",
    script_dir.parent / ".env"
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()  # Try default location

def main():
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "show sales from April"
        print(f"ℹ️  No query provided, using default: '{query}'")
        print(f"   Usage: python quick_test.py 'your query here'\n")
    
    print(f"📝 Testing query: {query}\n")
    
    # Debug: Show path info
    if os.getenv("DEBUG"):
        print(f"🔍 Debug Info:")
        print(f"   Script dir: {script_dir}")
        print(f"   Backend dir: {backend_dir}")
        print(f"   Backend exists: {backend_dir.exists()}")
        print(f"   Python path includes: {backend_path in sys.path}")
        print(f"   Services dir exists: {(backend_dir / 'services').exists()}")
        print()
    
    try:
        from services.intent_agent.agent_orchestrator import IntentAgentOrchestrator
        
        # Initialize
        print("🔧 Initializing...")
        orchestrator = IntentAgentOrchestrator(
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Process
        print("🚀 Processing...\n")
        result = orchestrator.run_intent_agent(query)
        
        # Display key results
        intent = result.get('intent_analysis', {})
        sql = result.get('generated_sql') or intent.get('generated_sql')
        
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"Intent: {intent.get('intent_type', 'N/A')}")
        print(f"Workspaces: {intent.get('workspaces', [])}")
        print(f"Confidence: {intent.get('confidence', 0):.2f}")
        
        if sql:
            print(f"\n✅ Generated SQL:")
            print(f"   {sql}")
        else:
            print(f"\n⚠️  No SQL generated")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

