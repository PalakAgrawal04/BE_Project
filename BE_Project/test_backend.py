#!/usr/bin/env python3
"""
Simple test script to test the IntelliQuery backend with text queries.

This script allows you to test the backend directly without using the frontend.
You can test:
1. Intent Agent directly (SQL generation)
2. Full API endpoint (if server is running)

Usage:
    python test_backend.py
    python test_backend.py --api  # Test via API endpoint
"""

import os
import sys
import json
import argparse
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

def test_intent_agent_direct(query: str):
    """Test the Intent Agent directly without API."""
    print(f"\n{'='*70}")
    print(f"Testing Intent Agent Directly")
    print(f"{'='*70}")
    print(f"\n📝 Query: {query}")
    print(f"\n{'─'*70}")
    
    try:
        from services.intent_agent.agent_orchestrator import IntentAgentOrchestrator
        
        # Initialize orchestrator
        print("🔧 Initializing Intent Agent...")
        orchestrator = IntentAgentOrchestrator(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            workspace_catalog_path=os.getenv("WORKSPACE_CATALOG_PATH"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH")
        )
        
        # Run intent agent
        print("🚀 Processing query...")
        result = orchestrator.run_intent_agent(query)
        
        # Display results
        print(f"\n{'='*70}")
        print("📊 RESULTS")
        print(f"{'='*70}")
        
        if 'intent_analysis' in result:
            intent = result['intent_analysis']
            
            print(f"\n✅ Intent Type: {intent.get('intent_type', 'N/A')}")
            print(f"📁 Workspaces: {intent.get('workspaces', [])}")
            print(f"🎯 Confidence: {intent.get('confidence', 0):.2f}")
            print(f"📝 Rationale: {intent.get('rationale', 'N/A')}")
            
            # Show entities
            entities = intent.get('entities', {})
            if any(entities.values()):
                print(f"\n🔍 Extracted Entities:")
                for key, values in entities.items():
                    if values:
                        print(f"   - {key}: {values}")
            
            # Show generated SQL
            generated_sql = result.get('generated_sql') or intent.get('generated_sql')
            if generated_sql:
                print(f"\n💾 Generated SQL:")
                print(f"   {generated_sql}")
            else:
                print(f"\n⚠️  No SQL generated")
        
        # Show metadata
        if 'metadata' in result:
            metadata = result['metadata']
            print(f"\n⏱️  Processing Time: {metadata.get('processing_time_seconds', 0):.3f}s")
            print(f"🕐 Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        # Show full JSON if verbose
        print(f"\n{'─'*70}")
        print("📄 Full Response (JSON):")
        print(json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_via_api(query: str, base_url: str = "http://localhost:5000"):
    """Test via API endpoint."""
    import requests
    
    print(f"\n{'='*70}")
    print(f"Testing via API Endpoint")
    print(f"{'='*70}")
    print(f"\n📝 Query: {query}")
    print(f"🌐 API URL: {base_url}/api/query")
    print(f"\n{'─'*70}")
    
    try:
        # Make API request
        response = requests.post(
            f"{base_url}/api/query",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success!")
            
            if result.get('success'):
                data = result.get('data', {})
                print(f"\n📊 Query Type: {result.get('query_type', 'N/A')}")
                print(f"⏱️  Execution Time: {result.get('execution_time', 0):.3f}s")
                
                # Show SQL if available
                if isinstance(data, dict) and 'generated_sql' in data:
                    print(f"\n💾 Generated SQL:")
                    print(f"   {data['generated_sql']}")
                
                # Show data preview
                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                    print(f"\n📈 Results: {len(results)} rows returned")
                    if results:
                        print(f"\n   First row: {json.dumps(results[0], indent=2, default=str)}")
            else:
                print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ API Error: {response.text}")
            result = response.json() if response.headers.get('content-type') == 'application/json' else {}
        
        print(f"\n{'─'*70}")
        print("📄 Full Response (JSON):")
        print(json.dumps(result, indent=2, default=str))
        
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {base_url}")
        print("   Make sure the server is running: python BE_Project/backend/app.py")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test IntelliQuery backend with text queries")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Test via API endpoint (requires server to be running)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to test (otherwise uses interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Test queries
    test_queries = [
        "show all sales",
        "show sales from April",
        "get orders above 5000",
        "list customers from Mumbai",
        "show revenue between January and March",
        "show sales from Mumbai in April above 5000"
    ]
    
    print("\n" + "="*70)
    print("🧪 IntelliQuery Backend Tester")
    print("="*70)
    
    # Get query/queries
    if args.query:
        queries = [args.query]
    else:
        print("\n📋 Available test queries:")
        for i, q in enumerate(test_queries, 1):
            print(f"   {i}. {q}")
        print(f"   {len(test_queries) + 1}. Enter custom query")
        print(f"   {len(test_queries) + 2}. Run all test queries")
        
        choice = input(f"\n👉 Select option (1-{len(test_queries) + 2}): ").strip()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(test_queries):
                queries = [test_queries[choice_num - 1]]
            elif choice_num == len(test_queries) + 1:
                custom_query = input("\n📝 Enter your query: ").strip()
                queries = [custom_query] if custom_query else []
            elif choice_num == len(test_queries) + 2:
                queries = test_queries
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    
    if not queries:
        print("❌ No queries to test")
        return
    
    # Test each query
    for i, query in enumerate(queries, 1):
        if len(queries) > 1:
            print(f"\n\n{'#'*70}")
            print(f"# Test {i}/{len(queries)}")
            print(f"{'#'*70}")
        
        if args.api:
            test_via_api(query, args.url)
        else:
            test_intent_agent_direct(query)
        
        if i < len(queries):
            input("\n⏸️  Press Enter to continue to next query...")
    
    print(f"\n\n{'='*70}")
    print("✅ Testing complete!")
    print("="*70)


if __name__ == "__main__":
    main()

