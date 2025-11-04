#!/usr/bin/env python3
"""
Startup script for IntelliQuery Intent Agent.

This script initializes and starts the Intent Agent with proper configuration
and error handling.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from dotenv import load_dotenv

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('intent_agent.log')
        ]
    )

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    print("✅ Environment variables check passed")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask',
        'google.generativeai',
        'sentence_transformers',
        'faiss',
        'spacy',
        'pydantic',
        'langdetect',
        'dateparser'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ Dependencies check passed")
    return True

def check_spacy_model():
    """Check if spaCy English model is installed."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy English model found")
        return True
    except OSError:
        print("❌ spaCy English model not found")
        print("Please install it using: python -m spacy download en_core_web_sm")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/faiss_index",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main startup function."""
    print("🚀 Starting IntelliQuery Intent Agent...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Run checks
    print("\n📋 Running startup checks...")
    
    checks = [
        ("Environment variables", check_environment),
        ("Dependencies", check_dependencies),
        ("spaCy model", check_spacy_model),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n❌ Startup checks failed. Please fix the issues above.")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    print("\n✅ All checks passed! Starting the server...")
    print("=" * 50)
    
    try:
        # Import and start the Flask app
        from backend.app import app, initialize_orchestrator
        
        # Initialize orchestrator
        print("🔧 Initializing Intent Agent orchestrator...")
        from backend.app import setup_app
        setup_app()
        
        # Start the server
        port = int(os.getenv('PORT', 5000))
        host = os.getenv('HOST', '0.0.0.0')
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print(f"🌐 Server starting on http://{host}:{port}")
        print(f"📊 Debug mode: {debug}")
        print("\n📖 API Documentation: http://localhost:5000/")
        print("🔍 Health Check: http://localhost:5000/api/health")
        print("📈 Status: http://localhost:5000/api/status")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        logger.info("Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {str(e)}")
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
