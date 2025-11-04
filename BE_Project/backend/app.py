"""
Flask application for IntelliQuery with unified query processing.
Integrates ML models, database operations, and analytics.
"""

import os
import json
import logging
import base64
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Services
from services.intent_agent.agent_orchestrator import IntentAgentOrchestrator
from services.validation_agent.validator import run_validation_agent
from services.database import MySQLClient, VectorClient
from services.query_executor import QueryExecutor
from services.metrics import MetricsCollector
from services.voice import VoiceProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'App Request Count',
                       ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds',
                           'Request latency in seconds',
                           ['method', 'endpoint'])

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Add Prometheus WSGI middleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Global service instances
orchestrator = None
mysql_client = None
vector_client = None
query_executor = None
metrics_collector = None
voice_processor = None


def setup_services():
    """Initialize all service components."""
    global orchestrator, mysql_client, vector_client, query_executor, \
           metrics_collector, voice_processor
           
    try:
        # Initialize metrics first for monitoring other services
        metrics_collector = MetricsCollector()
        
        # Initialize database clients
        mysql_client = MySQLClient(metrics_collector)
        vector_client = VectorClient(metrics_collector=metrics_collector)
        
        # Initialize ML components
        orchestrator = IntentAgentOrchestrator(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            workspace_catalog_path=os.getenv("WORKSPACE_CATALOG_PATH"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH")
        )
        
        # Initialize query executor
        query_executor = QueryExecutor(
            mysql_client=mysql_client,
            vector_client=vector_client,
            intent_agent=orchestrator,
            metrics=metrics_collector
        )
        
        # Initialize voice processor
        voice_processor = VoiceProcessor()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise

@app.before_first_request
def before_first_request():
    """Initialize services before first request."""
    setup_services()


# ==========================================================
# Core Query Processing Endpoints
# ==========================================================
@app.route('/api/query', methods=['POST'])
def process_query():
    """
    Process a natural language query through the full pipeline.
    """
    start_time = datetime.now()
    
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        query = data.get('query', '').strip()
        include_similar = data.get('include_similar_documents', True)

        if not query:
            return jsonify({'error': 'Query field is required'}), 400

        logger.info(f"Processing query: {query[:80]}...")

        # Run validation first
        validation_result = run_validation_agent(query)
        if not validation_result.get("is_valid", True):
            return jsonify({
                "success": False,
                "status": "invalid_query",
                "validation": validation_result,
                "timestamp": datetime.now().isoformat()
            }), 400

        # Execute query through full pipeline
        result = query_executor.execute_query(
            query,
            include_similar_docs=include_similar
        )

        # Track request metrics
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/api/query',
            status=200 if result.success else 400
        ).inc()

        return jsonify({
            "success": result.success,
            "data": result.data,
            "execution_time": result.execution_time,
            "query_type": result.query_type,
            "timestamp": result.timestamp
        }), 200 if result.success else 400

    except Exception as e:
        logger.exception("Query processing failed")
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/api/query',
            status=500
        ).inc()
        
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

    finally:
        REQUEST_LATENCY.labels(
            method='POST',
            endpoint='/api/query'
        ).observe((datetime.now() - start_time).total_seconds())

@app.route('/api/voice', methods=['POST'])
def process_voice():
    """
    Process voice query - transcribe audio and run through query pipeline.
    """
    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'Audio data required'}), 400
            
        audio_format = data.get('format', 'wav')
        include_similar = data.get('include_similar_documents', True)

        # Process audio to text
        success, text, error = voice_processor.process_audio(
            data['audio'],
            format=audio_format
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': error or 'Voice processing failed'
            }), 400

        # Process transcribed text as normal query
        result = query_executor.execute_query(
            text,
            include_similar_docs=include_similar
        )

        return jsonify({
            'success': result.success,
            'transcribed_text': text,
            'data': result.data,
            'execution_time': result.execution_time,
            'timestamp': result.timestamp
        }), 200 if result.success else 400

    except Exception as e:
        logger.exception("Voice processing failed")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
# ==========================================================
# Monitoring & Analytics Endpoints
# ==========================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current performance metrics."""
    try:
        return jsonify({
            'success': True,
            'metrics': metrics_collector.get_all_stats(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.exception("Failed to retrieve metrics")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Basic health check endpoint."""
    try:
        # Check all critical services
        health = {
            'database': mysql_client.get_connection() is not None,
            'vector_db': vector_client is not None,
            'intent_agent': orchestrator is not None,
            'metrics': metrics_collector is not None
        }
        
        status = 200 if all(health.values()) else 503
        return jsonify({
            'status': 'healthy' if status == 200 else 'degraded',
            'services': health,
            'timestamp': datetime.now().isoformat()
        }), status
        
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

if __name__ == '__main__':
    # Load config from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting IntelliQuery API on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
