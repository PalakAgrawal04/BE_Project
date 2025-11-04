# IntelliQuery: Unified Natural Language Query Platform

A comprehensive natural language processing platform that combines structured (SQL) and unstructured (document) data querying with voice support. Features advanced intent understanding, real-time analytics, and secure query execution.

## 🚀 Features

### Core Capabilities
- **Natural Language Understanding**: Advanced intent detection and entity extraction
- **Hybrid Query Processing**: Combines SQL and document vector search
- **Voice Query Support**: Process voice commands with Whisper integration
- **Real-time Analytics**: Monitor query performance and patterns

### Technical Features
- **Safe SQL Generation**: LLM-powered SQL generation with validation
- **Vector Search**: Qdrant-powered document similarity search
- **Query Validation**: Multi-step validation pipeline for reliability
- **Performance Monitoring**: Prometheus metrics and real-time stats
- **REST API**: Clean API design with comprehensive endpoints
- **Database Integration**: MySQL for structured data, Qdrant for vectors
- **Voice Processing**: Whisper model for accurate transcription

## 📁 Project Structure

```bash
backend/
├── app.py                 # Main application entry point
├── services/
│   ├── database/         # Database connections
│   │   ├── mysql_client.py
│   │   ├── vector_client.py
│   │   ├── connection_pool.py
│   │   └── schema.sql
│   ├── intent_agent/     # NLP understanding
│   │   ├── preprocessor.py
│   │   ├── embedding_retriever.py
│   │   ├── entity_extractor.py
│   │   ├── classifier.py
│   │   ├── llm_mapper.py
│   │   └── agent_orchestrator.py
│   ├── validation_agent/ # Query validation
│   │   ├── validator.py
│   │   └── schema_validator.py
│   ├── query_executor/   # Safe query execution
│   │   └── executor.py
│   ├── metrics/         # Performance tracking
│   │   └── collector.py
│   └── voice/           # Voice processing
│       └── processor.py
├── prompts/            # LLM prompts
├── config/            # Configuration files
└── tests/            # Test suites
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- MySQL Server
- Qdrant vector database
- pip or conda package manager

### Step 1: Database Setup

```bash
# Install MySQL
sudo apt install mysql-server  # Ubuntu/Debian
sudo systemctl start mysql

# Install Qdrant (using Docker)
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

# Create MySQL database and user
mysql -u root -p
mysql> CREATE DATABASE intelliquery;
mysql> CREATE USER 'intelliquery'@'localhost' IDENTIFIED BY 'your_password';
mysql> GRANT SELECT ON intelliquery.* TO 'intelliquery'@'localhost';
```

### Step 2: Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd intelliquery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Required Models

```bash
# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your settings:
# - Database credentials
# - API keys (Gemini)
# - Service ports
```

### Step 5: Initialize Database

```bash
# Create tables
python -m backend.services.database.init_db

## 🚀 Quick Start

```bash
# Start backend server
python backend/app.py
```

The API server will start on `http://localhost:5000`

## 📖 API Documentation

### Process Natural Language Query

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me sales in Mumbai for last month",
    "include_similar_documents": true
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sql_results": [
      {
        "date": "2024-10-15",
        "location": "Mumbai",
        "amount": 50000,
        "product": "Widget X"
      }
    ],
    "similar_documents": [
      {
        "document_id": "doc123",
        "score": 0.95,
        "metadata": {
          "title": "Mumbai Sales Report",
          "content": "Monthly analysis..."
        }
      }
    ],
    "intent_analysis": {
      "intent_type": "read",
      "entities": {
        "location": "Mumbai",
        "date_range": "last month"
      }
    },
    "generated_sql": "SELECT * FROM sales WHERE location='Mumbai' AND date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"
  },
  "execution_time": 0.245,
  "query_type": "mixed",
  "timestamp": "2024-11-04T10:30:00Z"
}
```

### Process Voice Query

```bash
curl -X POST http://localhost:5000/api/voice \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_audio_data",
    "format": "wav",
    "include_similar_documents": true
  }'
```

### Get Performance Metrics

```bash
curl http://localhost:5000/api/metrics
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "query_time": {
      "avg": 0.245,
      "p95": 0.500,
      "max": 0.750
    },
    "phase_intent_extraction": {
      "avg": 0.100,
      "p95": 0.200
    },
    "phase_sql_execution": {
      "avg": 0.050,
      "p95": 0.100
    }
  }
}
```

### Check Health Status

```bash
curl http://localhost:5000/api/health
```

## 🧪 Testing

Run the test suites:

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run with coverage
python -m pytest backend/tests/ --cov=backend --cov-report=html

# Run specific module tests
python -m pytest backend/tests/test_query_executor.py -v

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Generative AI API key | Required |
| `EMBEDDING_MODEL` | Sentence-BERT model name | `all-MiniLM-L6-v2` |
| `FAISS_INDEX_PATH` | Path to FAISS index | `data/faiss_index/index.faiss` |
| `WORKSPACE_CATALOG_PATH` | Workspace catalog JSON path | `backend/services/intent_agent/config/workspace_catalog.json` |
| `SPACY_MODEL` | spaCy model name | `en_core_web_sm` |
| `PORT` | Flask server port | `5000` |
| `DEBUG` | Debug mode | `False` |

### Workspace Configuration

Edit `backend/services/intent_agent/config/workspace_catalog.json` to customize workspaces:

```json
[
  {
    "id": "sales",
    "name": "Sales Analytics",
    "description": "Sales and transaction data, revenue analysis",
    "keywords": ["sales", "revenue", "profit", "customer", "order"],
    "tables": ["sales_transactions", "products", "customers"]
  }
]
```

## 🏗️ Architecture

### Pipeline Flow

1. **Preprocessing**: Text normalization and language detection
2. **Embedding Generation**: Sentence-BERT embeddings
3. **Similarity Retrieval**: FAISS-based context retrieval
4. **Entity Extraction**: spaCy NER + custom patterns
5. **Classification**: Intent type and workspace prediction
6. **LLM Mapping**: Gemini-powered intent synthesis
7. **Schema Validation**: Pydantic validation and repair
8. **Response Generation**: Structured JSON output

### Components

- **TextPreprocessor**: Handles text cleaning and normalization
- **EmbeddingRetriever**: Manages embeddings and similarity search
- **EntityExtractor**: Extracts structured entities using spaCy
- **IntentClassifier**: Classifies intent types and workspaces
- **LLMMapper**: Uses Gemini for intelligent intent mapping
- **SchemaValidator**: Validates and repairs intent schemas
- **IntentAgentOrchestrator**: Orchestrates the complete pipeline

## 🔍 Supported Intent Types

- **read**: Simple data retrieval ("show me", "get", "find")
- **compare**: Comparative analysis ("compare", "vs", "difference")
- **update**: Data modification ("change", "update", "modify")
- **summarize**: Aggregation ("total", "average", "summary")
- **analyze**: Complex analysis ("trend", "pattern", "analysis")
- **predict**: Predictive queries ("forecast", "predict", "estimate")

## 🎯 Supported Workspaces

- **sales**: Sales and transaction data
- **support**: Customer feedback and complaints
- **marketing**: Campaign performance and metrics
- **hr**: Employee data and performance
- **finance**: Financial reports and budgets
- **operations**: Inventory and logistics

## 🐛 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Gemini API Key Missing**
   - Set `GEMINI_API_KEY` in your `.env` file
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. **FAISS Index Issues**
   - Delete `data/faiss_index/` folder to regenerate
   - The system will create a new index automatically

4. **Memory Issues**
   - Reduce batch sizes in embedding generation
   - Use `faiss-cpu` instead of `faiss-gpu` if needed

### Logging

Check logs for detailed error information:
```bash
tail -f intent_agent.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google Generative AI](https://ai.google.dev/) for Gemini API
- [spaCy](https://spacy.io/) for NLP processing
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases for usage examples
3. Open an issue on GitHub
4. Check the API documentation at `http://localhost:5000/`

---

**Built with ❤️ for IntelliQuery**
"# BE_Project" 
