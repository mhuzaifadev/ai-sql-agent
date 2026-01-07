# SQL Compiler Agent

**Transform natural language questions into safe, executable SQL queries using AI**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Why This Project Exists

Many people need to query databases but don't know SQL. Writing SQL queries can be difficult, time-consuming, and error-prone. This project solves that problem by letting you ask questions in plain English and automatically generating safe, correct SQL queries.

**Our Goal:** Make database access easy for everyone while keeping your data safe and secure.

## ✨ What It Does

SQL Compiler Agent is an intelligent system that:

- **Understands your questions** - Ask in natural language like "How many users signed up last month?"
- **Generates SQL automatically** - Creates correct PostgreSQL queries for you
- **Keeps data safe** - Only allows read-only queries, blocks dangerous operations
- **Fixes mistakes** - Automatically corrects errors and retries up to 3 times
- **Works with any database** - Connects to your PostgreSQL database securely

## 🚀 Key Features

### 🔒 Security First
- **Read-only access** - Only SELECT queries allowed, no data modification
- **SQL injection protection** - Blocks dangerous patterns automatically
- **Keyword filtering** - Prevents DROP, DELETE, and other risky operations
- **Connection pooling** - Efficient and secure database connections

### 🧠 Smart AI Integration
- **Multiple AI models** - Supports OpenAI GPT and Google Gemini
- **Automatic fallback** - Switches models if one fails
- **Self-correction** - Learns from errors and fixes queries automatically
- **Schema understanding** - Uses RAG to find relevant database tables

### 📊 Production Ready
- **REST API** - Easy to integrate with any application
- **Rate limiting** - Protects your system from overload
- **Monitoring** - Track queries, errors, and performance
- **Circuit breaker** - Prevents cascading failures
- **Comprehensive logging** - Debug and monitor everything

## 📦 Quick Start

### Prerequisites

- Python 3.12 or higher
- PostgreSQL database
- OpenAI API key OR Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-sql-agent.git
   cd ai-sql-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys and database credentials
   ```

5. **Configure your database**
   ```bash
   # Run the setup script on your PostgreSQL database
   psql -U postgres -d your_database -f setup_database.sql
   ```

6. **Start the API server**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with these settings:

```bash
# AI Model Configuration
DEFAULT_MODEL=gemini  # or 'openai'
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
LLM_TEMPERATURE=0

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=ai_agent_user
DB_PASSWORD=secure_password

# API Configuration
HOST=0.0.0.0
PORT=8000
RATE_LIMIT=10/minute

# Logging
LOG_LEVEL=INFO
JSON_LOGS=false
```

### Database Setup

The project includes a SQL script to create a read-only database user:

```sql
-- Run this on your PostgreSQL database
psql -U postgres -d your_database -f setup_database.sql
```

This creates a safe, read-only user that can only execute SELECT queries.

## 💻 Usage Examples

### Using the REST API

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many users are in the database?"
  }'
```

**Response:**
```json
{
  "success": true,
  "sql_query": "SELECT COUNT(*) FROM users LIMIT 100;",
  "explanation": "Counts all users in the database",
  "execution_result": {
    "success": true,
    "row_count": 1,
    "data": [{"count": 42}]
  },
  "retries": 0,
  "execution_time": 1.23
}
```

**Complex question:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me all orders from customers in New York with total amount over $100"
  }'
```

### Using Python Code

```python
from src.pipeline.graph import create_graph
from src.database.connection import DatabaseConnection

# Connect to database
db = DatabaseConnection()
db.connect()

# Create the agent
agent = create_graph(db)

# Ask a question
result = agent.invoke("How many products do we have?")

# Check results
if result.get("execution_result"):
    print(result["execution_result"]["data"])
```

## 📡 API Endpoints

### `POST /query`

Convert natural language to SQL and execute.

**Request:**
```json
{
  "question": "Your question here",
  "max_retries": 3  // optional
}
```

**Response:**
```json
{
  "success": true,
  "sql_query": "SELECT ...",
  "explanation": "...",
  "execution_result": {...},
  "retries": 0,
  "execution_time": 1.23
}
```

### `GET /health`

Check system health and get basic metrics.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "metrics": {
    "query_executions": 150,
    "success_rate": 95.5,
    "average_response_time": 2.3
  }
}
```

### `GET /metrics`

Get detailed application metrics.

**Response:**
```json
{
  "metrics": {
    "query_executions": 150,
    "successful_queries": 143,
    "failed_queries": 7,
    "average_retries": 0.5,
    "llm_calls": 200,
    "average_tokens_per_call": 1500
  }
}
```

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI.

## 🏗️ How It Works

### Architecture Overview

```
User Question
    ↓
[Schema Router] → Finds relevant database tables using AI
    ↓
[SQL Generator] → Creates SQL query using LLM
    ↓
[Validator] → Checks safety and syntax
    ↓
[Executor] → Runs query on database
    ↓
Results → Returns data to user
```

### Key Components

1. **Schema Router** - Uses RAG (Retrieval-Augmented Generation) to find relevant tables
2. **SQL Generator** - AI model converts questions to SQL
3. **Validator** - Checks for security issues and syntax errors
4. **Executor** - Safely runs queries on your database
5. **Reflection** - Automatically fixes errors and retries

### Self-Correction Flow

If a query fails:
1. System analyzes the error
2. AI generates a corrected query
3. Validates the new query
4. Retries execution
5. Repeats up to 3 times if needed

## 🔐 Security Features

### What We Protect Against

- ✅ **SQL Injection** - Blocks dangerous patterns
- ✅ **Data Modification** - Only SELECT queries allowed
- ✅ **Schema Changes** - Blocks DROP, ALTER, CREATE
- ✅ **Unauthorized Access** - Read-only database user
- ✅ **Rate Limiting** - Prevents abuse
- ✅ **Circuit Breaker** - Prevents cascading failures

### Security Best Practices

1. **Use read-only database user** - Never use admin credentials
2. **Keep API keys secret** - Never commit `.env` file
3. **Enable rate limiting** - Protect against abuse
4. **Monitor logs** - Watch for suspicious activity
5. **Regular updates** - Keep dependencies updated

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_security.py -v      # Security tests
pytest tests/test_validators.py -v    # Validator tests
pytest tests/test_pipeline.py -v     # Pipeline tests

# Run with coverage
pytest --cov=src tests/
```

## 📊 Monitoring

### Metrics Available

- Query execution count and success rate
- Average response time
- Retry statistics
- LLM token usage
- Error types and frequencies

### Logging

Logs are structured and can be output as JSON:

```bash
# Enable JSON logs
export JSON_LOGS=true
python main.py
```

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## 🛠️ Development

### Project Structure

```
ai-sql-agent/
├── src/
│   ├── models/          # AI model factory and schemas
│   ├── pipeline/         # LangGraph pipeline
│   ├── database/         # Database connection
│   ├── validators/       # SQL validation and safety
│   └── utils/            # Logging and error handling
├── tests/                # Test suite
├── main.py               # FastAPI application
└── requirements.txt      # Dependencies
```

### Running in Development

```bash
# Enable auto-reload
export RELOAD=true
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🐛 Troubleshooting

### Common Issues

**"Database connection failed"**
- Check database credentials in `.env`
- Verify database is running
- Check network connectivity

**"API key not found"**
- Ensure `.env` file exists
- Verify API key is correct
- Check for typos in variable names

**"Circuit breaker is open"**
- Too many failures occurred
- Wait 60 seconds and try again
- Check LLM API status

**"Rate limit exceeded"**
- Too many requests
- Wait before trying again
- Adjust `RATE_LIMIT` in `.env`

## 📈 Performance

### Typical Response Times

- Simple queries: 1-3 seconds
- Complex queries: 3-5 seconds
- With retries: 5-10 seconds

### Optimization Tips

1. **Use connection pooling** - Already enabled by default
2. **Cache frequent queries** - Implement caching layer
3. **Limit schema size** - Reduce number of tables
4. **Use faster AI models** - GPT-4o-mini is faster than GPT-4

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **Pytest** for testing

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [FastAPI](https://fastapi.tiangolo.com/) for the API
- PostgreSQL for database support

## 📞 Support

- **Documentation**: Check `/docs` endpoint for API documentation
- **Issues**: Open an issue on GitHub
- **Questions**: Check the troubleshooting section above

## 🎯 Roadmap

Future improvements:
- [ ] Support for MySQL and SQLite
- [ ] Query result visualization
- [ ] Natural language result explanations
- [ ] Multi-database support
- [ ] Web UI for non-technical users

Open for contributions 

---

**Made with ❤️ for making databases accessible to everyone**
