# Sleep Coach LLM API Service

AI-powered Sleep Coach service with wearable data processing and LLM integration. Processes sleep data from wearables, provides personalized insights, and answers questions using a local LLM.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials (copy from `.env.example`):
```
# Required
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_database_name
LLAMA_API_URL=http://your_llama_host:11434/api/generate
LLAMA_MODEL=llama3

# Required for knowledge base (vector search)
QDRANT_URL=http://your_qdrant_host:6333
QDRANT_COLLECTION_NAME=docs
EMBEDDING_API_URL=http://your_embedding_service:8000/embed

# Optional
OPENAI_API_KEY=your_openai_key_here  # Only if using OpenAI embeddings
```

### 3. Run the Service

```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:
```bash
docker-compose up --build
```

## API Endpoints

Once running, visit **http://localhost:8000/docs** for interactive API documentation.

### Main Endpoints

- **POST `/query`** - Ask questions about sleep data
- **POST `/wearable`** - Submit wearable sleep data
- **POST `/knowledge`** - Add knowledge documents to vector DB
- **GET `/health`** - Health check
- **GET `/stats`** - Database statistics
- **GET `/trends/{user_id}`** - Get user sleep trends

## Example Usage

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "query": "How was my sleep over the last 7 days?"
  }'
```

### Submit Wearable Data

```bash
curl -X POST http://localhost:8000/wearable \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "date": "2024-11-02",
    "sleep_duration": 7.5,
    "deep_sleep": 1.2,
    "rem_sleep": 1.8,
    "light_sleep": 4.0
  }'
```

### Get User Trends

```bash
curl http://localhost:8000/trends/123?days=30
```

## Example Queries

- **Personal Data**: "How was my sleep over the last 7 days?", "What's my average sleep duration?"
- **Cohort Comparison**: "How does my deep sleep compare to others?", "What percentile is my sleep?"
- **Knowledge-Based**: "What's the importance of REM sleep?", "How can I improve sleep quality?"

## Project Structure

```
local-llm/
├── app.py              # FastAPI application
├── sleep_coach_llm.py  # Sleep Coach LLM system
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── docker-entrypoint.sh # Docker startup script
├── .env.example        # Environment variables template
├── CODE_FLOW.md        # Code flow documentation
├── frontend/           # React frontend application
└── README.md          # This file
```

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# View logs
docker logs ai-sleep-coach-query -f

# Restart
docker-compose restart ai-query-service
```

## Troubleshooting

### Database Connection
- Verify MySQL credentials in `.env`
- Check network connectivity to MySQL server

### LLM API
- Verify Llama/Ollama is running at the configured `LLAMA_API_URL`
- Test: `curl http://your-llama-host:11434/api/generate`

### Vector Database
- Verify Qdrant is accessible at `QDRANT_URL`
- Verify embedding service is accessible at `EMBEDDING_API_URL`
- Test: `curl http://your-qdrant-host:6333/collections/docs`

### Dependencies
- Ensure all packages are installed: `pip install -r requirements.txt`
- Check Python version: requires 3.8+

## License

[Your License Here]
