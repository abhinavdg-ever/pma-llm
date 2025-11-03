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
cp env.example .env
```

Edit `.env` with your credentials:
```
MYSQL_HOST=62.72.57.99
MYSQL_USER=aabo
MYSQL_PASSWORD=3#hxFkBFKJ2Ph!$@
MYSQL_DATABASE=aaboRing10Jan
OLLAMA_API_URL=http://34.131.0.29:11434/api/generate

# Optional: For Pinecone vector database
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=aabosleepcoach
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
LocalLLM/
├── app.py              # FastAPI application
├── sleep_coach_llm.py  # Sleep Coach LLM system
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── env.example         # Environment variables template
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
- Verify Ollama is running at `http://34.131.0.29:11434`
- Test: `curl http://34.131.0.29:11434/api/version`

### Dependencies
- Ensure all packages are installed: `pip install -r requirements.txt`
- Check Python version: requires 3.8+

## License

[Your License Here]
