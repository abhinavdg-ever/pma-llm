# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (to clone the repository)
- MySQL database access (host: 62.72.57.99)
- Ollama LLM service access (host: 34.131.0.29:11434)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd LocalLLM
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp env.example .env

# Edit .env file with your credentials
# The file should contain:
MYSQL_HOST=62.72.57.99
MYSQL_USER=aabo
MYSQL_PASSWORD=3#hxFkBFKJ2Ph!$@
MYSQL_DATABASE=aaboRing10Jan
OLLAMA_API_URL=http://34.131.0.29:11434/api/generate
```

### 5. Test Connections (Optional)

```bash
python test_connection.py
```

This will verify:
- ✅ MySQL database connection
- ✅ LLM API availability

### 6. Run the Application

**Option A: Run directly**
```bash
python app.py
```

**Option B: Run with uvicorn (recommended for development)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Run with Docker**
```bash
docker-compose up --build
```

### 7. Verify Service is Running

Open your browser and visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

Or test with curl:
```bash
curl http://localhost:8000/health
```

## Testing the Service

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers are in the database?"}'
```

### Example Questions

```bash
# Count customers by risk level
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers have High Risk insomnia classification?"}'

# Get average scores
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average insomnia score?"}'

# Query by chronotype
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers are Evening Types?"}'
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Database Connection Failed

- Verify MySQL credentials in `.env`
- Check network connectivity: `ping 62.72.57.99`
- Ensure MySQL user has proper permissions

### LLM API Failed

- Verify Ollama API URL in `.env`
- Test direct API call:
  ```bash
  curl http://34.131.0.29:11434/api/version
  ```
- Check if Ollama service is running

### Import Errors

- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## File Structure

```
LocalLLM/
├── app.py              # Main application file (run this!)
├── test_connection.py  # Optional: test connections
├── requirements.txt    # Python dependencies
├── .env                # Your credentials (create from env.example)
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── README.md          # Full documentation
├── SETUP.md           # This file
├── QUICKSTART.md      # Quick start guide
└── TESTING.md         # Testing guide
```

## Next Steps

1. ✅ Service is running on http://localhost:8000
2. 📖 Visit http://localhost:8000/docs for interactive API documentation
3. 🧪 Test with sample questions using the `/query` endpoint
4. 🔧 Customize as needed for your use case

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review TESTING.md for testing examples
- Verify all connections are working with test_connection.py

